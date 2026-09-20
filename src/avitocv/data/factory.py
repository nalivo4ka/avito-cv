"""Сборка датасетов по конфигу.

Единственное место, где перечислены конкретные реализации абстракций пайплайна, — остальной код
зависит только от интерфейсов. Здесь же задаётся разница между обучением и валидацией: сид
и стратегия выбора окна по ширине.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from avitocv.config import DataConfig
from avitocv.data.synthesis.backgrounds import WeightedBackgroundMixture
from avitocv.data.synthesis.composition import CropComposer, CropLayoutSampler, FittedLineRenderer
from avitocv.data.synthesis.corpus import LineFileSource, PatternSource
from avitocv.data.datasets import (
    CenterWindowFit,
    CombinedOrientationDataset,
    DatasetShare,
    HorizontalWindowFit,
    ImagePreprocessor,
    OrientationDataset,
    PreprocessConfig,
    RandomWindowFit,
)
from avitocv.data.degradation import DegradationPipeline
from avitocv.data.synthesis.fonts import FontRegistry, SupportedTextFilter
from avitocv.data.manifest import CropManifest
from avitocv.data.matching.profile import CropProfile
from avitocv.data.synthesis.rendering import LetterCaseSampler, LineStyleSampler, TextLineRenderer
from avitocv.data.sampling import SeedScheme
from avitocv.data.synthesis.writing_systems import Script
from avitocv.data.storage import CropIndex
from avitocv.data.sources import (
    GeometryMatchedResampler,
    ManifestTextLineSource,
    MaterializedTextLineSource,
    ResampledTextLineSource,
    ScriptedTextSource,
    SyntheticComponents,
    SyntheticSourceConfig,
    SyntheticTextLineSource,
    TextLineSource,
)


class OrientationDatasetAssembler:
    """Собирает `OrientationDataset` поверх любого источника кропов."""

    # Деградации работают вдвое выше входа сети: мельче этого после ужатия неразличимо,
    # а на крупных кропах разница в стоимости десятикратная. Множитель, а не константа, —
    # чтобы при смене высоты входа порог менялся сам: при входе 48 он равен 96.
    WORK_HEIGHT_FACTOR = 2

    def __init__(self, preprocess: PreprocessConfig) -> None:
        self._preprocess = preprocess
        self._work_height = preprocess.height * self.WORK_HEIGHT_FACTOR

    @property
    def work_height(self) -> int:
        return self._work_height

    def assemble(
        self,
        source: TextLineSource,
        seed: int,
        window_fit: HorizontalWindowFit,
        codec_stage: DegradationPipeline | None = None,
    ) -> OrientationDataset:
        return OrientationDataset(
            source=source,
            codec_stage=codec_stage or DegradationPipeline.build_codec_stage(self._work_height),
            preprocessor=ImagePreprocessor(self._preprocess, window_fit),
            seed_scheme=SeedScheme(seed),
        )


class SyntheticDatasetFactory:
    """Строит генерацию на лету: шрифты, корпуса, профиль, деградации."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._assembler = OrientationDatasetAssembler(config.preprocess)
        self._components = self._build_components()

    @property
    def components(self) -> SyntheticComponents:
        return self._components

    def build_source(self, length: int) -> SyntheticTextLineSource:
        return SyntheticTextLineSource(self._components, SyntheticSourceConfig(virtual_length=length))

    def build_training(self) -> OrientationDataset:
        source = self.build_source(self._config.train_length)
        return self._assembler.assemble(source, self._config.train_seed, RandomWindowFit())

    def build_validation(self) -> OrientationDataset:
        source = self.build_source(self._config.validation_length)
        return self._assembler.assemble(source, self._config.validation_seed, CenterWindowFit())

    def _build_components(self) -> SyntheticComponents:
        paths = self._config.paths
        registry = FontRegistry.load_index(paths.font_index)
        return SyntheticComponents(
            text_sources=self._build_text_sources(),
            style_sampler=LineStyleSampler(registry=registry),
            line_renderer=FittedLineRenderer(TextLineRenderer(), LetterCaseSampler(), SupportedTextFilter()),
            layout_sampler=CropLayoutSampler(),
            composer=CropComposer(WeightedBackgroundMixture.build_default(paths.background_dir)),
            profile=CropProfile.load(paths.test_profile),
            capture_stage=DegradationPipeline.build_capture_stage(),
        )

    def _build_text_sources(self) -> tuple[ScriptedTextSource, ...]:
        paths = self._config.paths
        weights = self._config.weights
        return (
            ScriptedTextSource(LineFileSource(paths.russian_corpus), Script.CYRILLIC, weights.russian),
            ScriptedTextSource(LineFileSource(paths.english_corpus), Script.LATIN, weights.english),
            ScriptedTextSource(PatternSource(), Script.LATIN, weights.patterns),
        )


@dataclass(frozen=True)
class StoreShare:
    """Хранилище кропов, сколько взять за эпоху и настоящие ли это фотографии."""

    directory: Path
    sample_count: int
    is_real: bool = False
    is_height_matched: bool = False
    # На сколько эпох вперёд строится таблица пересэмплинга. Единица означает, что все эпохи
    # увидят один и тот же набор кропов, — см. `GeometryMatchedResampler.build`.
    epoch_span: int = 1


class MaterializedDatasetFactory:
    """Строит датасет поверх заранее разложенных по шардам кропов."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._assembler = OrientationDatasetAssembler(config.preprocess)

    def build_training(self, directory: Path, length: int | None = None) -> OrientationDataset:
        source = MaterializedTextLineSource(directory, length)
        return self._assembler.assemble(source, self._config.train_seed, RandomWindowFit())

    def build_validation(self, directory: Path) -> OrientationDataset:
        source = MaterializedTextLineSource(directory)
        return self._assembler.assemble(source, self._config.validation_seed, CenterWindowFit())

    def build_mixed_training(self, shares: Sequence[StoreShare]) -> CombinedOrientationDataset:
        """Обучение из нескольких хранилищ сразу: синтетика плюс реальные фотографии.

        Доли задаются числом кропов за эпоху, а не пропорцией: хранилища разного размера, и при
        пропорции было бы неочевидно, сколько раз за эпоху повторится меньшее из них.

        Каждое хранилище получает свою стадию деградаций — см. `DatasetShare`.
        """
        return CombinedOrientationDataset([
            DatasetShare(self._build_part(share), share.sample_count) for share in shares
        ])

    def _build_part(self, share: StoreShare) -> OrientationDataset:
        codec = (
            DegradationPipeline.build_real_training_stage(self._assembler.work_height)
            if share.is_real
            else DegradationPipeline.build_codec_stage(self._assembler.work_height)
        )
        source = self._build_source(share)
        return self._assembler.assemble(source, self._config.train_seed, RandomWindowFit(), codec_stage=codec)

    def _build_source(self, share: StoreShare):
        source = MaterializedTextLineSource(share.directory)
        if not share.is_height_matched:
            return source
        frame = CropIndex.load(share.directory).frame
        heights = frame["height"].to_numpy()
        aspects = (frame["width"] / frame["height"]).to_numpy()
        reference = CropProfile.load(self._config.paths.test_profile)
        indices = GeometryMatchedResampler().build(
            heights, aspects, reference.geometry, share.sample_count,
            table_size=share.sample_count * max(share.epoch_span, 1),
        )
        return ResampledTextLineSource(source, indices)


class ManifestDatasetFactory:
    """Строит датасет поверх реальных кропов, перечисленных в манифесте."""

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._assembler = OrientationDatasetAssembler(config.preprocess)

    def build_validation(
        self,
        manifest_path: Path,
        slice_name: str | None = None,
        root: Path = Path("."),
    ) -> OrientationDataset:
        manifest = CropManifest.load(manifest_path)
        if slice_name is not None:
            manifest = manifest.take_slice(slice_name)
        source = ManifestTextLineSource.from_manifest(manifest, root)
        return self._assembler.assemble(
            source,
            self._config.validation_seed,
            CenterWindowFit(),
            codec_stage=DegradationPipeline.build_real_codec_stage(self._assembler.work_height),
        )
