from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from avitocv.data.synthesis.composition import CropLayoutSampler
from avitocv.data.degradation import ArcDegradation, DegradationPipeline
from avitocv.data.manifest import CropManifest, CropRecord
from avitocv.data.sampling import SeedScheme, ValueRange
from avitocv.data.datasets import (
    CenterWindowFit,
    CombinedOrientationDataset,
    DatasetShare,
    ImagePreprocessor,
    OrientationDataset,
    PreprocessConfig,
)
from avitocv.data.matching.profile import EmpiricalDistribution, GeometryReference
from avitocv.data.sources import (
    GeometryMatchedResampler,
    ManifestTextLineSource,
    ResampledTextLineSource,
    TextLineSource,
)
from tests.conftest import WORK_HEIGHT

CROP_SHAPE = (48, 300, 3)
SAMPLE_COUNT = 400


def _striped_crop() -> np.ndarray:
    """Кроп с горизонтальной полосой: по её форме видно и поворот, и изгиб."""
    pixels = np.full(CROP_SHAPE, 240, dtype=np.uint8)
    pixels[20:28, :] = 20
    return pixels


class _ConstantSource(TextLineSource):
    def __init__(self, marker: int, length: int) -> None:
        self._marker = marker
        self._length = length

    def __len__(self) -> int:
        return self._length

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        return Image.new("RGB", (40 + index % 3, 20), (self._marker, self._marker, self._marker))

    def weight_of(self, index: int) -> float:
        return float(self._marker)


class TestTiltInComposition:
    """Наклон живёт в композиторе: детектор обводит уже наклонённый текст."""

    def test_tilt_is_sampled_symmetrically(self) -> None:
        """Постоянный знак наклона сам стал бы признаком ориентации."""
        sampler = CropLayoutSampler()
        tilts = [sampler.sample(np.random.default_rng(seed), 3.0).tilt_degrees for seed in range(600)]
        assert abs(np.mean(tilts)) < 1.0

    def test_small_tilts_dominate_but_the_tail_exists(self) -> None:
        """Наклон берётся при коротком кропе: у длинного он ограничен раздуванием бокса."""
        sampler = CropLayoutSampler()
        tilts = np.abs([sampler.sample(np.random.default_rng(seed), target_aspect=2.5).tilt_degrees
                        for seed in range(600)])
        assert np.median(tilts) < 6.0
        assert tilts.max() > 12.0

    def test_long_lines_are_tilted_less_than_short_ones(self) -> None:
        """Ключевое свойство: бокс длинной строки раздувается пропорционально её длине.

        Без этого ограничения строка с aspect 12 при наклоне 10 градусов давала бокс втрое выше
        себя, и от текста оставалась тонкая диагональ — таких кропов в реальных данных нет.
        """
        sampler = CropLayoutSampler()
        short = np.abs([sampler.sample(np.random.default_rng(seed), 2.5).tilt_degrees for seed in range(400)])
        long = np.abs([sampler.sample(np.random.default_rng(seed), 14.0).tilt_degrees for seed in range(400)])
        assert long.max() < short.max() / 2

    def test_tilt_ceiling_falls_as_the_line_gets_longer(self) -> None:
        sampler = CropLayoutSampler()
        ceilings = [sampler._tilt_ceiling(aspect) for aspect in (2, 5, 8, 12, 20)]
        assert ceilings == sorted(ceilings, reverse=True)

    def test_tilt_never_exceeds_the_limit(self) -> None:
        sampler = CropLayoutSampler()
        tilts = np.abs([sampler.sample(np.random.default_rng(seed), 2.0).tilt_degrees for seed in range(600)])
        assert tilts.max() <= sampler.max_tilt_degrees

    def test_degradations_no_longer_rotate(self) -> None:
        """Механизм наклона должен быть один, иначе бокс раздуется дважды."""
        kinds = {type(step.degradation).__name__ for step in DegradationPipeline.build_capture_stage()._steps}
        assert "RotationDegradation" not in kinds


class TestArcDegradation:
    """Текст на округлых предметах идёт дугой, прямых строк там не бывает."""

    def test_geometry_is_preserved(self) -> None:
        result = ArcDegradation().apply(_striped_crop(), np.random.default_rng(0))
        assert result.shape == CROP_SHAPE and result.dtype == np.uint8

    def test_middle_of_the_stripe_moves_relative_to_the_edges(self) -> None:
        crop = _striped_crop()
        curved = ArcDegradation(ValueRange(0.30, 0.30)).apply(crop, np.random.default_rng(0))
        middle_row = float(np.argmin(curved[:, crop.shape[1] // 2].mean(axis=-1)))
        edge_row = float(np.argmin(curved[:, 5].mean(axis=-1)))
        assert abs(middle_row - edge_row) > 2.0, f"изгиба не видно: {middle_row} против {edge_row}"

    def test_curvature_sign_is_balanced(self) -> None:
        """Постоянная выпуклость выдавала бы ориентацию не хуже самого текста."""
        crop = _striped_crop()
        middle_rows = []
        for seed in range(200):
            curved = ArcDegradation().apply(crop, np.random.default_rng(seed))
            middle_rows.append(float(np.argmin(curved[:, crop.shape[1] // 2].mean(axis=-1))))
        original = float(np.argmin(crop[:, crop.shape[1] // 2].mean(axis=-1)))
        assert abs(np.mean(middle_rows) - original) < 1.5

    def test_tiny_crops_are_left_alone(self) -> None:
        tiny = np.full((3, 3, 3), 128, dtype=np.uint8)
        assert np.array_equal(ArcDegradation().apply(tiny, np.random.default_rng(0)), tiny)


class TestCaptureStageIncludesNewDistortions:
    def test_arc_is_part_of_the_capture_stage(self) -> None:
        kinds = {type(step.degradation).__name__ for step in DegradationPipeline.build_capture_stage()._steps}
        assert "ArcDegradation" in kinds

    def test_codec_stage_has_no_geometry_distortions(self) -> None:
        """Геометрия обязана меняться до поворота, иначе она станет утечкой метки."""
        kinds = {type(step.degradation).__name__ for step in DegradationPipeline.build_codec_stage(WORK_HEIGHT)._steps}
        assert not {"RotationDegradation", "ArcDegradation", "PerspectiveDegradation"} & kinds


class TestCombinedDataset:
    """Смешивание живёт на уровне датасета, чтобы у частей были свои деградации."""

    def _part(self, marker: int, length: int) -> OrientationDataset:
        return OrientationDataset(
            source=_ConstantSource(marker, length),
            codec_stage=DegradationPipeline([]),
            preprocessor=ImagePreprocessor(PreprocessConfig(), CenterWindowFit()),
            seed_scheme=SeedScheme(0),
        )

    def test_length_is_the_sum_of_shares(self) -> None:
        combined = CombinedOrientationDataset([
            DatasetShare(self._part(10, 100), 30),
            DatasetShare(self._part(200, 50), 70),
        ])
        assert len(combined) == 100

    def test_index_ranges_map_to_the_right_part(self) -> None:
        combined = CombinedOrientationDataset([
            DatasetShare(self._part(10, 100), 30),
            DatasetShare(self._part(200, 50), 70),
        ])
        first, _ = combined.load_crop(0)
        boundary, _ = combined.load_crop(29)
        second, _ = combined.load_crop(30)
        assert np.asarray(first).mean() == pytest.approx(10.0)
        assert np.asarray(boundary).mean() == pytest.approx(10.0)
        assert np.asarray(second).mean() == pytest.approx(200.0)

    def test_small_part_repeats_when_asked_for_more(self) -> None:
        combined = CombinedOrientationDataset([DatasetShare(self._part(7, 4), 12)])
        sizes = [combined.load_crop(index)[0].size for index in range(12)]
        assert sizes[:4] == sizes[4:8] == sizes[8:]

    def test_epochs_walk_through_the_whole_store(self) -> None:
        """Если из части просят меньше, чем в ней есть, окно обязано сдвигаться по эпохам.

        Иначе датасет на 600 тысяч кропов при запросе 300 тысяч показывал бы всегда первую
        половину, а вторую не показал бы никогда.
        """
        combined = CombinedOrientationDataset([DatasetShare(self._part(10, 100), 40)])
        touched = set()
        for epoch in range(3):
            combined.set_epoch(epoch)
            touched.update(combined._locate(position)[1] for position in range(40))
        assert len(touched) == 100, f"за три эпохи покрыто {len(touched)} из 100"

    def test_window_wraps_around_the_store(self) -> None:
        combined = CombinedOrientationDataset([DatasetShare(self._part(10, 10), 4)])
        combined.set_epoch(3)
        assert [combined._locate(position)[1] for position in range(4)] == [2, 3, 4, 5]

    def test_epoch_reaches_every_part(self) -> None:
        parts = [self._part(10, 20), self._part(200, 20)]
        combined = CombinedOrientationDataset([DatasetShare(part, 10) for part in parts])
        combined.set_epoch(3)
        assert all(part._epoch == 3 for part in parts)

    def test_empty_share_list_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CombinedOrientationDataset([])

    def test_non_positive_share_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            DatasetShare(self._part(1, 10), 0)


class TestRealTrainingStage:
    """Реальные кропы не надо доводить до похожести на съёмку — они уже съёмка."""

    def test_real_stage_does_not_blur(self) -> None:
        kinds = {type(step.degradation).__name__
                 for step in DegradationPipeline.build_real_training_stage(WORK_HEIGHT)._steps}
        assert "GaussianBlurDegradation" not in kinds

    def test_synthetic_stage_does_blur(self) -> None:
        kinds = {type(step.degradation).__name__ for step in DegradationPipeline.build_codec_stage(WORK_HEIGHT)._steps}
        assert "GaussianBlurDegradation" in kinds

    def test_real_stage_keeps_more_sharpness_than_the_synthetic_one(self) -> None:
        from avitocv.data.matching.profiling import CropMeasurer

        measurer = CropMeasurer()
        crop = _striped_crop()
        original = measurer.measure(crop).sharpness
        real = [measurer.measure(DegradationPipeline.build_real_training_stage(WORK_HEIGHT).apply(
            crop, np.random.default_rng(seed))).sharpness for seed in range(16)]
        synthetic = [measurer.measure(DegradationPipeline.build_codec_stage(WORK_HEIGHT).apply(
            crop, np.random.default_rng(seed))).sharpness for seed in range(16)]
        assert np.median(real) > np.median(synthetic)
        assert np.median(real) / original > 0.5


class TestManifestMaterialization:
    """Реальные кропы должны проходить тем же путём, что и синтетические."""

    def test_manifest_source_feeds_the_materializer(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (200, 400, 3), dtype=np.uint8)).save(tmp_path / "photo.png")
        records = [
            CropRecord(image_path="photo.png", left=10, top=10 + index * 15, width=180, height=30)
            for index in range(6)
        ]
        source = ManifestTextLineSource.from_manifest(CropManifest.from_records(records), tmp_path)
        scheme = SeedScheme(0)
        crops = [source.load_upright(index, scheme.rng_for(index)) for index in range(len(records))]
        assert all(crop.width > crop.height for crop in crops)


class TestGeometryMatchedResampling:
    """Реальные данные смещены и по высоте, и по пропорциям; выравнивать надо обе величины."""

    def _profile(self, height_low, height_high, aspect_low, aspect_high, seed=0):
        from avitocv.data.matching.profile import CropProfile, EmpiricalDistribution

        rng = np.random.default_rng(seed)
        return CropProfile(
            crop_height=EmpiricalDistribution.from_samples(rng.uniform(height_low, height_high, 20_000)),
            aspect_ratio=EmpiricalDistribution.from_samples(rng.uniform(aspect_low, aspect_high, 20_000)),
            sharpness=EmpiricalDistribution.from_samples(rng.uniform(0.005, 0.1, 20_000)),
            ink_spread=EmpiricalDistribution.from_samples(rng.uniform(0.05, 0.4, 20_000)),
            grayscale_share=0.16,
            sample_size=20_000,
        )

    def test_both_medians_move_toward_the_reference(self) -> None:
        from avitocv.data.sources import GeometryMatchedResampler

        rng = np.random.default_rng(1)
        heights = rng.uniform(10, 200, 30_000)
        aspects = rng.uniform(1.2, 6.0, 30_000)
        reference = self._profile(40, 200, 4.0, 12.0)
        indices = GeometryMatchedResampler().build(heights, aspects, reference, 30_000)
        assert np.median(heights[indices]) > np.median(heights)
        assert np.median(aspects[indices]) > np.median(aspects)

    def test_short_crops_stop_dominating(self) -> None:
        """В HierText 27% строк короче aspect 2.5 против 7% в тесте — перекос надо снять."""
        from avitocv.data.sources import GeometryMatchedResampler

        rng = np.random.default_rng(2)
        aspects = np.concatenate([rng.uniform(1.1, 2.5, 20_000), rng.uniform(2.5, 12.0, 10_000)])
        heights = rng.uniform(15, 150, 30_000)
        reference = self._profile(15, 150, 3.0, 12.0)
        indices = GeometryMatchedResampler().build(heights, aspects, reference, 30_000)
        before = float(np.mean(aspects < 2.5))
        after = float(np.mean(aspects[indices] < 2.5))
        assert after < before / 2, f"коротких было {before:.1%}, стало {after:.1%}"

    def test_requested_count_is_produced(self) -> None:
        from avitocv.data.sources import GeometryMatchedResampler

        rng = np.random.default_rng(3)
        indices = GeometryMatchedResampler().build(
            rng.uniform(20, 120, 10_000), rng.uniform(2.0, 9.0, 10_000), self._profile(20, 120, 2.0, 9.0), 5_000)
        assert len(indices) == 5_000

    def test_disjoint_distributions_are_reported(self) -> None:
        from avitocv.data.sources import GeometryMatchedResampler

        rng = np.random.default_rng(4)
        reference = self._profile(500, 900, 20.0, 40.0)
        with pytest.raises(ValueError):
            GeometryMatchedResampler().build(rng.uniform(10, 20, 500), rng.uniform(1.2, 2.0, 500), reference, 100)

    def test_resampled_source_follows_the_index_table(self) -> None:
        from avitocv.data.sources import ResampledTextLineSource

        source = ResampledTextLineSource(_ConstantSource(11, 10), np.array([3, 3, 7]))
        scheme = SeedScheme(0)
        sizes = [source.load_upright(index, scheme.rng_for(index)).size for index in range(3)]
        assert len(source) == 3 and sizes[0] == sizes[1] and sizes[0] != sizes[2]

    def test_empty_index_table_is_rejected(self) -> None:
        from avitocv.data.sources import ResampledTextLineSource

        with pytest.raises(ValueError):
            ResampledTextLineSource(_ConstantSource(1, 5), np.array([], dtype=np.int64))


class TestResamplingTableSpan:
    """Таблица пересэмплинга должна покрывать весь прогон, а не одну эпоху.

    Если её длина равна спросу эпохи, сдвиг окна в `CombinedOrientationDataset` берётся по
    модулю той же величины и тождественно равен нулю: все эпохи видят один и тот же набор.
    Так из 990 тысяч реальных кропов в обучение попадало около 180 тысяч.
    """

    def _geometry(self, size: int = 4000):
        rng = np.random.default_rng(0)
        return rng.uniform(12, 90, size=size), rng.uniform(1.5, 9.0, size=size)

    def _reference(self):
        rng = np.random.default_rng(1)
        return GeometryReference(
            crop_height=EmpiricalDistribution.from_samples(rng.uniform(20, 80, size=5000)),
            aspect_ratio=EmpiricalDistribution.from_samples(rng.uniform(2.0, 8.0, size=5000)),
        )

    def test_table_length_follows_the_requested_span(self) -> None:
        heights, aspects = self._geometry()
        table = GeometryMatchedResampler().build(heights, aspects, self._reference(), 500, table_size=6000)
        assert len(table) == 6000

    def test_a_longer_table_covers_more_unique_crops(self) -> None:
        heights, aspects = self._geometry()
        resampler = GeometryMatchedResampler()
        one_epoch = resampler.build(heights, aspects, self._reference(), 500)
        whole_run = resampler.build(heights, aspects, self._reference(), 500, table_size=500 * 12)
        assert len(np.unique(whole_run)) > 2 * len(np.unique(one_epoch))

    def test_epochs_see_different_crops(self) -> None:
        heights, aspects = self._geometry()
        table = GeometryMatchedResampler().build(heights, aspects, self._reference(), 500, table_size=500 * 4)
        source = ResampledTextLineSource(_ConstantSource(10, len(heights)), table)
        dataset = OrientationDataset(
            source=source,
            codec_stage=DegradationPipeline([]),
            preprocessor=ImagePreprocessor(PreprocessConfig(), CenterWindowFit()),
            seed_scheme=SeedScheme(0),
        )
        combined = CombinedOrientationDataset([DatasetShare(dataset, 500)])
        windows = []
        for epoch in range(4):
            combined.set_epoch(epoch)
            windows.append(tuple(combined._locate(position)[1] for position in range(5)))
        assert len(set(windows)) == 4, "каждая эпоха обязана брать свой участок таблицы"

    def test_table_length_does_not_change_the_sampled_geometry(self) -> None:
        """Длина таблицы — это сколько раз тянем, а не откуда; распределение обязано совпасть.

        Потолок повторов считается от спроса эпохи. Если бы он считался от длины таблицы, то
        сама по себе длинная таблица ужесточала бы потолок и размывала выравнивание.
        """
        heights, aspects = self._geometry()
        resampler = GeometryMatchedResampler()
        short = resampler.build(heights, aspects, self._reference(), 4000)
        long_table = resampler.build(heights, aspects, self._reference(), 4000, table_size=4000 * 8)
        edges = np.linspace(1.5, 9.0, 8)
        short_shares, _ = np.histogram(aspects[short], bins=edges, density=True)
        long_shares, _ = np.histogram(aspects[long_table], bins=edges, density=True)
        assert np.abs(short_shares - long_shares).sum() * np.diff(edges)[0] / 2 < 0.05
