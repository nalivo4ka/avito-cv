from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from avitocv.data.datasets import (
    CenterWindowFit,
    ImagePreprocessor,
    Orientation,
    OrientationDataset,
    PreprocessConfig,
    RandomWindowFit,
)
from avitocv.data.degradation import DegradationPipeline
from avitocv.data.manifest import CropRecord
from avitocv.data.sampling import SeedScheme, ValueRange
from pathlib import Path

from avitocv.data.synthesis.fonts import Script
from avitocv.data.synthesis.rendering import FontRenderError
from avitocv.data.sources import ManifestTextLineSource, SyntheticTextLineSource, TextLineSource
from tests.conftest import WORK_HEIGHT

STUB_HEIGHT = 24
STUB_WIDTH = 120
DISTRIBUTION_SAMPLE_SIZE = 300
LEAKAGE_SAMPLE_SIZE = 400
MEDIAN_TOLERANCE = 0.25
TILTED_ASPECT_TOLERANCE = 0.35


class ConstantLineSource(TextLineSource):
    def __init__(self, image: Image.Image, length: int = 10_000) -> None:
        self._image = image
        self._length = length

    def __len__(self) -> int:
        return self._length

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        return self._image.copy()


def _asymmetric_image() -> Image.Image:
    pixels = np.zeros((STUB_HEIGHT, STUB_WIDTH, 3), dtype=np.uint8)
    pixels[: STUB_HEIGHT // 3, :, :] = 255
    pixels[:, : STUB_WIDTH // 4, 0] = 180
    return Image.fromarray(pixels, mode="RGB")


def _vertically_symmetric_image() -> Image.Image:
    half = np.random.default_rng(0).integers(0, 256, size=(STUB_HEIGHT // 2, STUB_WIDTH, 3), dtype=np.uint8)
    full = np.concatenate([half, np.rot90(half, 2)], axis=0)
    return Image.fromarray(full, mode="RGB")


def _build_dataset(source: TextLineSource, codec: DegradationPipeline, seed: int = 5) -> OrientationDataset:
    return OrientationDataset(
        source=source,
        codec_stage=codec,
        preprocessor=ImagePreprocessor(PreprocessConfig(), CenterWindowFit()),
        seed_scheme=SeedScheme(seed),
    )


class TestOrientationLabelling:
    def test_rotated_label_means_the_crop_is_upside_down(self) -> None:
        upright = _asymmetric_image()
        dataset = _build_dataset(ConstantLineSource(upright), DegradationPipeline([]))
        for index in range(40):
            crop, orientation = dataset.load_crop(index)
            expected = upright if orientation is Orientation.UPRIGHT else upright.transpose(Image.ROTATE_180)
            assert np.array_equal(np.asarray(crop), np.asarray(expected))

    def test_labels_are_balanced(self) -> None:
        dataset = _build_dataset(ConstantLineSource(_asymmetric_image()), DegradationPipeline([]))
        labels = [int(dataset[index][1].item()) for index in range(2000)]
        assert abs(np.mean(labels) - 0.5) < 0.04

    def test_a_symmetric_crop_is_identical_for_both_labels(self) -> None:
        symmetric = _vertically_symmetric_image()
        dataset = _build_dataset(ConstantLineSource(symmetric), DegradationPipeline([]))
        for index in range(40):
            crop, _ = dataset.load_crop(index)
            assert np.array_equal(np.asarray(crop), np.asarray(symmetric))

    def test_codec_stage_does_not_shift_brightness_between_labels(self) -> None:
        dataset = _build_dataset(
            ConstantLineSource(_vertically_symmetric_image()),
            DegradationPipeline.build_codec_stage(WORK_HEIGHT),
        )
        brightness = {Orientation.UPRIGHT: [], Orientation.ROTATED: []}
        for index in range(LEAKAGE_SAMPLE_SIZE):
            crop, orientation = dataset.load_crop(index)
            brightness[orientation].append(float(np.mean(np.asarray(crop, dtype=np.float32))))
        difference = abs(np.mean(brightness[Orientation.UPRIGHT]) - np.mean(brightness[Orientation.ROTATED]))
        assert difference < 1.0


class TestDatasetDeterminism:
    def test_the_same_index_yields_the_same_tensor(self, synthetic_source) -> None:
        first = _build_dataset(synthetic_source, DegradationPipeline.build_codec_stage(WORK_HEIGHT))
        second = _build_dataset(synthetic_source, DegradationPipeline.build_codec_stage(WORK_HEIGHT))
        for index in range(12):
            assert np.array_equal(first[index][0].numpy(), second[index][0].numpy())
            assert first[index][1].item() == second[index][1].item()

    def test_a_new_epoch_yields_different_samples(self, synthetic_source) -> None:
        dataset = _build_dataset(synthetic_source, DegradationPipeline.build_codec_stage(WORK_HEIGHT))
        first = dataset[3][0].numpy()
        dataset.set_epoch(1)
        assert not np.array_equal(first, dataset[3][0].numpy())

    def test_window_fit_choice_changes_the_crop_window(self, synthetic_source) -> None:
        config = PreprocessConfig()
        wide = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 900, 3), dtype=np.uint8), "RGB")
        rng = np.random.default_rng(0)
        centered = ImagePreprocessor(config, CenterWindowFit()).to_tensor(wide, rng)
        randomized = ImagePreprocessor(config, RandomWindowFit()).to_tensor(wide, np.random.default_rng(3))
        assert not np.array_equal(centered.numpy(), randomized.numpy())


class TestSyntheticGeometry:
    def test_generated_crops_follow_the_requested_height_distribution(self, synthetic_source, test_profile) -> None:
        heights = self._collect(synthetic_source, lambda image: image.height)
        expected = test_profile.crop_height.median
        assert abs(np.median(heights) - expected) / expected < MEDIAN_TOLERANCE

    def test_generated_crops_follow_the_requested_aspect_distribution(self, synthetic_source, test_profile) -> None:
        """Допуск шире, чем по высоте: наклон строки неизбежно делает бокс более квадратным.

        У строки, наклонённой сильнее чем arctan(1/A), бокс физически не может иметь пропорции A,
        сколько её ни удлиняй. Часть дефицита компенсируется длиной строки, остаток — цена
        реализма, и она осознанная: именно на наклонных кропах модель слабее всего.
        """
        aspects = self._collect(synthetic_source, lambda image: image.width / image.height)
        expected = test_profile.aspect_ratio.median
        assert abs(np.median(aspects) - expected) / expected < TILTED_ASPECT_TOLERANCE

    def test_every_crop_is_wider_than_it_is_tall(self, synthetic_source) -> None:
        aspects = self._collect(synthetic_source, lambda image: image.width / image.height)
        assert min(aspects) > 1.0

    @staticmethod
    def _collect(source: SyntheticTextLineSource, measure) -> list[float]:
        scheme = SeedScheme(base_seed=11)
        return [measure(source.load_upright(index, scheme.rng_for(index))) for index in range(DISTRIBUTION_SAMPLE_SIZE)]


class TestPreprocessing:
    def test_tensor_shape_matches_the_configuration(self) -> None:
        config = PreprocessConfig(height=48, width=160, channel_count=3)
        image = Image.fromarray(np.zeros((30, 300, 3), dtype=np.uint8), "RGB")
        tensor = ImagePreprocessor(config, CenterWindowFit()).to_tensor(image, np.random.default_rng(0))
        assert tuple(tensor.shape) == (3, 48, 160)

    def test_narrow_crops_are_padded_to_the_target_width(self) -> None:
        image = Image.fromarray(np.full((20, 30, 3), 200, dtype=np.uint8), "RGB")
        tensor = ImagePreprocessor(PreprocessConfig(), CenterWindowFit()).to_tensor(image, np.random.default_rng(0))
        assert tuple(tensor.shape) == (1, 32, 192)

    def test_invalid_channel_count_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            PreprocessConfig(channel_count=2)


class TestManifestSource:
    """Вырезки из реальных фотографий: бокс с джиттером не должен выходить за края кадра."""

    UNIFORM_COLOR = (123, 45, 67)
    IMAGE_SIZE = (100, 60)

    def _write_image(self, directory) -> str:
        image = Image.new("RGB", self.IMAGE_SIZE, self.UNIFORM_COLOR)
        image.save(directory / "photo.png")
        return "photo.png"

    def _source(self, directory, box: CropRecord) -> ManifestTextLineSource:
        return ManifestTextLineSource([box], directory, ValueRange(0.4, 0.6))

    def test_corner_box_stays_inside_the_image(self, tmp_path) -> None:
        name = self._write_image(tmp_path)
        record = CropRecord(image_path=name, left=0, top=0, width=30, height=20)
        crop = self._source(tmp_path, record).load_upright(0, np.random.default_rng(0))
        assert crop.width <= self.IMAGE_SIZE[0] and crop.height <= self.IMAGE_SIZE[1]

    def test_no_black_border_appears_at_the_image_edge(self, tmp_path) -> None:
        name = self._write_image(tmp_path)
        record = CropRecord(image_path=name, left=0, top=0, width=30, height=20)
        source = self._source(tmp_path, record)
        for index in range(20):
            pixels = np.asarray(source.load_upright(index, np.random.default_rng(index)))
            assert np.array_equal(np.unique(pixels.reshape(-1, 3), axis=0), np.array([self.UNIFORM_COLOR]))

    def test_interior_box_is_padded_on_every_side(self, tmp_path) -> None:
        name = self._write_image(tmp_path)
        record = CropRecord(image_path=name, left=40, top=25, width=20, height=10)
        crop = self._source(tmp_path, record).load_upright(0, np.random.default_rng(0))
        assert crop.width > record.width and crop.height > record.height

    def test_empty_manifest_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            ManifestTextLineSource([], tmp_path, ValueRange(0.0, 0.1))


class TestFontFallback:
    """Одна неудачная пара «шрифт + текст» не должна ронять часовую генерацию.

    Среди полутора тысяч шрифтов попадаются такие, где PIL сообщает один размер растра, а
    выделяет другой; обе пробы их пропускают, а падение обнуляло час работы.
    """

    def _break_fonts(self, source: SyntheticTextLineSource, doomed_names: set[str]) -> None:
        original = source._components.line_renderer.render_at_least

        def render(text_source, style, min_aspect, rng):
            if Path(style.font.path).name in doomed_names:
                raise FontRenderError(style.font.path, "текст")
            return original(text_source, style, min_aspect, rng)

        source._components.line_renderer.render_at_least = render

    def test_a_failing_font_is_replaced(self, synthetic_source: SyntheticTextLineSource) -> None:
        rng = SeedScheme(0).rng_for(0)
        style_font = synthetic_source._make_style(
            SeedScheme(0).rng_for(0), Script.CYRILLIC,
            synthetic_source._components.layout_sampler.sample(SeedScheme(0).rng_for(0), 5.0), 48).font
        self._break_fonts(synthetic_source, {Path(style_font.path).name})
        assert synthetic_source.load_upright(0, rng).width > 0

    def test_replacements_are_counted(self, synthetic_source: SyntheticTextLineSource) -> None:
        """Счётчик — единственный способ отличить единичный битый шрифт от системной беды."""
        original = synthetic_source._components.line_renderer.render_at_least
        remaining = [2]

        def fails_twice(text_source, style, min_aspect, rng):
            if remaining[0] > 0:
                remaining[0] -= 1
                raise FontRenderError(style.font.path, "текст")
            return original(text_source, style, min_aspect, rng)

        synthetic_source._components.line_renderer.render_at_least = fails_twice
        synthetic_source.load_upright(0, SeedScheme(0).rng_for(0))
        assert synthetic_source.fallback_count == 2

    def test_a_hopeless_registry_names_the_font(self, synthetic_source: SyntheticTextLineSource) -> None:
        def always_fails(*args, **kwargs):
            raise FontRenderError(Path("broken.ttf"), "текст")

        synthetic_source._components.line_renderer.render_at_least = always_fails
        with pytest.raises(FontRenderError):
            synthetic_source.load_upright(0, SeedScheme(0).rng_for(0))
