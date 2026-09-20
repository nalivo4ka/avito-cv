from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from avitocv.data.datasets import CenterWindowFit, ImagePreprocessor, OrientationDataset, PreprocessConfig
from avitocv.data.degradation import DegradationPipeline
from avitocv.data.real.hiertext import BoundingBox, HierTextManifestBuilder, LineFilter, top_edge_angle
from avitocv.data.matching.geometry import CropGeometry, JointGeometryWeighter, TruncatedImportanceSampler
from avitocv.data.manifest import CropManifest, CropRecord
from avitocv.data.matching.profile import EmpiricalDistribution, GeometryReference
from avitocv.data.matching.profiling import CropMeasurer
from avitocv.data.real.textocr import (
    LineGrouper,
    PlacedWord,
    RunEmitter,
    TextOcrManifestBuilder,
    WordFilter,
    merge_boxes,
)
from avitocv.data.sampling import SeedScheme, ValueRange
from avitocv.data.sources import ManifestTextLineSource
from tests.conftest import WORK_HEIGHT

UPRIGHT_QUAD = [[10, 10], [90, 12], [90, 30], [10, 28]]
UPSIDE_DOWN_QUAD = [[90, 30], [10, 28], [10, 10], [90, 12]]
STEEP_QUAD = [[10, 10], [30, 80], [45, 75], [25, 5]]


def _line(vertices, legible=True, vertical=False, handwritten=False) -> dict:
    return {
        "vertices": vertices,
        "text": "sample",
        "legible": legible,
        "vertical": vertical,
        "handwritten": handwritten,
    }


class TestOrientationFromVertexOrder:
    """Порядок вершин четырёхугольника — единственный источник метки для реальных кропов."""

    def test_upright_line_is_recognized(self) -> None:
        assert LineFilter().is_upright(_line(UPRIGHT_QUAD)) is True

    def test_upside_down_line_is_recognized(self) -> None:
        assert LineFilter().is_upright(_line(UPSIDE_DOWN_QUAD)) is False

    def test_steeply_tilted_line_has_no_verdict(self) -> None:
        assert LineFilter().is_upright(_line(STEEP_QUAD)) is None

    def test_angle_of_upright_line_is_near_zero(self) -> None:
        assert abs(top_edge_angle(UPRIGHT_QUAD)) < 5.0

    def test_angle_of_upside_down_line_is_near_half_turn(self) -> None:
        assert abs(abs(top_edge_angle(UPSIDE_DOWN_QUAD)) - 180.0) < 5.0


class TestLineFilter:
    def test_illegible_lines_are_rejected(self) -> None:
        assert not LineFilter().is_usable(_line(UPRIGHT_QUAD, legible=False))

    def test_vertical_lines_are_rejected(self) -> None:
        assert not LineFilter().is_usable(_line(UPRIGHT_QUAD, vertical=True))

    def test_short_lines_are_rejected(self) -> None:
        assert not LineFilter().is_usable(_line([[0, 0], [8, 0], [8, 6], [0, 6]]))

    def test_tall_narrow_lines_are_rejected(self) -> None:
        assert not LineFilter().is_usable(_line([[0, 0], [20, 0], [20, 40], [0, 40]]))

    def test_horizontal_legible_line_is_accepted(self) -> None:
        assert LineFilter().is_usable(_line(UPRIGHT_QUAD))


class TestBoundingBox:
    def test_box_encloses_every_vertex(self) -> None:
        box = BoundingBox.around(UPRIGHT_QUAD)
        assert (box.left, box.top, box.width, box.height) == (10, 10, 80, 20)

    def test_aspect_ratio_matches_the_enclosed_shape(self) -> None:
        assert BoundingBox.around(UPRIGHT_QUAD).aspect_ratio == pytest.approx(4.0)


class TestManifestBuilder:
    def _annotation(self, lines: list[dict]) -> dict:
        return {"image_id": "abc", "paragraphs": [{"lines": lines}]}

    def test_only_upright_lines_reach_the_manifest(self, tmp_path) -> None:
        builder = HierTextManifestBuilder(LineFilter())
        records = builder.build([self._annotation([_line(UPRIGHT_QUAD), _line(UPSIDE_DOWN_QUAD)])], tmp_path)
        assert len(records) == 1
        assert builder.report.upside_down == 1

    def test_handwritten_lines_land_in_their_own_slice(self, tmp_path) -> None:
        builder = HierTextManifestBuilder(LineFilter())
        records = builder.build([self._annotation([_line(UPRIGHT_QUAD, handwritten=True)])], tmp_path)
        assert records[0].slice_name == "hiertext_handwritten"

    def test_rejected_lines_are_counted(self, tmp_path) -> None:
        builder = HierTextManifestBuilder(LineFilter())
        builder.build([self._annotation([_line(UPRIGHT_QUAD, legible=False)])], tmp_path)
        assert builder.report.rejected == 1 and builder.report.accepted == 0

class TestJointGeometryWeighter:
    """Веса приводят к тестовой геометрии обе величины сразу, а не одну высоту."""

    def _geometry(self, seed: int, height_range, aspect_range, size: int = 20_000) -> CropGeometry:
        rng = np.random.default_rng(seed)
        return CropGeometry.of(rng.uniform(*height_range, size=size), rng.uniform(*aspect_range, size=size))

    def _profile(self, geometry: CropGeometry) -> GeometryReference:
        return GeometryReference(
            crop_height=EmpiricalDistribution.from_samples(geometry.heights),
            aspect_ratio=EmpiricalDistribution.from_samples(geometry.aspects),
        )

    def test_weighting_moves_both_medians_toward_the_reference(self) -> None:
        reference = self._geometry(0, (30, 200), (4.0, 12.0))
        actual = self._geometry(1, (10, 200), (1.2, 12.0))
        weights = JointGeometryWeighter().weights_for(actual, self._profile(reference))
        for values, target in ((actual.heights, reference.heights), (actual.aspects, reference.aspects)):
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order]) / weights.sum()
            weighted_median = np.interp(0.5, cumulative, values[order])
            assert abs(weighted_median - np.median(target)) / np.median(target) < 0.15

    def test_aspect_skew_is_corrected_where_height_alone_would_miss_it(self) -> None:
        """Ровно та ошибка, из-за которой валидация занижала балл: высоты совпадают, пропорции нет."""
        reference = self._geometry(0, (20, 100), (5.0, 12.0))
        actual = self._geometry(1, (20, 100), (1.2, 12.0))
        weights = JointGeometryWeighter().weights_for(actual, self._profile(reference))
        short = actual.aspects < 2.5
        assert np.mean(short) > 0.1, "в исходной выборке коротких кропов должно быть заметно"
        assert weights[short].sum() / weights.sum() < 0.02

    def test_weights_average_to_one(self) -> None:
        reference = self._geometry(0, (20, 120), (2.0, 9.0))
        weights = JointGeometryWeighter().weights_for(self._geometry(1, (20, 120), (2.0, 9.0)), self._profile(reference))
        assert np.mean(weights) == pytest.approx(1.0, rel=1e-6)

    def test_effective_sample_size_never_exceeds_the_crop_count(self) -> None:
        reference = self._geometry(0, (30, 200), (4.0, 12.0))
        weights = JointGeometryWeighter().weights_for(self._geometry(1, (10, 200), (1.2, 12.0)), self._profile(reference))
        assert JointGeometryWeighter.effective_sample_size(weights) <= len(weights)

    def test_disjoint_distributions_are_reported(self) -> None:
        reference = self._geometry(0, (500, 900), (20.0, 30.0))
        with pytest.raises(ValueError):
            JointGeometryWeighter().weights_for(self._geometry(1, (10, 20), (1.2, 1.5)), self._profile(reference))

    def test_grid_spans_both_ranges_so_impossible_crops_get_no_weight(self) -> None:
        """Сетка по одному эталону прижала бы чужие кропы к крайней корзине и дала бы им вес."""
        reference = self._geometry(0, (30, 60), (4.0, 6.0))
        actual = CropGeometry.of(
            np.concatenate([reference.heights, np.full(2000, 400.0)]),
            np.concatenate([reference.aspects, np.full(2000, 30.0)]),
        )
        weights = JointGeometryWeighter().weights_for(actual, self._profile(reference))
        assert weights[-2000:].sum() == pytest.approx(0.0)


class TestWeightedManifest:
    def _manifest(self) -> CropManifest:
        records = [
            CropRecord(image_path="a.png", left=0, top=0, width=60, height=20, weight=2.5),
            CropRecord(image_path="a.png", left=0, top=0, width=90, height=30, weight=0.5),
        ]
        return CropManifest.from_records(records)

    def test_weights_survive_a_parquet_round_trip(self, tmp_path) -> None:
        path = tmp_path / "manifest.parquet"
        self._manifest().save(path)
        assert np.allclose(CropManifest.load(path).weights, [2.5, 0.5])

    def test_manifest_without_weight_column_defaults_to_one(self) -> None:
        frame = self._manifest().frame.drop(columns=["weight"])
        assert np.allclose(CropManifest(frame).weights, [1.0, 1.0])

    def test_source_reports_the_weight_of_each_crop(self, tmp_path) -> None:
        source = ManifestTextLineSource.from_manifest(self._manifest(), tmp_path)
        assert (source.weight_of(0), source.weight_of(1)) == (2.5, 0.5)

    def test_unknown_slice_is_reported(self) -> None:
        with pytest.raises(ValueError):
            self._manifest().take_slice("missing")


class TestRealCodecStage:
    """Для реальных кропов кодековая стадия не должна работать аугментацией."""

    def _crop(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        pixels = np.full((40, 200, 3), 240, dtype=np.uint8)
        pixels[12:28, ::7] = 20
        return np.clip(pixels.astype(np.int16) + rng.integers(-4, 5, pixels.shape), 0, 255).astype(np.uint8)

    def test_real_stage_preserves_sharpness(self) -> None:
        crop = self._crop()
        measurer = CropMeasurer()
        processed = DegradationPipeline.build_real_codec_stage(WORK_HEIGHT).apply(crop, np.random.default_rng(0))
        before = measurer.measure(crop).sharpness
        after = measurer.measure(processed).sharpness
        assert after / before > 0.8

    def test_training_stage_degrades_much_harder(self) -> None:
        crop = self._crop()
        measurer = CropMeasurer()
        ratios = []
        for seed in range(12):
            processed = DegradationPipeline.build_codec_stage(WORK_HEIGHT).apply(crop, np.random.default_rng(seed))
            ratios.append(measurer.measure(processed).sharpness / measurer.measure(crop).sharpness)
        assert np.median(ratios) < 0.8

    def test_real_stage_always_reencodes(self) -> None:
        crop = self._crop()
        processed = DegradationPipeline.build_real_codec_stage(WORK_HEIGHT).apply(crop, np.random.default_rng(0))
        assert not np.array_equal(processed, crop)


class TestWeightedSamples:
    def test_dataset_exposes_the_crop_weight(self, tmp_path) -> None:
        Image.new("RGB", (200, 80), (200, 180, 160)).save(tmp_path / "photo.png")
        manifest = CropManifest.from_records([
            CropRecord(image_path="photo.png", left=20, top=20, width=100, height=25, weight=3.0)
        ])
        dataset = OrientationDataset(
            source=ManifestTextLineSource.from_manifest(manifest, tmp_path, ValueRange(0.0, 0.05)),
            codec_stage=DegradationPipeline.build_real_codec_stage(WORK_HEIGHT),
            preprocessor=ImagePreprocessor(PreprocessConfig(), CenterWindowFit()),
            seed_scheme=SeedScheme(0),
        )
        sample = dataset[0]
        assert sample.weight.item() == pytest.approx(3.0)
        assert sample._fields == ("image", "label", "weight")

    def test_synthetic_sources_report_unit_weight(self, synthetic_source) -> None:
        assert synthetic_source.weight_of(0) == 1.0


class TestTextOcrWords:
    """Словарная разметка TextOCR: то же соглашение о вершинах, другой формат файла."""

    def _annotation(self, points: list[float], text: str = "SALE") -> dict:
        return {"points": points, "utf8_string": text}

    def _upright(self) -> dict:
        return self._annotation([10.0, 10.0, 90.0, 10.0, 90.0, 40.0, 10.0, 40.0])

    def test_vertex_order_gives_the_orientation(self) -> None:
        flipped = self._annotation([90.0, 40.0, 10.0, 40.0, 10.0, 10.0, 90.0, 10.0])
        assert WordFilter().is_upright(self._upright()) is True
        assert WordFilter().is_upright(flipped) is False

    def test_vertical_word_has_no_orientation(self) -> None:
        vertical = self._annotation([10.0, 10.0, 10.0, 90.0, 40.0, 90.0, 40.0, 10.0])
        assert WordFilter().is_upright(vertical) is None

    def test_illegible_word_is_rejected(self) -> None:
        assert not WordFilter().is_usable(self._annotation(self._upright()["points"], text="."))

    def test_short_words_are_kept(self) -> None:
        """Ради коротких кропов датасет и берётся, порог HierText отсёк бы их."""
        square = self._annotation([10.0, 10.0, 45.0, 10.0, 45.0, 40.0, 10.0, 40.0])
        assert WordFilter().is_usable(square)
        assert not LineFilter().is_usable(
            {"vertices": [[10, 10], [45, 10], [45, 40], [10, 40]],
             "legible": True, "vertical": False, "handwritten": False}
        )

    def test_polygon_with_more_than_four_vertices_is_rejected(self) -> None:
        assert not WordFilter().is_usable(self._annotation([10.0, 10.0, 50.0, 8.0, 90.0, 10.0,
                                                            90.0, 40.0, 10.0, 40.0]))

    def _payload(self) -> dict:
        return {
            "imgs": {"keep": {}, "drop": {}},
            "anns": {"a": self._upright()},
            "imgToAnns": {"keep": ["a"], "drop": ["a"]},
        }

    def test_images_shared_with_validation_are_excluded(self) -> None:
        """26 снимков TextOCR лежат в валидации HierText: без исключения валидация течёт."""
        builder = TextOcrManifestBuilder(WordFilter(), excluded_image_ids={"drop"})
        records = builder.build(self._payload(), Path("data/real/textocr/train_images"))
        assert [record.image_path for record in records] == ["data/real/textocr/train_images/keep.jpg"]
        assert builder.report.excluded_images == 1


class TestTextOcrLineComposition:
    """Склейка слов в строки: без неё словарный TextOCR не совпадает с тестом по геометрии."""

    def _word(self, left: int, top: int, width: int = 60, height: int = 20, angle: float = 0.0) -> PlacedWord:
        return PlacedWord(box=BoundingBox(left=left, top=top, width=width, height=height), angle=angle)

    def test_neighbouring_words_join_one_line(self) -> None:
        words = [self._word(0, 100), self._word(70, 100), self._word(140, 102)]
        assert [len(line) for line in LineGrouper().group(words)] == [3]

    def test_words_on_different_rows_stay_apart(self) -> None:
        lines = LineGrouper().group([self._word(0, 100), self._word(0, 200)])
        assert [len(line) for line in lines] == [1, 1]

    def test_a_wide_gap_breaks_the_line(self) -> None:
        """Подпись слева и значение справа — одна строка на глаз, но детектор их не склеит."""
        lines = LineGrouper().group([self._word(0, 100), self._word(400, 100)])
        assert [len(line) for line in lines] == [1, 1]

    def test_differently_tilted_words_stay_apart(self) -> None:
        lines = LineGrouper().group([self._word(0, 100), self._word(70, 100, angle=30.0)])
        assert [len(line) for line in lines] == [1, 1]

    def test_merged_box_covers_every_word(self) -> None:
        box = merge_boxes([self._word(10, 100), self._word(80, 104, width=40, height=16)])
        assert (box.left, box.top, box.left + box.width, box.top + box.height) == (10, 100, 120, 120)

    def test_runs_cover_lengths_from_one_word_to_the_whole_line(self) -> None:
        line = [self._word(index * 70, 100) for index in range(5)]
        lengths = sorted({len(run) for run in RunEmitter().emit(line)})
        assert lengths == [1, 2, 3, 5]

    def test_merged_crops_are_longer_than_single_words(self) -> None:
        payload = {
            "imgs": {"photo": {}},
            "anns": {
                str(index): {"utf8_string": "word",
                             "points": [index * 70.0, 100.0, index * 70.0 + 60, 100.0,
                                        index * 70.0 + 60, 120.0, index * 70.0, 120.0]}
                for index in range(4)
            },
            "imgToAnns": {"photo": [str(index) for index in range(4)]},
        }
        builder = TextOcrManifestBuilder(WordFilter(), grouper=LineGrouper())
        records = builder.build(payload, Path("images"))
        aspects = [record.width / record.height for record in records]
        assert max(aspects) > 10.0, "строка целиком обязана дать длинный кроп"
        assert min(aspects) == pytest.approx(3.0), "одиночные слова тоже остаются"


class TestTruncatedImportanceSampling:
    """Точное выравнивание геометрии платит независимыми снимками, которых у нас мало."""

    def _skewed_ratio(self, size: int = 10_000) -> np.ndarray:
        """Отношение плотностей с тяжёлым хвостом — как в редких ячейках сетки."""
        ratio = np.full(size, 1.0)
        ratio[:5] = 5000.0
        return ratio

    def test_capping_raises_the_effective_sample_size(self) -> None:
        ratio = self._skewed_ratio()
        draws = 100_000
        plain = TruncatedImportanceSampler(max_repeats=0).probabilities(ratio, draws)
        capped = TruncatedImportanceSampler(max_repeats=10).probabilities(ratio, draws)
        assert 1.0 / np.sum(capped ** 2) > 10 * (1.0 / np.sum(plain ** 2))

    def test_no_crop_exceeds_the_repeat_cap(self) -> None:
        draws = 100_000
        capped = TruncatedImportanceSampler(max_repeats=10).probabilities(self._skewed_ratio(), draws)
        assert capped.max() * draws <= 10.0 + 1e-6

    def test_probabilities_still_sum_to_one(self) -> None:
        capped = TruncatedImportanceSampler(max_repeats=10).probabilities(self._skewed_ratio(), 100_000)
        assert capped.sum() == pytest.approx(1.0)

    def test_capping_off_leaves_the_ratio_untouched(self) -> None:
        ratio = self._skewed_ratio()
        plain = TruncatedImportanceSampler(max_repeats=0).probabilities(ratio, 100_000)
        assert plain == pytest.approx(ratio / ratio.sum())

    def test_a_ratio_already_under_the_cap_is_untouched(self) -> None:
        ratio = np.ones(1000)
        capped = TruncatedImportanceSampler(max_repeats=10).probabilities(ratio, 1000)
        assert capped == pytest.approx(ratio / ratio.sum())
