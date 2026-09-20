from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from avitocv.data.synthesis.corpus import LineFileSource, MixedTextSource, PatternSource
from avitocv.data.degradation import (
    DegradationPipeline,
    GaussianNoiseDegradation,
    JpegDegradation,
    ProbabilisticDegradation,
    ResizeToHeight,
)
from avitocv.data.synthesis.fonts import FontCoverageInspector, FontRegistry, FontRenderProbe, SupportedTextFilter
from avitocv.data.synthesis.palettes import ColorSchemeSampler, contrast_ratio
from avitocv.data.matching.profile import EmpiricalDistribution
from avitocv.data.sampling import SeedScheme, ValueRange, weighted_choice
from avitocv.data.synthesis.writing_systems import Script, ScriptDetector
from tests.conftest import WORK_HEIGHT

CYRILLIC_SAMPLE = "Продам стол"
LATIN_SAMPLE = "For sale"


class TestSeedScheme:
    def test_same_index_and_epoch_reproduce_the_same_draws(self) -> None:
        scheme = SeedScheme(base_seed=17)
        first = scheme.rng_for(5, epoch=2).random(8)
        second = scheme.rng_for(5, epoch=2).random(8)
        assert np.array_equal(first, second)

    def test_different_epochs_produce_different_draws(self) -> None:
        scheme = SeedScheme(base_seed=17)
        assert not np.array_equal(scheme.rng_for(5, epoch=0).random(8), scheme.rng_for(5, epoch=1).random(8))

    def test_different_indices_produce_different_draws(self) -> None:
        scheme = SeedScheme(base_seed=17)
        assert not np.array_equal(scheme.rng_for(5).random(8), scheme.rng_for(6).random(8))


class TestValueRange:
    def test_sample_stays_inside_the_range(self) -> None:
        rng = np.random.default_rng(0)
        values = [ValueRange(2.0, 5.0).sample(rng) for _ in range(500)]
        assert all(2.0 <= value <= 5.0 for value in values)

    def test_sample_int_includes_both_bounds(self) -> None:
        rng = np.random.default_rng(0)
        values = {ValueRange(1, 3).sample_int(rng) for _ in range(200)}
        assert values == {1, 2, 3}

    def test_inverted_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ValueRange(5.0, 1.0)


class TestWeightedChoice:
    def test_zero_weight_items_are_never_chosen(self) -> None:
        rng = np.random.default_rng(0)
        picks = {weighted_choice(rng, ["a", "b"], [1.0, 0.0]) for _ in range(200)}
        assert picks == {"a"}

    def test_mismatched_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            weighted_choice(np.random.default_rng(0), ["a", "b"], [1.0])


class TestEmpiricalDistribution:
    def test_quantiles_are_monotonic(self) -> None:
        distribution = EmpiricalDistribution.from_samples(np.random.default_rng(0).normal(size=10_000))
        quantiles = np.asarray(distribution.quantiles)
        assert np.all(np.diff(quantiles) >= 0)

    def test_sampling_reproduces_the_source_distribution(self) -> None:
        source = np.random.default_rng(0).lognormal(mean=3.0, sigma=0.6, size=50_000)
        distribution = EmpiricalDistribution.from_samples(source)
        drawn = distribution.sample(np.random.default_rng(1), size=50_000)
        assert abs(np.median(drawn) - np.median(source)) / np.median(source) < 0.05

    def test_round_trip_through_a_dictionary_is_lossless(self) -> None:
        distribution = EmpiricalDistribution.from_samples(np.arange(1000))
        assert EmpiricalDistribution.from_dict(distribution.to_dict()) == distribution

    def test_empty_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            EmpiricalDistribution.from_samples([])


class TestScriptDetector:
    def test_cyrillic_text_is_detected(self) -> None:
        assert ScriptDetector().dominant_script(CYRILLIC_SAMPLE) is Script.CYRILLIC

    def test_latin_text_is_detected(self) -> None:
        assert ScriptDetector().dominant_script(LATIN_SAMPLE) is Script.LATIN

    def test_digits_alone_have_no_dominant_script(self) -> None:
        assert ScriptDetector().dominant_script("12 500") is None


class TestFontRegistry:
    def test_indexed_fonts_declare_the_scripts_they_cover(self, font_registry: FontRegistry) -> None:
        assert font_registry.count_for(Script.CYRILLIC) > 0
        assert font_registry.count_for(Script.LATIN) > 0

    def test_sampled_font_supports_the_requested_script(self, font_registry: FontRegistry) -> None:
        rng = np.random.default_rng(0)
        assert all(
            font_registry.sample(rng, Script.CYRILLIC).can_render(Script.CYRILLIC)
            for _ in range(50)
        )

    def test_empty_registry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            FontRegistry([])


class TestSupportedTextFilter:
    def test_unsupported_characters_are_dropped(self, cyrillic_font) -> None:
        filtered = SupportedTextFilter().filter(f"{CYRILLIC_SAMPLE}￿", cyrillic_font)
        assert "￿" not in filtered and "" not in filtered

    def test_supported_characters_survive(self, cyrillic_font) -> None:
        assert SupportedTextFilter().filter(CYRILLIC_SAMPLE, cyrillic_font) == CYRILLIC_SAMPLE


class TestColorSchemeSampler:
    def test_schemes_keep_text_readable(self) -> None:
        rng = np.random.default_rng(0)
        ratios = [contrast_ratio(*self._pair(ColorSchemeSampler().sample(rng))) for _ in range(500)]
        assert np.median(ratios) > 3.0
        assert min(ratios) > 1.2

    @staticmethod
    def _pair(scheme):
        return scheme.foreground, scheme.background


class TestCorpusSources:
    def test_file_source_reads_every_line(self, corpus_path) -> None:
        assert len(LineFileSource(corpus_path)) == 64

    def test_missing_corpus_file_is_rejected(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            LineFileSource(tmp_path / "absent.txt")

    def test_pattern_source_never_returns_empty_text(self) -> None:
        rng = np.random.default_rng(0)
        source = PatternSource()
        assert all(source.sample_line(rng).strip() for _ in range(500))

    def test_mixture_respects_zero_weights(self, corpus_path) -> None:
        rng = np.random.default_rng(0)
        mixture = MixedTextSource([LineFileSource(corpus_path), PatternSource()], [1.0, 0.0])
        file_lines = set(LineFileSource(corpus_path)._lines)
        assert all(mixture.sample_line(rng) in file_lines for _ in range(100))


class TestDegradation:
    def test_pipeline_preserves_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(40, 200, 3), dtype=np.uint8)
        result = DegradationPipeline.build_codec_stage(WORK_HEIGHT).apply(image, rng)
        assert result.shape == image.shape and result.dtype == np.uint8

    def test_zero_probability_leaves_the_image_untouched(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(20, 60, 3), dtype=np.uint8)
        pipeline = DegradationPipeline([ProbabilisticDegradation(GaussianNoiseDegradation(), 0.0)])
        assert np.array_equal(pipeline.apply(image, rng), image)

    def test_jpeg_encoding_changes_pixels_but_keeps_geometry(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 128, 3), dtype=np.uint8)
        result = JpegDegradation(ValueRange(20, 20)).apply(image, rng)
        assert result.shape == image.shape and not np.array_equal(result, image)

    def test_resize_reaches_the_requested_height(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(97, 311, 3), dtype=np.uint8)
        resized = ResizeToHeight().apply_to(image, 32, rng)
        assert resized.shape[0] == 32
        assert abs(resized.shape[1] / 32 - 311 / 97) < 0.05


class TestFontRenderProbe:
    """Некоторые шрифты заявляют символы в cmap, но падают при отрисовке с обводкой."""

    def test_usable_fonts_pass_the_probe(self, font_registry: FontRegistry) -> None:
        rng = np.random.default_rng(0)
        probe = FontRenderProbe()
        assert all(probe.can_render(font_registry.sample(rng, Script.CYRILLIC)) for _ in range(5))

    def test_probe_matches_the_real_rendering_parameters(self) -> None:
        """Проверка в более мягких условиях, чем настоящая отрисовка, пропускает битые шрифты.

        Так и вышло: проба в 24 px с обводкой 1 признала пригодными шрифты, взрывающиеся
        в 64 px с обводкой 3, и генерация упала на 135 тысячах кропов из 600.
        """
        from avitocv.data.synthesis.fonts import PROBE_FONT_SIZE, PROBE_STROKE_WIDTH
        from avitocv.data.synthesis.rendering import NOMINAL_FONT_SIZE, LineStyleSampler

        assert PROBE_FONT_SIZE >= NOMINAL_FONT_SIZE
        assert PROBE_STROKE_WIDTH >= LineStyleSampler.stroke_width.high

    def test_broken_font_is_rejected(self) -> None:
        from pathlib import Path

        broken = Path("data/fonts/google/RubikPixels-400.ttf")
        if not broken.exists():
            pytest.skip("шрифт-образец не скачан")
        asset = FontCoverageInspector().inspect(broken)
        assert asset is not None, "по cmap шрифт выглядит пригодным"
        assert not FontRenderProbe().can_render(asset), "отрисовка обязана его отбраковать"


class TestRasterBounds:
    """Растр под маску текста обязан быть ограничен до вызова PIL, а не после падения.

    Генерация уже дважды умирала на `OSError: array allocation size too large` — сначала
    из-за нетронутой длинной строки, потом из-за декоративного шрифта. Обрезка по advance-ширине
    этого не гарантирует: выделяется растр по настоящим границам глифов вместе с обводкой.
    """

    def _style(self, registry: FontRegistry, stroke_width: int = 3):
        from avitocv.data.synthesis.rendering import NOMINAL_FONT_SIZE, ColorScheme, LineStyle

        return LineStyle(
            font=registry.sample(np.random.default_rng(0), Script.LATIN),
            font_size=NOMINAL_FONT_SIZE,
            letter_spacing_ratio=0.0,
            stroke_width=stroke_width,
            color_scheme=ColorScheme(foreground=(0, 0, 0), background=(255, 255, 255), stroke=None),
            skew_degrees=0.0,
        )

    def test_long_line_is_truncated_within_the_raster_budget(self, font_registry: FontRegistry) -> None:
        from avitocv.data.synthesis.rendering import MAX_LAYER_PIXELS, MAX_LAYER_WIDTH, TextLineRenderer

        renderer = TextLineRenderer()
        style = self._style(font_registry)
        truncated = renderer._truncate_to_layer("длинная строка без конца " * 400, style)
        width, height = renderer.measure_raster(truncated, style)
        assert width <= MAX_LAYER_WIDTH
        assert width * max(height, 1) <= MAX_LAYER_PIXELS

    def test_short_line_is_left_alone(self, font_registry: FontRegistry) -> None:
        from avitocv.data.synthesis.rendering import TextLineRenderer

        renderer = TextLineRenderer()
        assert renderer._truncate_to_layer("Распродажа", self._style(font_registry)) == "Распродажа"

    def test_every_indexed_font_renders_a_line_without_blowing_up(self, font_registry: FontRegistry) -> None:
        """Проба проверяет одиночные глифы; тут строка целиком, как в настоящей генерации."""
        from avitocv.data.synthesis.rendering import TextLineRenderer

        renderer = TextLineRenderer()
        rng = np.random.default_rng(1)
        for _ in range(40):
            style = replace(self._style(font_registry), font=font_registry.sample(rng, Script.LATIN))
            renderer.render("Sample Line 0123", style)

    def test_truncation_result_always_fits(self, font_registry: FontRegistry) -> None:
        """Ровно то, на чём споткнулось пропорциональное ужатие: оно возвращало строку сверх лимита."""
        from avitocv.data.synthesis.rendering import TextLineRenderer

        renderer = TextLineRenderer()
        rng = np.random.default_rng(2)
        line = "The quick brown fox jumps over the lazy dog 0123456789 " * 30
        for _ in range(60):
            style = replace(self._style(font_registry), font=font_registry.sample(rng, Script.LATIN))
            assert renderer._fits(renderer._truncate_to_layer(line, style), style)

    def test_truncation_keeps_the_longest_prefix_that_fits(self, font_registry: FontRegistry) -> None:
        """Ответ должен быть не просто допустимым, а максимальным: иначе кропы короче нужного."""
        from avitocv.data.synthesis.rendering import TextLineRenderer

        renderer = TextLineRenderer()
        style = self._style(font_registry)
        line = "распродажа сегодня " * 100
        truncated = renderer._truncate_to_layer(line, style)
        assert renderer._fits(truncated, style)
        if len(truncated) < len(line):
            assert not renderer._fits(line[:len(truncated) + 1], style)
