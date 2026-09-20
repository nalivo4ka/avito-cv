from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avitocv.data.synthesis.backgrounds import WeightedBackgroundMixture
from avitocv.data.synthesis.composition import CropComposer, CropLayoutSampler, FittedLineRenderer
from avitocv.data.synthesis.corpus import LineFileSource
from avitocv.data.degradation import DegradationPipeline
from avitocv.data.synthesis.fonts import FontRegistry, SupportedTextFilter
from avitocv.data.synthesis.rendering import LetterCaseSampler, LineStyleSampler, TextLineRenderer
from avitocv.data.sources import (
    ScriptedTextSource,
    SyntheticComponents,
    SyntheticSourceConfig,
    SyntheticTextLineSource,
)
from avitocv.data.factory import OrientationDatasetAssembler
from avitocv.data.matching.profile import CropProfile, EmpiricalDistribution
from avitocv.data.synthesis.writing_systems import Script

FONT_DIRECTORY = Path("data/fonts/google")
CORPUS_LINE_COUNT = 64
SAMPLE_FONT_COUNT = 6

# Высота входа отправленных моделей и рабочая высота деградаций при ней. Стадии деградаций
# принимают её параметром, и в обучении она приходит от `OrientationDatasetAssembler`; если
# вызывать их без аргумента, тесты проверяли бы порог 64, а исполняется 96.
MODEL_INPUT_HEIGHT = 48
WORK_HEIGHT = MODEL_INPUT_HEIGHT * OrientationDatasetAssembler.WORK_HEIGHT_FACTOR


@pytest.fixture(scope="session")
def font_registry() -> FontRegistry:
    index_path = Path("data/fonts/index.json")
    if not index_path.exists():
        pytest.skip("font index is missing, run scripts/collect_fonts.py")
    full = FontRegistry.load_index(index_path)
    rng = np.random.default_rng(0)
    assets = [full.sample(rng, Script.CYRILLIC) for _ in range(SAMPLE_FONT_COUNT)]
    assets += [full.sample(rng, Script.LATIN) for _ in range(SAMPLE_FONT_COUNT)]
    return FontRegistry(assets)


@pytest.fixture(scope="session")
def cyrillic_font(font_registry: FontRegistry):
    return font_registry.sample(np.random.default_rng(1), Script.CYRILLIC)


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    words = ["авито", "доставка",
             "цена", "товар", "продам"]
    lines = [" ".join(words[index % len(words)] for index in range(offset, offset + 4))
             for offset in range(CORPUS_LINE_COUNT)]
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def test_profile() -> CropProfile:
    rng = np.random.default_rng(7)
    return CropProfile(
        crop_height=EmpiricalDistribution.from_samples(rng.uniform(16, 120, size=5000)),
        aspect_ratio=EmpiricalDistribution.from_samples(rng.uniform(2.0, 12.0, size=5000)),
        sharpness=EmpiricalDistribution.from_samples(rng.uniform(0.005, 0.1, size=5000)),
        ink_spread=EmpiricalDistribution.from_samples(rng.uniform(0.05, 0.4, size=5000)),
        grayscale_share=0.16,
        sample_size=5000,
    )


SYNTHETIC_SOURCE_LENGTH = 300


@pytest.fixture
def synthetic_source(font_registry, corpus_path, test_profile) -> SyntheticTextLineSource:
    components = SyntheticComponents(
        text_sources=(ScriptedTextSource(LineFileSource(corpus_path), Script.CYRILLIC, 1.0),),
        style_sampler=LineStyleSampler(registry=font_registry),
        line_renderer=FittedLineRenderer(TextLineRenderer(), LetterCaseSampler(), SupportedTextFilter()),
        layout_sampler=CropLayoutSampler(),
        composer=CropComposer(WeightedBackgroundMixture.build_default(None)),
        profile=test_profile,
        capture_stage=DegradationPipeline.build_capture_stage(),
    )
    return SyntheticTextLineSource(components, SyntheticSourceConfig(virtual_length=SYNTHETIC_SOURCE_LENGTH))


