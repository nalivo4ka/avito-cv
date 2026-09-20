"""Измерение готовых кропов и построение профиля выборки.

Одним и тем же измерителем считается профиль тестовой выборки и профиль сгенерированных
данных, поэтому `scripts/compare_profiles.py` может сравнить их поквантильно и показать,
где синтетика расходится с реальностью.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from avitocv.data.matching.profile import CropProfile, EmpiricalDistribution

SHARPNESS_REFERENCE_HEIGHT = 32
GRAYSCALE_CHANNEL_TOLERANCE = 2.0


@dataclass(frozen=True)
class CropMeasurement:
    """Результат измерения одного кропа."""

    height: int
    aspect_ratio: float
    sharpness: float
    ink_spread: float
    is_grayscale: bool


class CropMeasurer:
    """Считает по кропу высоту, aspect ratio, резкость и контраст."""

    def measure(self, rgb: np.ndarray) -> CropMeasurement:
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return CropMeasurement(
            height=height,
            aspect_ratio=width / height,
            sharpness=self._measure_sharpness(gray),
            ink_spread=float(gray.std()) / 255.0,
            is_grayscale=self._is_grayscale(rgb),
        )

    def _measure_sharpness(self, gray: np.ndarray) -> float:
        # Дисперсия лапласиана зависит от масштаба, поэтому кроп сначала приводится к общей
        # высоте. Без этого крупные кропы всегда выглядели бы резче мелких, и сравнение
        # синтетики с тестом ничего бы не значило.
        scale = SHARPNESS_REFERENCE_HEIGHT / gray.shape[0]
        target_size = (max(1, int(round(gray.shape[1] * scale))), SHARPNESS_REFERENCE_HEIGHT)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(resized.astype(np.float32) / 255.0, cv2.CV_32F).var())

    def _is_grayscale(self, rgb: np.ndarray) -> bool:
        channels = rgb.astype(np.int16)
        red_green = np.abs(channels[..., 0] - channels[..., 1]).mean()
        green_blue = np.abs(channels[..., 1] - channels[..., 2]).mean()
        return bool(max(red_green, green_blue) < GRAYSCALE_CHANNEL_TOLERANCE)


class CropSetProfiler:
    """Превращает поток кропов в `CropProfile`."""

    def __init__(self, measurer: CropMeasurer) -> None:
        self._measurer = measurer

    def build(self, images: Iterable[np.ndarray]) -> CropProfile:
        measurements = [self._measurer.measure(image) for image in images]
        if not measurements:
            raise ValueError("cannot profile an empty crop set")
        return CropProfile(
            crop_height=EmpiricalDistribution.from_samples([item.height for item in measurements]),
            aspect_ratio=EmpiricalDistribution.from_samples([item.aspect_ratio for item in measurements]),
            sharpness=EmpiricalDistribution.from_samples([item.sharpness for item in measurements]),
            ink_spread=EmpiricalDistribution.from_samples([item.ink_spread for item in measurements]),
            grayscale_share=float(np.mean([item.is_grayscale for item in measurements])),
            sample_size=len(measurements),
        )
