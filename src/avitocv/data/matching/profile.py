"""Эмпирические распределения характеристик кропов и их сериализация.

`CropProfile` — паспорт выборки кропов. Профиль тестовой выборки строится скриптом
`scripts/profile_test_set.py`, лежит в `configs/test_profile.json`, и генератор берёт из него
высоту и aspect ratio, чтобы синтетика совпадала с тестом по геометрии.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

QUANTILE_COUNT = 101
QUANTILE_GRID = np.linspace(0.0, 1.0, QUANTILE_COUNT)


@dataclass(frozen=True)
class EmpiricalDistribution:
    """Распределение, заданное 101 квантилем; сэмплирует обратным преобразованием."""

    quantiles: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.quantiles) != QUANTILE_COUNT:
            raise ValueError(f"expected {QUANTILE_COUNT} quantiles, got {len(self.quantiles)}")

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "EmpiricalDistribution":
        if len(samples) == 0:
            raise ValueError("cannot build a distribution from an empty sample")
        values = np.percentile(np.asarray(samples, dtype=float), QUANTILE_GRID * 100.0)
        return cls(tuple(float(value) for value in values))

    @classmethod
    def from_dict(cls, payload: dict) -> "EmpiricalDistribution":
        return cls(tuple(float(value) for value in payload["quantiles"]))

    def to_dict(self) -> dict:
        return {"quantiles": list(self.quantiles)}

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        # Квантили — это уже табулированная обратная функция распределения, поэтому
        # интерполяция по равномерному числу даёт выборку из исходного распределения.
        uniform = rng.random() if size is None else rng.random(size)
        return np.interp(uniform, QUANTILE_GRID, np.asarray(self.quantiles))

    def quantile(self, probability: float) -> float:
        return float(np.interp(probability, QUANTILE_GRID, np.asarray(self.quantiles)))

    @property
    def median(self) -> float:
        return self.quantile(0.5)


@dataclass(frozen=True)
class GeometryReference:
    """Эталонные распределения высоты и пропорций — всё, что нужно для выравнивания выборок."""

    crop_height: EmpiricalDistribution
    aspect_ratio: EmpiricalDistribution


@dataclass(frozen=True)
class CropProfile:
    """Паспорт выборки кропов: геометрия, резкость, контраст, доля серых."""

    crop_height: EmpiricalDistribution
    aspect_ratio: EmpiricalDistribution
    sharpness: EmpiricalDistribution
    ink_spread: EmpiricalDistribution
    grayscale_share: float
    sample_size: int

    @property
    def geometry(self) -> "GeometryReference":
        """Только те две величины, по которым выравнивают выборки."""
        return GeometryReference(self.crop_height, self.aspect_ratio)

    @classmethod
    def load(cls, path: Path) -> "CropProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            crop_height=EmpiricalDistribution.from_dict(payload["crop_height"]),
            aspect_ratio=EmpiricalDistribution.from_dict(payload["aspect_ratio"]),
            sharpness=EmpiricalDistribution.from_dict(payload["sharpness"]),
            ink_spread=EmpiricalDistribution.from_dict(payload["ink_spread"]),
            grayscale_share=float(payload["grayscale_share"]),
            sample_size=int(payload["sample_size"]),
        )

    def save(self, path: Path) -> None:
        payload = {
            "crop_height": self.crop_height.to_dict(),
            "aspect_ratio": self.aspect_ratio.to_dict(),
            "sharpness": self.sharpness.to_dict(),
            "ink_spread": self.ink_spread.to_dict(),
            "grayscale_share": self.grayscale_share,
            "sample_size": self.sample_size,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
