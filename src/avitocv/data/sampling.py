"""Детерминированный источник случайности и примитивы выборки параметров.

Весь пайплайн берёт случайные величины только из генератора, выданного `SeedScheme`, поэтому
сэмпл полностью определяется тройкой (сид, эпоха, индекс). Именно это делает обучающий набор
воспроизводимым.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class SeedScheme:
    """Отображает пару (индекс сэмпла, эпоха) в независимый генератор случайных чисел."""

    base_seed: int

    def rng_for(self, sample_index: int, epoch: int = 0) -> np.random.Generator:
        # SeedSequence нужен именно потому, что default_rng(base_seed + sample_index) даёт
        # коррелированные потоки на соседних сидах — сэмплы под индексами 5 и 6 оказались бы
        # похожими. SeedSequence перемешивает энтропию и гарантирует независимость.
        entropy = [self.base_seed, epoch, sample_index]
        return np.random.default_rng(np.random.SeedSequence(entropy))


@dataclass(frozen=True)
class ValueRange:
    """Отрезок, из которого берётся параметр; заменяет магические числа в коде."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"invalid range: [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))

    def sample_int(self, rng: np.random.Generator) -> int:
        return int(rng.integers(int(self.low), int(self.high) + 1))


def weighted_choice(rng: np.random.Generator, items: Sequence[T], weights: Sequence[float]) -> T:
    if len(items) != len(weights):
        raise ValueError("items and weights must have the same length")
    if not items:
        raise ValueError("cannot choose from an empty sequence")
    probabilities = np.asarray(weights, dtype=float)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return items[int(rng.choice(len(items), p=probabilities / total))]


def happens(rng: np.random.Generator, probability: float) -> bool:
    return bool(rng.random() < probability)
