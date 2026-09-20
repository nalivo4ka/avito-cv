"""Предсказание вероятности поворота: симметризация и калибровка.

Задача обладает точной симметрией: для любого кропа ровно одна из двух ориентаций верна,
поэтому `p(x) + p(rot180(x))` обязано равняться единице. Обученная сеть это соотношение лишь
приближает. `SymmetricPredictor` навязывает его тождественно, усредняя логиты прямого и
перевёрнутого прохода: результат антисимметричен по построению, и средняя вероятность по любой
выборке равна ровно 0.5. Стоит это двойного прогона.

Калибровка температурой отдельна от симметризации, потому что Brier наказывает переуверенность
сильнее, чем ошибки: сеть, обученная на синтетике, почти всегда слишком уверена на реальных
кропах, и один скалярный параметр забирает заметную часть разрыва.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn

HALF_TURN_DIMENSIONS = (-2, -1)
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
DEFAULT_TEMPERATURE = 1.0


def rotate_half_turn(images: torch.Tensor) -> torch.Tensor:
    """Поворот батча на 180 градусов: отражение по обеим пространственным осям."""
    return torch.flip(images, dims=HALF_TURN_DIMENSIONS)


class LogitPredictor(ABC):
    """Выдаёт логит «кроп повёрнут» для батча картинок."""

    @abstractmethod
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class DirectPredictor(LogitPredictor):
    """Один прогон сети."""

    model: nn.Module

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


@dataclass(frozen=True)
class SymmetricPredictor(LogitPredictor):
    """Два прогона: прямой и перевёрнутый. Логит антисимметричен по построению.

    Сеть даёт z(x) и z(x'), где x' это повёрнутый кроп. Согласованная модель обязана давать
    z(x') = -z(x); полусумма z(x) - z(x') убирает несогласованность и заодно усредняет шум
    двух независимых взглядов на один кроп.
    """

    model: nn.Module

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        forward = self.model(images)
        flipped = self.model(rotate_half_turn(images))
        return (forward - flipped) / 2.0


@dataclass(frozen=True)
class TemperatureScaler:
    """Масштабирование логитов одним скаляром.

    Смещение сознательно не вводится: тестовая выборка сбалансирована, а после симметризации
    средняя вероятность и так равна 0.5. Свободный сдвиг мог бы только испортить это свойство,
    подстроившись под шум валидации.
    """

    temperature: float = DEFAULT_TEMPERATURE

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


class TemperatureCalibrator:
    """Подбирает температуру, напрямую минимизируя Brier — ту метрику, по которой судят."""

    def fit(self, logits: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> TemperatureScaler:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        sample_weights = np.ones_like(labels) if weights is None else np.asarray(weights, dtype=np.float64)

        def weighted_brier(temperature: float) -> float:
            probabilities = TemperatureScaler(temperature).probabilities(logits)
            return float(np.sum(sample_weights * (probabilities - labels) ** 2) / sample_weights.sum())

        found = minimize_scalar(
            weighted_brier,
            bounds=(MIN_TEMPERATURE, MAX_TEMPERATURE),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return TemperatureScaler(float(found.x))


@dataclass(frozen=True)
class CalibratedPredictor:
    """Готовое к инференсу сочетание предсказателя и калибровки."""

    predictor: LogitPredictor
    scaler: TemperatureScaler = TemperatureScaler()

    @torch.no_grad()
    def probabilities(self, images: torch.Tensor) -> np.ndarray:
        return self.scaler.probabilities(self.predictor.logits(images).detach().float().cpu().numpy())


class ProbabilityEnsemble:
    """Усредняет вероятности нескольких моделей.

    Усреднять надо именно вероятности, а не логиты: Brier строго выпукла по вероятности, поэтому
    по неравенству Йенсена ошибка среднего не превышает среднюю ошибку участников. Хуже среднего
    участника ансамбль быть не может — в отличие от усреднения логитов, где такой гарантии нет.

    Антисимметрия при этом сохраняется: у каждого участника p(x) + p(rot180 x) = 1, а значит и
    у среднего, так что ансамбль остаётся согласованным предсказателем.
    """

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self._weights = weights

    def combine(self, members: list[np.ndarray]) -> np.ndarray:
        if not members:
            raise ValueError("ансамбль пуст")
        stacked = np.stack(members)
        if self._weights is None:
            return stacked.mean(axis=0)
        if len(self._weights) != len(members):
            raise ValueError(f"весов {len(self._weights)}, участников {len(members)}")
        weights = np.asarray(self._weights, dtype=float).reshape(-1, 1)
        return (stacked * weights).sum(axis=0) / weights.sum()
