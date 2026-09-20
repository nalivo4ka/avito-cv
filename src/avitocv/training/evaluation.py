"""Оценка модели на именованных наборах.

Метрики считаются по слоям валидации отдельно и никогда не усредняются в одно число: слои
отвечают на разные вопросы — синтетика на сходимость, реальные фотографии на разрыв домена,
кириллица на письменность. Среднее по ним скрыло бы ровно то, что мы хотим видеть.

Логиты собираются целиком, а не сворачиваются на лету: по ним потом подбирается температура,
и повторный прогон сети ради этого был бы лишним.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from avitocv.model.inference import LogitPredictor, TemperatureScaler
from avitocv.training.metrics import MetricsAccumulator, OrientationMetrics


@dataclass(frozen=True)
class EvaluationTarget:
    """Именованный набор кропов: имя попадает в отчёт как название среза."""

    name: str
    dataset: Dataset


@dataclass(frozen=True)
class PredictionBatch:
    """Собранные по набору логиты, метки и веса."""

    logits: np.ndarray
    labels: np.ndarray
    weights: np.ndarray

    def metrics(self, scaler: TemperatureScaler) -> OrientationMetrics:
        accumulator = MetricsAccumulator()
        accumulator.add(scaler.probabilities(self.logits), self.labels, self.weights)
        return accumulator.result()


@dataclass(frozen=True)
class LoaderSettings:
    """Как читать данные при оценке.

    Воркеров по умолчанию нет намеренно. Оценка идёт по нескольким небольшим срезам, загрузчик
    для каждого создаётся заново, а запуск процесса на Windows стоит секунды: замер дал 1.6 с
    без воркеров против 13.2 с с четырьмя на тех же 2048 кропах.
    """

    batch_size: int = 512
    worker_count: int = 0
    limit: int = 0

    @property
    def is_limited(self) -> bool:
        return self.limit > 0


class Evaluator:
    """Прогоняет модель по набору и возвращает логиты вместе с метками и весами."""

    def __init__(self, device: torch.device, settings: LoaderSettings = LoaderSettings()) -> None:
        self._device = device
        self._settings = settings

    @torch.no_grad()
    def collect(self, predictor: LogitPredictor, dataset: Dataset) -> PredictionBatch:
        loader = DataLoader(
            dataset,
            batch_size=self._settings.batch_size,
            num_workers=self._settings.worker_count,
            persistent_workers=self._settings.worker_count > 0,
        )
        logits, labels, weights = [], [], []
        seen = 0
        for batch in loader:
            images = batch.image.to(self._device, non_blocking=True)
            logits.append(predictor.logits(images).float().cpu().numpy())
            labels.append(batch.label.numpy())
            weights.append(batch.weight.numpy())
            seen += len(batch.label)
            if self._settings.is_limited and seen >= self._settings.limit:
                break
        return PredictionBatch(
            logits=np.concatenate(logits),
            labels=np.concatenate(labels),
            weights=np.concatenate(weights),
        )

    def collect_all(self, predictor: LogitPredictor, targets: list[EvaluationTarget]) -> dict[str, PredictionBatch]:
        return {target.name: self.collect(predictor, target.dataset) for target in targets}


class EvaluationReport:
    """Метрики по срезам плюс печать в читаемом виде."""

    def __init__(self, predictions: dict[str, PredictionBatch], scaler: TemperatureScaler) -> None:
        self._scaler = scaler
        self._metrics = {name: batch.metrics(scaler) for name, batch in predictions.items()}

    @property
    def metrics(self) -> dict[str, OrientationMetrics]:
        return dict(self._metrics)

    def score_of(self, name: str) -> float:
        return self._metrics[name].score

    def describe(self) -> str:
        lines = [f"температура {self._scaler.temperature:.3f}"]
        width = max(len(name) for name in self._metrics)
        for name, metrics in self._metrics.items():
            lines.append(f"  {name:<{width}}  {metrics.describe()}")
        return "\n".join(lines)
