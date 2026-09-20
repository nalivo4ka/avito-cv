"""Метрики задачи и их накопление по срезам.

Основная метрика — `1 - Brier Score`. Brier это средний квадрат отклонения предсказанной
вероятности от метки, то есть строго правильная функция потерь: она наказывает не только
ошибки, но и переуверенность. Поэтому рядом считается accuracy — разрыв между ними показывает,
теряем ли мы на калибровке или на самих ошибках.

Веса кропов учитываются везде: валидационный набор из реальных фотографий выровнен по высоте
importance-весами, и невзвешенное среднее отвечало бы на вопрос про другое распределение.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROBABILITY_EPSILON = 1e-7
DECISION_THRESHOLD = 0.5
DEFAULT_BIN_COUNT = 20


@dataclass(frozen=True)
class OrientationMetrics:
    """Результат замера на одном наборе кропов."""

    score: float
    brier: float
    accuracy: float
    log_loss: float
    mean_probability: float
    sample_count: int
    effective_count: float

    def describe(self) -> str:
        return (
            f"score {self.score:.5f}  brier {self.brier:.5f}  acc {self.accuracy:.4f}"
            f"  logloss {self.log_loss:.4f}  p̄ {self.mean_probability:.3f}"
            f"  n {self.sample_count} (эфф. {self.effective_count:.0f})"
        )


class MetricsAccumulator:
    """Накапливает предсказания и считает метрики одним проходом без хранения всей истории."""

    def __init__(self) -> None:
        self._weight_total = 0.0
        self._weight_square_total = 0.0
        self._squared_error = 0.0
        self._correct = 0.0
        self._log_loss = 0.0
        self._probability_total = 0.0
        self._count = 0

    def add(self, probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if not (len(probabilities) == len(labels) == len(weights)):
            raise ValueError("предсказания, метки и веса должны быть одной длины")
        safe = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        self._weight_total += float(weights.sum())
        self._weight_square_total += float(np.sum(weights ** 2))
        self._squared_error += float(np.sum(weights * (probabilities - labels) ** 2))
        self._correct += float(np.sum(weights * ((probabilities >= DECISION_THRESHOLD) == (labels >= DECISION_THRESHOLD))))
        self._log_loss -= float(np.sum(weights * (labels * np.log(safe) + (1.0 - labels) * np.log(1.0 - safe))))
        self._probability_total += float(np.sum(weights * probabilities))
        self._count += len(probabilities)

    def result(self) -> OrientationMetrics:
        if self._weight_total <= 0.0:
            raise ValueError("нечего измерять: накоплено ноль веса")
        brier = self._squared_error / self._weight_total
        return OrientationMetrics(
            score=1.0 - brier,
            brier=brier,
            accuracy=self._correct / self._weight_total,
            log_loss=self._log_loss / self._weight_total,
            mean_probability=self._probability_total / self._weight_total,
            sample_count=self._count,
            effective_count=self._weight_total ** 2 / max(self._weight_square_total, PROBABILITY_EPSILON),
        )


@dataclass(frozen=True)
class ConsistencyReport:
    """Безметочная диагностика: насколько модель согласована с симметрией задачи.

    Для любого кропа ровно одна из ориентаций верна, поэтому `p(x) + p(rot180(x))` обязано
    равняться единице. Отклонение считается без меток и потому измеримо прямо на тестовой
    выборке — это главный инструмент, чтобы не тратить попытки отправки вслепую.
    """

    mean_absolute_deviation: float
    max_absolute_deviation: float
    mean_probability: float
    bimodality: float
    sample_count: int

    @classmethod
    def from_pairs(cls, forward: np.ndarray, flipped: np.ndarray) -> "ConsistencyReport":
        deviation = np.abs(np.asarray(forward) + np.asarray(flipped) - 1.0)
        symmetric = (np.asarray(forward) + 1.0 - np.asarray(flipped)) / 2.0
        return cls(
            mean_absolute_deviation=float(deviation.mean()),
            max_absolute_deviation=float(deviation.max()),
            mean_probability=float(symmetric.mean()),
            bimodality=float(np.mean(np.abs(symmetric - DECISION_THRESHOLD) > 0.4)),
            sample_count=len(deviation),
        )

    def describe(self) -> str:
        return (
            f"несогласованность {self.mean_absolute_deviation:.4f}"
            f" (макс {self.max_absolute_deviation:.3f}),"
            f" p̄ {self.mean_probability:.3f},"
            f" уверенных {self.bimodality:.1%},"
            f" n {self.sample_count}"
        )


@dataclass(frozen=True)
class BrierDecomposition:
    """Разложение Brier на составляющие по Мёрфи.

        Brier = ненадёжность - разрешающая способность + неопределённость

    Это не общая диагностика, а разбор именно нашей метрики, и он отвечает на вопрос, который
    иначе приходится угадывать: мы теряем на нечестных вероятностях или на неумении различать
    классы. Болезни разные и лечатся по-разному.

    **Ненадёжность** — насколько предсказанная вероятность расходится с наблюдаемой частотой
    внутри своей корзины. Лечится калибровкой, то есть почти бесплатно.

    **Разрешающая способность** — насколько уверенно модель разводит классы; чем больше,
    тем лучше, поэтому она вычитается. Растёт только от лучших признаков, данных или модели.

    **Неопределённость** зависит только от разметки и не уменьшается ничем: при сбалансированной
    выборке она равна 0.25. Это и есть тот самый Brier константного предсказания 0.5.
    """

    reliability: float
    resolution: float
    uncertainty: float
    bin_count: int

    @property
    def brier(self) -> float:
        return self.reliability - self.resolution + self.uncertainty

    @property
    def calibration_headroom(self) -> float:
        """Сколько Brier можно вернуть идеальной калибровкой, не трогая саму модель."""
        return self.reliability

    def describe(self) -> str:
        return (
            f"ненадёжность {self.reliability:.5f}"
            f"  разрешение {self.resolution:.5f}"
            f"  неопределённость {self.uncertainty:.5f}"
        )


class BrierDecomposer:
    """Раскладывает Brier, разбивая предсказания на корзины по уверенности."""

    def __init__(self, bin_count: int = DEFAULT_BIN_COUNT) -> None:
        if bin_count < 2:
            raise ValueError(f"нужно хотя бы две корзины, запрошено {bin_count}")
        self._bin_count = bin_count

    def decompose(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> BrierDecomposition:
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
        labels = np.asarray(labels, dtype=np.float64)
        sample_weights = np.ones_like(labels) if weights is None else np.asarray(weights, dtype=np.float64)
        total_weight = sample_weights.sum()
        if total_weight <= 0.0:
            raise ValueError("суммарный вес должен быть положительным")

        base_rate = float(np.sum(sample_weights * labels) / total_weight)
        edges = np.linspace(0.0, 1.0, self._bin_count + 1)
        # Правый край последней корзины включаем, иначе p = 1 выпадает из разбиения.
        index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, self._bin_count - 1)

        reliability, resolution = 0.0, 0.0
        for position in range(self._bin_count):
            selected = index == position
            bin_weight = float(sample_weights[selected].sum())
            if bin_weight <= 0.0:
                continue
            mean_probability = float(np.sum(sample_weights[selected] * probabilities[selected]) / bin_weight)
            observed_rate = float(np.sum(sample_weights[selected] * labels[selected]) / bin_weight)
            share = bin_weight / total_weight
            reliability += share * (mean_probability - observed_rate) ** 2
            resolution += share * (observed_rate - base_rate) ** 2

        return BrierDecomposition(
            reliability=reliability,
            resolution=resolution,
            uncertainty=base_rate * (1.0 - base_rate),
            bin_count=self._bin_count,
        )
