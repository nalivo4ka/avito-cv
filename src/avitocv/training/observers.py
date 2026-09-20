"""Наблюдение за ходом обучения.

Цикл обучения не должен ничего знать про то, как его показывают: в фоновом прогоне вывод
мешает, в интерактивном нужен прогресс-бар. Поэтому отображение вынесено за интерфейс
`TrainingObserver`, а цикл только сообщает о событиях.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RunDescription:
    """Что предстоит сделать за прогон."""

    architecture: str
    total_epochs: int
    start_epoch: int
    steps_per_epoch: int
    samples_per_epoch: int
    parameter_count: int
    device: str
    tracked_slice: str = "real_scene"

    @property
    def is_resumed(self) -> bool:
        return self.start_epoch > 0


class TrainingObserver(ABC):
    """Получает события цикла обучения. Реализации решают, показывать их или нет."""

    def on_run_start(self, description: RunDescription) -> None:
        return None

    def on_epoch_start(self, epoch: int, total_epochs: int, steps: int) -> None:
        return None

    def on_batch(self, step: int, loss: float, learning_rate: float) -> None:
        return None

    def on_evaluation_start(self) -> None:
        return None

    @abstractmethod
    def on_epoch_end(self, record, is_best: bool) -> None:
        raise NotImplementedError

    def on_run_end(self, history: list, best_score: float) -> None:
        return None

    def on_interrupt(self, epoch: int) -> None:
        return None


class SilentObserver(TrainingObserver):
    """Ничего не показывает: для тестов и фоновых прогонов."""

    def on_epoch_end(self, record, is_best: bool) -> None:
        return None


class PlainObserver(TrainingObserver):
    """Одна строка на эпоху: для логов, где разметка только мешает."""

    def on_run_start(self, description: RunDescription) -> None:
        print(f"{description.architecture}: {description.parameter_count} параметров, {description.device}")
        if description.is_resumed:
            print(f"продолжение с эпохи {description.start_epoch}")

    def on_epoch_end(self, record, is_best: bool) -> None:
        marker = " *" if is_best else ""
        print(record.describe() + marker)

    def on_run_end(self, history: list, best_score: float) -> None:
        print(f"лучший score: {best_score:.5f}")
