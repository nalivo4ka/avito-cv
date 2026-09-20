"""Сохранение и восстановление обучения.

Разделены два файла, потому что у них разные потребители.

`session.pt` — полное состояние прогона: веса, оптимизатор, расписание скорости обучения,
масштабировщик градиента, номер эпохи и история. Нужен, чтобы прерванное обучение
продолжилось ровно с того места, а не началось заново. Без состояния оптимизатора и
расписания продолжение было бы не продолжением, а новым прогоном с тёплого старта:
моменты Adam обнулились бы, а скорость обучения прыгнула бы обратно к началу косинуса.

`best.pt` — только веса лучшей эпохи и температура калибровки. Его читает инференс, и он
обязан оставаться маленьким: это тот файл, который поедет в репозиторий с решением.

Состояние сохраняется на границах эпох. Прерывание в середине эпохи откатывает её к началу —
при эпохе в несколько минут это дешевле, чем тащить состояние загрузчика данных.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

BEST_CHECKPOINT_NAME = "best.pt"
SESSION_CHECKPOINT_NAME = "session.pt"
WORST_POSSIBLE_SCORE = -1.0


@dataclass
class TrainingState:
    """Прогресс обучения: что уже сделано и какой результат лучший."""

    epoch: int = 0
    best_score: float = WORST_POSSIBLE_SCORE
    temperature: float = 1.0
    history: list[dict] = field(default_factory=list)

    @property
    def is_fresh(self) -> bool:
        return self.epoch == 0 and not self.history

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingState":
        return cls(
            epoch=int(payload["epoch"]),
            best_score=float(payload["best_score"]),
            temperature=float(payload["temperature"]),
            history=list(payload["history"]),
        )


@dataclass
class TrainingSession:
    """Изменяемые части прогона, которые нужно сохранять вместе."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LRScheduler
    grad_scaler: torch.amp.GradScaler
    state: TrainingState


class SessionStore:
    """Хранит полное состояние прогона и отдельно веса лучшей эпохи."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    @property
    def session_path(self) -> Path:
        return self._directory / SESSION_CHECKPOINT_NAME

    @property
    def best_path(self) -> Path:
        return self._directory / BEST_CHECKPOINT_NAME

    @property
    def has_session(self) -> bool:
        return self.session_path.exists()

    def save_session(self, session: TrainingSession) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": session.model.state_dict(),
                "optimizer": session.optimizer.state_dict(),
                "schedule": session.schedule.state_dict(),
                "grad_scaler": session.grad_scaler.state_dict(),
                "state": session.state.to_dict(),
            },
            self.session_path,
        )

    def restore_session(self, session: TrainingSession, device: torch.device) -> TrainingState:
        payload = torch.load(self.session_path, map_location=device, weights_only=False)
        session.model.load_state_dict(payload["model"])
        session.optimizer.load_state_dict(payload["optimizer"])
        session.schedule.load_state_dict(payload["schedule"])
        session.grad_scaler.load_state_dict(payload["grad_scaler"])
        return TrainingState.from_dict(payload["state"])

    def save_best(self, model: nn.Module, temperature: float, score: float, architecture: str) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "temperature": float(temperature),
                "score": float(score),
                "architecture": architecture,
            },
            self.best_path,
        )
