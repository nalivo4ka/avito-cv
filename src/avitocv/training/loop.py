"""Цикл обучения классификатора ориентации.

Функция потерь — взвешенный BCE по логитам. Именно логиты, а не вероятности: `BCEWithLogits`
считается численно устойчиво и не требует клампов у краёв, где как раз живут уверенные
предсказания.

Эпоха здесь виртуальная: обучающий набор кропов фиксирован, но поворот и кодековые деградации
разыгрываются заново на каждой эпохе, поэтому одни и те же кропы каждый раз приходят по-новому.
Смена эпохи меняет сид этого розыгрыша.

Цикл ничего не знает ни про отображение (за это отвечает `TrainingObserver`), ни про то, как
хранится состояние (за это отвечает `SessionStore`). Благодаря этому один и тот же цикл
работает и в фоновом прогоне, и в интерактивном скрипте с прогресс-баром, и прерванное
обучение продолжается ровно с того места, где остановилось.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from avitocv.data.datasets import OrientationDataset
from avitocv.model.inference import SymmetricPredictor, TemperatureCalibrator, TemperatureScaler
from avitocv.training.checkpoint import SessionStore, TrainingSession, TrainingState
from avitocv.training.evaluation import EvaluationReport, EvaluationTarget, Evaluator, PredictionBatch
from avitocv.training.observers import RunDescription, SilentObserver, TrainingObserver

GRADIENT_CLIP_NORM = 5.0


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры обучения."""

    architecture: str = "tiny"
    epochs: int = 8
    batch_size: int = 256
    worker_count: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.03
    is_amp_enabled: bool = True
    calibration_slice: str = "real_scene"
    shuffle_seed: int = 17
    output_dir: Path = Path("artifacts/tiny")


@dataclass
class EpochRecord:
    """Что произошло за одну эпоху."""

    index: int
    loss: float
    seconds: float
    scores: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0

    def describe(self) -> str:
        scores = "  ".join(f"{name} {value:.5f}" for name, value in self.scores.items())
        return f"эпоха {self.index}  loss {self.loss:.4f}  {self.seconds:.0f}с  {scores}"


class TrainingInterrupted(Exception):
    """Обучение прервано пользователем; состояние уже сохранено и прогон можно продолжить."""


class Trainer:
    """Обучает модель, снимает метрики по срезам и умеет продолжать прерванный прогон."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        evaluator: Evaluator,
        observer: TrainingObserver | None = None,
    ) -> None:
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._evaluator = evaluator
        self._observer = observer or SilentObserver()
        self._criterion = nn.BCEWithLogitsLoss(reduction="none")
        self._store = SessionStore(config.output_dir)
        self._scaler = TemperatureScaler()

    @property
    def store(self) -> SessionStore:
        return self._store

    def fit(self, dataset: OrientationDataset, targets: list[EvaluationTarget]) -> list[EpochRecord]:
        loader = self._build_loader(dataset)
        session = self._build_session(len(loader))
        state = self._restore_if_possible(session)
        history = [EpochRecord(**record) for record in state.history]

        self._observer.on_run_start(RunDescription(
            architecture=self._config.architecture,
            total_epochs=self._config.epochs,
            start_epoch=state.epoch,
            steps_per_epoch=len(loader),
            samples_per_epoch=len(loader) * self._config.batch_size,
            parameter_count=sum(parameter.numel() for parameter in self._model.parameters()),
            device=str(self._device),
            tracked_slice=self._config.calibration_slice,
        ))

        try:
            for epoch in range(state.epoch, self._config.epochs):
                history.append(self._run_single_epoch(epoch, loader, session, dataset, targets))
                state.epoch = epoch + 1
                state.history = [asdict(record) for record in history]
                self._store.save_session(session)
        except KeyboardInterrupt:
            self._store.save_session(session)
            self._observer.on_interrupt(state.epoch)
            raise TrainingInterrupted(f"прервано после эпохи {state.epoch}") from None

        self._observer.on_run_end(history, state.best_score)
        return history

    def evaluate(self, targets: list[EvaluationTarget]) -> EvaluationReport:
        self._model.eval()
        predictions = self._evaluator.collect_all(SymmetricPredictor(self._model), targets)
        self._scaler = self._calibrate(predictions.get(self._config.calibration_slice))
        self._model.train()
        return EvaluationReport(self._without_calibration_half(predictions), self._scaler)

    def _calibrate(self, calibration: PredictionBatch | None) -> TemperatureScaler:
        """Температура подбирается на первой половине среза, метрика считается на второй.

        Без разделения оценка отслеживаемого среза завышена: калибровка подогнана ровно под те
        кропы, по которым потом считается балл. Замер показал разницу около 0.015 — больше, чем
        отличаются последние варианты модели друг от друга, то есть достаточно, чтобы выбрать
        не тот чекпоинт.
        """
        if calibration is None:
            return TemperatureScaler()
        middle = max(1, len(calibration.logits) // 2)
        return TemperatureCalibrator().fit(
            calibration.logits[:middle], calibration.labels[:middle], calibration.weights[:middle]
        )

    def _without_calibration_half(self, predictions: dict[str, PredictionBatch]) -> dict[str, PredictionBatch]:
        tracked = predictions.get(self._config.calibration_slice)
        if tracked is None:
            return predictions
        middle = max(1, len(tracked.logits) // 2)
        held_out = PredictionBatch(
            logits=tracked.logits[middle:],
            labels=tracked.labels[middle:],
            weights=tracked.weights[middle:],
        )
        return {**predictions, self._config.calibration_slice: held_out}

    def _run_single_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        session: TrainingSession,
        dataset: OrientationDataset,
        targets: list[EvaluationTarget],
    ) -> EpochRecord:
        dataset.set_epoch(epoch)
        self._observer.on_epoch_start(epoch, self._config.epochs, len(loader))
        started = time.perf_counter()
        loss = self._run_batches(loader, session)

        self._observer.on_evaluation_start()
        report = self.evaluate(targets)
        record = EpochRecord(
            index=epoch,
            loss=loss,
            seconds=time.perf_counter() - started,
            scores={name: metrics.score for name, metrics in report.metrics.items()},
            temperature=self._scaler.temperature,
        )
        self._observer.on_epoch_end(record, self._remember_if_best(session.state, report))
        return record

    def _remember_if_best(self, state: TrainingState, report: EvaluationReport) -> bool:
        score = report.score_of(self._config.calibration_slice)
        if score <= state.best_score:
            return False
        state.best_score = score
        state.temperature = self._scaler.temperature
        self._store.save_best(self._model, self._scaler.temperature, score, self._config.architecture)
        return True

    def _restore_if_possible(self, session: TrainingSession) -> TrainingState:
        if not self._store.has_session:
            return session.state
        session.state = self._store.restore_session(session, self._device)
        return session.state

    def _build_session(self, steps_per_epoch: int) -> TrainingSession:
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        return TrainingSession(
            model=self._model,
            optimizer=optimizer,
            schedule=self._build_schedule(optimizer, steps_per_epoch),
            grad_scaler=torch.amp.GradScaler("cuda", enabled=self._is_amp_active),
            state=TrainingState(),
        )

    @property
    def _is_amp_active(self) -> bool:
        return self._config.is_amp_enabled and self._device.type == "cuda"

    def _build_loader(self, dataset: OrientationDataset) -> DataLoader:
        # Перемешивание обязательно с тех пор, как в обучение попали реальные кропы: в манифесте
        # они идут подряд по снимкам, и без перемешивания весь батч приходил бы с одной
        # фотографии — с общим фоном, шрифтом и освещением.
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self._config.shuffle_seed),
            num_workers=self._config.worker_count,
            persistent_workers=self._config.worker_count > 0,
            prefetch_factor=6 if self._config.worker_count > 0 else None,
            drop_last=True,
            pin_memory=self._device.type == "cuda",
        )

    def _build_schedule(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
        total_steps = steps_per_epoch * self._config.epochs
        warmup_steps = max(1, int(total_steps * self._config.warmup_fraction))

        def factor(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def _run_batches(self, loader: DataLoader, session: TrainingSession) -> float:
        self._model.train()
        total_loss, seen = 0.0, 0
        for step, batch in enumerate(loader, start=1):
            images = batch.image.to(self._device, non_blocking=True)
            labels = batch.label.to(self._device, non_blocking=True)
            weights = batch.weight.to(self._device, non_blocking=True)
            session.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self._is_amp_active):
                loss = (self._criterion(self._model(images), labels) * weights).sum() / weights.sum()
            session.grad_scaler.scale(loss).backward()
            session.grad_scaler.unscale_(session.optimizer)
            nn.utils.clip_grad_norm_(self._model.parameters(), GRADIENT_CLIP_NORM)
            session.grad_scaler.step(session.optimizer)
            session.grad_scaler.update()
            session.schedule.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
            self._observer.on_batch(step, total_loss / seen, session.schedule.get_last_lr()[0])
        return total_loss / max(seen, 1)
