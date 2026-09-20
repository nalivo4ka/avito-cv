from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from avitocv.data.datasets import CenterWindowFit, ImagePreprocessor, OrientationDataset, PreprocessConfig
from avitocv.data.degradation import DegradationPipeline
from avitocv.data.manifest import CropManifest, CropRecord
from avitocv.data.sampling import SeedScheme
from avitocv.data.sources import ManifestTextLineSource
from avitocv.model.architecture import ModelFactory
from avitocv.training.checkpoint import SessionStore, TrainingSession, TrainingState
from avitocv.training.evaluation import EvaluationTarget, Evaluator, LoaderSettings
from avitocv.training.loop import EpochRecord, Trainer, TrainingConfig
from avitocv.training.observers import PlainObserver, RunDescription, SilentObserver

PHOTO_SIZE = (320, 120)
CROP_COUNT = 24
CPU = torch.device("cpu")


def _build_session(learning_rate: float = 1e-3) -> TrainingSession:
    model = ModelFactory().create("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return TrainingSession(
        model=model,
        optimizer=optimizer,
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        grad_scaler=torch.amp.GradScaler("cuda", enabled=False),
        state=TrainingState(),
    )


def _take_one_step(session: TrainingSession) -> None:
    session.model(torch.zeros(4, 1, 32, 192)).mean().backward()
    session.optimizer.step()
    session.schedule.step()


@pytest.fixture
def toy_dataset(tmp_path):
    """Крошечный набор реальных кропов: несколько боксов из одной картинки."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(PHOTO_SIZE[1], PHOTO_SIZE[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "photo.png")
    records = [
        CropRecord(image_path="photo.png", left=4, top=4 + (index % 3) * 20, width=120, height=24)
        for index in range(CROP_COUNT)
    ]
    source = ManifestTextLineSource.from_manifest(CropManifest.from_records(records), tmp_path)
    return OrientationDataset(
        source=source,
        codec_stage=DegradationPipeline([]),
        preprocessor=ImagePreprocessor(PreprocessConfig(), CenterWindowFit()),
        seed_scheme=SeedScheme(0),
    )


class TestSessionStore:
    def test_saved_session_restores_the_epoch_and_best_score(self, tmp_path) -> None:
        store = SessionStore(tmp_path)
        session = _build_session()
        session.state = TrainingState(epoch=3, best_score=0.91, temperature=1.4, history=[{"index": 0}])
        store.save_session(session)

        restored = store.restore_session(_build_session(), CPU)
        assert (restored.epoch, restored.best_score) == (3, 0.91)
        assert restored.temperature == pytest.approx(1.4)
        assert restored.history == [{"index": 0}]

    def test_optimizer_state_survives_the_round_trip(self, tmp_path) -> None:
        """Без моментов Adam продолжение было бы новым прогоном с тёплого старта."""
        store = SessionStore(tmp_path)
        session = _build_session()
        _take_one_step(session)
        store.save_session(session)

        fresh = _build_session()
        assert not fresh.optimizer.state_dict()["state"], "у нового оптимизатора моментов быть не должно"
        store.restore_session(fresh, CPU)
        assert fresh.optimizer.state_dict()["state"], "моменты обязаны восстановиться"

    def test_schedule_position_survives_the_round_trip(self, tmp_path) -> None:
        store = SessionStore(tmp_path)
        session = _build_session()
        for _ in range(5):
            _take_one_step(session)
        store.save_session(session)

        fresh = _build_session()
        store.restore_session(fresh, CPU)
        assert fresh.schedule.state_dict()["last_epoch"] == session.schedule.state_dict()["last_epoch"]

    def test_absent_session_is_reported(self, tmp_path) -> None:
        assert not SessionStore(tmp_path / "absent").has_session

    def test_best_checkpoint_keeps_architecture_and_temperature(self, tmp_path) -> None:
        store = SessionStore(tmp_path)
        store.save_best(ModelFactory().create("tiny"), temperature=1.7, score=0.93, architecture="tiny")
        payload = torch.load(store.best_path, map_location="cpu", weights_only=True)
        assert payload["architecture"] == "tiny"
        assert payload["temperature"] == pytest.approx(1.7)
        assert payload["score"] == pytest.approx(0.93)


class TestTrainingState:
    def test_fresh_state_is_recognized(self) -> None:
        assert TrainingState().is_fresh

    def test_state_with_history_is_not_fresh(self) -> None:
        assert not TrainingState(epoch=2, history=[{"index": 0}]).is_fresh

    def test_dictionary_round_trip_is_lossless(self) -> None:
        state = TrainingState(epoch=7, best_score=0.88, temperature=2.1, history=[{"a": 1}])
        assert TrainingState.from_dict(state.to_dict()) == state


class TestResumedTraining:
    def _train(self, dataset, directory, epochs: int) -> list:
        config = TrainingConfig(
            architecture="tiny",
            epochs=epochs,
            batch_size=8,
            worker_count=0,
            calibration_slice="toy",
            output_dir=directory,
        )
        evaluator = Evaluator(CPU, LoaderSettings(batch_size=8, worker_count=0))
        trainer = Trainer(ModelFactory().create("tiny"), config, CPU, evaluator, SilentObserver())
        return trainer.fit(dataset, [EvaluationTarget("toy", dataset)])

    def test_second_run_continues_instead_of_restarting(self, toy_dataset, tmp_path) -> None:
        first = self._train(toy_dataset, tmp_path / "run", epochs=2)
        assert [record.index for record in first] == [0, 1]

        continued = self._train(toy_dataset, tmp_path / "run", epochs=4)
        assert [record.index for record in continued] == [0, 1, 2, 3]

    def test_completed_run_does_nothing_on_a_repeat(self, toy_dataset, tmp_path) -> None:
        self._train(toy_dataset, tmp_path / "run", epochs=2)
        repeated = self._train(toy_dataset, tmp_path / "run", epochs=2)
        assert [record.index for record in repeated] == [0, 1]

    def test_best_checkpoint_is_written(self, toy_dataset, tmp_path) -> None:
        self._train(toy_dataset, tmp_path / "run", epochs=1)
        assert SessionStore(tmp_path / "run").best_path.exists()


class TestObservers:
    def test_silent_observer_accepts_every_event(self) -> None:
        observer = SilentObserver()
        observer.on_run_start(RunDescription("tiny", 4, 0, 10, 2560, 21729, "cpu"))
        observer.on_epoch_start(0, 4, 10)
        observer.on_batch(1, 0.5, 1e-3)
        observer.on_evaluation_start()
        observer.on_epoch_end(None, False)
        observer.on_interrupt(2)

    def test_resumed_run_is_recognized(self) -> None:
        assert RunDescription("tiny", 4, 2, 10, 2560, 21729, "cpu").is_resumed
        assert not RunDescription("tiny", 4, 0, 10, 2560, 21729, "cpu").is_resumed

    def test_plain_observer_prints_the_epoch(self, capsys) -> None:
        PlainObserver().on_epoch_end(EpochRecord(index=0, loss=0.5, seconds=1.0, scores={"toy": 0.9}), True)
        assert "toy 0.90000" in capsys.readouterr().out


class TestPhotoCache:
    """Кэш распакованных снимков — причина десятикратного ускорения валидации."""

    def _source(self, tmp_path, crop_count: int) -> ManifestTextLineSource:
        Image.new("RGB", PHOTO_SIZE, (120, 130, 140)).save(tmp_path / "photo.png")
        records = [
            CropRecord(image_path="photo.png", left=0, top=index, width=100, height=20)
            for index in range(crop_count)
        ]
        return ManifestTextLineSource.from_manifest(CropManifest.from_records(records), tmp_path)

    def test_repeated_crops_do_not_redecode_the_photo(self, tmp_path, monkeypatch) -> None:
        source = self._source(tmp_path, crop_count=10)
        opened: list = []
        original = Image.open

        def counting_open(path, *args, **kwargs):
            opened.append(path)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Image, "open", counting_open)
        scheme = SeedScheme(0)
        for index in range(10):
            source.load_upright(index, scheme.rng_for(index))
        assert len(opened) == 1, f"снимок открыт {len(opened)} раз вместо одного"

    def test_cache_is_dropped_when_pickled_for_workers(self, tmp_path) -> None:
        source = self._source(tmp_path, crop_count=1)
        source.load_upright(0, SeedScheme(0).rng_for(0))
        revived = pickle.loads(pickle.dumps(source))
        assert revived.load_upright(0, SeedScheme(0).rng_for(0)).size[0] > 0


class TestHeldOutCalibration:
    """Температура не должна подбираться на тех же кропах, по которым считается балл."""

    def _trainer(self, directory) -> Trainer:
        config = TrainingConfig(architecture="tiny", epochs=1, batch_size=8, worker_count=0,
                                calibration_slice="toy", output_dir=directory)
        evaluator = Evaluator(CPU, LoaderSettings(batch_size=8, worker_count=0))
        return Trainer(ModelFactory().create("tiny"), config, CPU, evaluator, SilentObserver())

    def test_tracked_slice_is_scored_on_the_half_not_used_for_calibration(self, toy_dataset, tmp_path) -> None:
        trainer = self._trainer(tmp_path / "run")
        report = trainer.evaluate([EvaluationTarget("toy", toy_dataset)])
        assert report.metrics["toy"].sample_count == len(toy_dataset) // 2

    def test_other_slices_keep_every_sample(self, toy_dataset, tmp_path) -> None:
        trainer = self._trainer(tmp_path / "run")
        report = trainer.evaluate([
            EvaluationTarget("toy", toy_dataset),
            EvaluationTarget("other", toy_dataset),
        ])
        assert report.metrics["other"].sample_count == len(toy_dataset)

    def test_missing_calibration_slice_leaves_temperature_at_one(self, toy_dataset, tmp_path) -> None:
        trainer = self._trainer(tmp_path / "run")
        report = trainer.evaluate([EvaluationTarget("other", toy_dataset)])
        assert "температура 1.000" in report.describe()
