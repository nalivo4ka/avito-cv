"""Обучение с прогресс-баром, которое можно прерывать и продолжать.

Запускать руками:

    python scripts/train_interactive.py --architecture tiny --epochs 8

Ctrl+C в любой момент сохраняет состояние; повторный запуск той же команды продолжает с той же
эпохи, с теми же моментами оптимизатора и той же точкой на расписании скорости обучения.
Чтобы начать заново, добавьте `--restart` или удалите каталог прогона.

Состояние лежит в `artifacts/<имя прогона>/`: `session.pt` для продолжения и `best.pt` с весами
лучшей эпохи — именно его читает `scripts/predict.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from avitocv.config import DataConfig
from avitocv.data.factory import (
    ManifestDatasetFactory,
    MaterializedDatasetFactory,
    StoreShare,
    SyntheticDatasetFactory,
)
from avitocv.model.architecture import ModelCostMeter, ModelFactory, TinyNetConfig
from avitocv.training.evaluation import EvaluationTarget, Evaluator, LoaderSettings
from avitocv.training.loop import Trainer, TrainingConfig, TrainingInterrupted
from avitocv.training.reporting import RichProgressObserver

console = Console()

DEFAULT_EPOCH_SAMPLES = 600_000

REAL_MANIFEST = Path("data/real/validation_manifest.parquet")
CYRILLIC_MANIFEST = Path("data/real/cyrillic_manifest.parquet")
VALIDATION_SLICES = {
    "real_scene": (REAL_MANIFEST, "hiertext_scene"),
    "real_hand": (REAL_MANIFEST, "hiertext_handwritten"),
    "cyr_plate": (CYRILLIC_MANIFEST, "cyrillic_plate"),
    "cyr_hand": (CYRILLIC_MANIFEST, "cyrillic_handwriting"),
}


def build_targets(
    config: DataConfig,
    synthetic: SyntheticDatasetFactory,
    validation_store: Path | None,
) -> list[EvaluationTarget]:
    """Синтетика плюс все доступные реальные срезы; отсутствующие манифесты пропускаются."""
    # Готовое хранилище предпочтительнее генерации на лету: та же выборка каждый раз
    # и десятикратная разница в скорости оценки.
    if validation_store is not None and validation_store.exists():
        synthetic_dataset = MaterializedDatasetFactory(config).build_validation(validation_store)
    else:
        synthetic_dataset = synthetic.build_validation()
    targets = [EvaluationTarget("synthetic", synthetic_dataset)]
    manifests = ManifestDatasetFactory(config)
    for name, (path, slice_name) in VALIDATION_SLICES.items():
        if path.exists():
            targets.append(EvaluationTarget(name, manifests.build_validation(path, slice_name)))
    return targets


def build_training_dataset(config: DataConfig, arguments: argparse.Namespace, synthetic: SyntheticDatasetFactory):
    """Собирает обучающий набор: синтетика, реальные кропы или их смесь.

    Доля реальных задаётся явно. Смысл смеси в разделении труда: реальные фотографии учат
    внешнему виду настоящей съёмки и настоящих боксов детектора, синтетика — кириллице,
    которой в открытых датасетах сцены практически нет.
    """
    store = arguments.store
    if store is None or str(store) in ("", "."):
        return synthetic.build_training(), "генерация на лету"

    factory = MaterializedDatasetFactory(config)
    real_stores = [path for path in (arguments.real_store or []) if path.exists()]
    if not real_stores:
        dataset = factory.build_training(store, arguments.train_length or None)
        return dataset, f"синтетика: {len(dataset)}"

    total = arguments.train_length or DEFAULT_EPOCH_SAMPLES
    real_count = int(total * arguments.real_share)
    # Реальная доля делится между хранилищами поровну: они разной природы (строки сцен, слова
    # и склеенные строки), и пропорционально размеру перевес ушёл бы к самому крупному.
    per_store = real_count // len(real_stores)
    shares = [StoreShare(store, total - per_store * len(real_stores))]
    shares += [
        StoreShare(path, per_store, is_real=True, is_height_matched=not arguments.no_height_match,
                   epoch_span=arguments.epochs)
        for path in real_stores
    ]
    dataset = factory.build_mixed_training(shares)
    matching = "" if arguments.no_height_match else ", геометрия выровнена под тест"
    listing = " + ".join(f"{path.name} {per_store}" for path in real_stores)
    return dataset, f"синтетика {shares[0].sample_count} + {listing}{matching}"


def save_history(path: Path, architecture: str, cost, training_config: TrainingConfig, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "architecture": architecture,
                "cost": {"parameters": cost.parameter_count, "macs": cost.multiply_accumulates},
                "config": {key: str(value) for key, value in asdict(training_config).items()},
                "epochs": [asdict(record) for record in history],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


DEFAULT_REAL_STORES = (Path("data/generated/real"),)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение с прогресс-баром и возобновлением")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--store", type=Path, default=Path("data/generated/train"),
                        help="хранилище синтетических кропов; без него генерация на лету")
    parser.add_argument("--real-store", type=Path, action="append", default=None,
                        help="хранилище реальных кропов; можно повторять, доля делится поровну")
    parser.add_argument("--real-share", type=float, default=0.5,
                        help="доля реальных кропов в эпохе")
    parser.add_argument("--no-height-match", action="store_true",
                        help="не выравнивать геометрию реальных кропов под тестовую")
    parser.add_argument("--architecture", default="tiny", choices=("tiny", "tiny_tall", "tiny_mean", "tiny_dense"))
    parser.add_argument("--run-name", default=None, help="имя каталога прогона, по умолчанию совпадает с архитектурой")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--validation-store", type=Path, default=Path("data/generated/val"),
                        help="хранилище синтетической валидации")
    parser.add_argument("--eval-limit", type=int, default=6000, help="кропов на срез при оценке, 0 — весь срез")
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--input-height", type=int, default=0,
                        help="переопределить высоту входа сети")
    parser.add_argument("--input-width", type=int, default=0,
                        help="переопределить ширину входа сети")
    parser.add_argument("--train-length", type=int, default=0, help="ограничить размер эпохи")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restart", action="store_true", help="начать заново, забыв сохранённое состояние")
    arguments = parser.parse_args()
    if arguments.real_store is None:
        arguments.real_store = list(DEFAULT_REAL_STORES)
    return arguments


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    config = DataConfig.from_yaml(arguments.config)
    if arguments.train_length:
        config = replace(config, train_length=arguments.train_length)
    if arguments.input_height or arguments.input_width:
        config = replace(config, preprocess=replace(
            config.preprocess,
            height=arguments.input_height or config.preprocess.height,
            width=arguments.input_width or config.preprocess.width,
        ))

    synthetic = SyntheticDatasetFactory(config)
    dataset, composition = build_training_dataset(config, arguments, synthetic)
    targets = build_targets(config, synthetic, arguments.validation_store)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Высота входа задаётся конфигом данных: от неё зависит размер головы сети.
    factory = ModelFactory(TinyNetConfig(input_height=config.preprocess.height))
    model = factory.create(arguments.architecture)
    shape = (config.preprocess.channel_count, config.preprocess.height, config.preprocess.width)
    cost = ModelCostMeter().measure(model, shape)

    output_dir = Path("artifacts") / (arguments.run_name or arguments.architecture)
    if arguments.restart and output_dir.exists():
        for leftover in output_dir.glob("*.pt"):
            leftover.unlink()
        console.print(f"[yellow]состояние прогона в {output_dir} удалено, начинаем заново[/yellow]")

    training_config = TrainingConfig(
        architecture=arguments.architecture,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        worker_count=arguments.workers,
        learning_rate=arguments.learning_rate,
        output_dir=output_dir,
    )
    evaluator = Evaluator(device, LoaderSettings(worker_count=arguments.eval_workers, limit=arguments.eval_limit))
    trainer = Trainer(model, training_config, device, evaluator, RichProgressObserver())

    console.print(f"[dim]{cost.describe()}[/dim]")
    console.print(f"[dim]состав эпохи: {composition}  ·  срезов валидации: {len(targets)}[/dim]")
    try:
        history = trainer.fit(dataset, targets)
    except TrainingInterrupted:
        return

    console.print(trainer.evaluate(targets).describe())
    save_history(output_dir / "history.json", arguments.architecture, cost, training_config, history)
    console.print(f"[green]веса лучшей эпохи:[/green] {trainer.store.best_path}")


if __name__ == "__main__":
    main()
