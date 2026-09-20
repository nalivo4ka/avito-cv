"""Разбирает, где именно модель теряет баллы.

Отвечает на три вопроса, которые иначе приходится угадывать.

**Теряем на калибровке или на различении?** Разложение Brier по Мёрфи: ненадёжность лечится
почти бесплатно температурой, разрешающая способность — только лучшими данными или моделью.

**Упёрлись в ёмкость или в домен?** Метрика на обучающей выборке рядом с валидационной. Если
обе низкие и близки — не хватает ёмкости, стоит усложнять сеть. Если обучающая заметно выше —
ёмкости достаточно, усложнение только усилит переобучение.

**Где сосредоточены потери?** Разбивка по высоте кропа. Если провал в мелких кропах, помогать
будет разрешение входа, а не число параметров.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from avitocv.config import DataConfig
from avitocv.data.factory import (
    ManifestDatasetFactory,
    MaterializedDatasetFactory,
    StoreShare,
)
from avitocv.model.architecture import ModelCostMeter, ModelFactory, TinyNetConfig
from avitocv.model.inference import SymmetricPredictor, TemperatureScaler
from avitocv.training.metrics import BrierDecomposer, MetricsAccumulator

console = Console()

REAL_MANIFEST = Path("data/real/validation_manifest.parquet")
CYRILLIC_MANIFEST = Path("data/real/cyrillic_manifest.parquet")
HEIGHT_EDGES = (0, 20, 28, 40, 60, 100, 10_000)
ASPECT_EDGES = (0.0, 2.5, 3.5, 5.0, 7.5, 12.0, 1000.0)


@dataclass(frozen=True)
class SlicePredictions:
    """Предсказания по одному срезу вместе с геометрией кропов."""

    name: str
    probabilities: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    heights: np.ndarray
    aspects: np.ndarray

    def metrics(self):
        accumulator = MetricsAccumulator()
        accumulator.add(self.probabilities, self.labels, self.weights)
        return accumulator.result()

    def subset(self, selected: np.ndarray) -> "SlicePredictions":
        return SlicePredictions(
            name=self.name,
            probabilities=self.probabilities[selected],
            labels=self.labels[selected],
            weights=self.weights[selected],
            heights=self.heights[selected],
            aspects=self.aspects[selected],
        )


class SliceCollector:
    """Прогоняет модель по срезу и собирает предсказания вместе с высотой каждого кропа."""

    def __init__(self, device: torch.device, scaler: TemperatureScaler, batch_size: int, limit: int) -> None:
        self._device = device
        self._scaler = scaler
        self._batch_size = batch_size
        self._limit = limit

    @torch.no_grad()
    def collect(self, name: str, dataset, model) -> SlicePredictions:
        predictor = SymmetricPredictor(model)
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=0)
        logits, labels, weights, heights, aspects = [], [], [], [], []
        seen = 0
        for position, batch in enumerate(loader):
            images = batch.image.to(self._device, non_blocking=True)
            logits.append(predictor.logits(images).float().cpu().numpy())
            labels.append(batch.label.numpy())
            weights.append(batch.weight.numpy())
            for offset in range(len(batch.label)):
                crop = dataset.load_crop(position * self._batch_size + offset)[0]
                heights.append(crop.height)
                aspects.append(crop.width / crop.height)
            seen += len(batch.label)
            if self._limit and seen >= self._limit:
                break
        return SlicePredictions(
            name=name,
            probabilities=self._scaler.probabilities(np.concatenate(logits)),
            labels=np.concatenate(labels),
            weights=np.concatenate(weights),
            heights=np.asarray(heights, dtype=float),
            aspects=np.asarray(aspects, dtype=float),
        )


def decomposition_table(slices: list[SlicePredictions]) -> Table:
    table = Table(title="Разложение Brier: где теряются баллы", header_style="bold magenta")
    table.add_column("срез", style="bold")
    table.add_column("score", justify="right")
    table.add_column("ненадёжность", justify="right")
    table.add_column("разрешение", justify="right")
    table.add_column("запас калибровки", justify="right")
    decomposer = BrierDecomposer()
    for item in slices:
        parts = decomposer.decompose(item.probabilities, item.labels, item.weights)
        table.add_row(
            item.name,
            f"{item.metrics().score:.5f}",
            f"{parts.reliability:.5f}",
            f"{parts.resolution:.5f}",
            f"{parts.calibration_headroom:.5f}",
        )
    return table


def height_table(slices: list[SlicePredictions]) -> Table:
    table = Table(title="Потери по высоте кропа", header_style="bold magenta")
    table.add_column("высота, px", style="bold")
    for item in slices:
        table.add_column(item.name, justify="right")
    for low, high in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:]):
        cells = [f"{low}..{high}" if high < 10_000 else f"{low}+"]
        for item in slices:
            selected = (item.heights >= low) & (item.heights < high)
            cells.append(f"{item.subset(selected).metrics().score:.4f}" if selected.sum() > 30 else "—")
        table.add_row(*cells)
    return table


def aspect_table(slices: list[SlicePredictions]) -> Table:
    """Разбивка по пропорциям кропа.

    Короткий кроп это мало символов, то есть мало независимых свидетельств об ориентации.
    Если провал именно там, помогать будет не обзор строки, а способность вытянуть признак
    из отдельного глифа.
    """
    table = Table(title="Потери по пропорциям кропа", header_style="bold magenta")
    table.add_column("aspect", style="bold")
    for item in slices:
        table.add_column(item.name, justify="right")
    for low, high in zip(ASPECT_EDGES[:-1], ASPECT_EDGES[1:]):
        cells = [f"{low:.1f}..{high:.1f}" if high < 1000 else f"{low:.1f}+"]
        for item in slices:
            selected = (item.aspects >= low) & (item.aspects < high)
            cells.append(f"{item.subset(selected).metrics().score:.4f}" if selected.sum() > 30 else "—")
        table.add_row(*cells)
    return table


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Диагностика обученной модели")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tall_v2/best.pt"))
    parser.add_argument("--store", type=Path, default=Path("data/generated/train"))
    parser.add_argument("--real-store", type=Path, default=Path("data/generated/real"))
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--height-downsample", type=int, default=4,
                        help="прореживание высоты внутри сети, как при обучении")
    parser.add_argument("--input-height", type=int, default=0,
                        help="высота входа сети, если отличается от конфига")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = DataConfig.from_yaml(arguments.config)
    if arguments.input_height:
        config = replace(config, preprocess=replace(config.preprocess, height=arguments.input_height))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model = ModelFactory(TinyNetConfig(input_height=config.preprocess.height,
                              height_downsample=arguments.height_downsample)).create(payload["architecture"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    scaler = TemperatureScaler(float(payload["temperature"]))
    shape = (config.preprocess.channel_count, config.preprocess.height, config.preprocess.width)
    console.print(f"[dim]{ModelCostMeter().measure(model.cpu(), shape).describe()}  ·  T={scaler.temperature:.3f}[/dim]")
    model.to(device)

    materialized = MaterializedDatasetFactory(config)
    manifests = ManifestDatasetFactory(config)
    targets = [
        # Обучающая выборка идёт первой: сравнение с валидацией отвечает на вопрос про ёмкость.
        ("train_synth", materialized.build_mixed_training([StoreShare(arguments.store, arguments.limit)])),
        ("train_real", materialized.build_mixed_training(
            [StoreShare(arguments.real_store, arguments.limit, is_real=True)])),
        ("synthetic", materialized.build_validation(Path("data/generated/val"))),
        ("real_scene", manifests.build_validation(REAL_MANIFEST, "hiertext_scene")),
        ("real_hand", manifests.build_validation(REAL_MANIFEST, "hiertext_handwritten")),
        ("cyr_plate", manifests.build_validation(CYRILLIC_MANIFEST, "cyrillic_plate")),
        ("cyr_hand", manifests.build_validation(CYRILLIC_MANIFEST, "cyrillic_handwriting")),
    ]

    collector = SliceCollector(device, scaler, arguments.batch_size, arguments.limit)
    slices = [collector.collect(name, dataset, model) for name, dataset in targets]

    console.print(decomposition_table(slices))
    visible = [item for item in slices if not item.name.startswith("train_")]
    console.print(height_table(visible))
    console.print(aspect_table(visible))

    train_scene = next(item for item in slices if item.name == "train_real").metrics().score
    validation_scene = next(item for item in slices if item.name == "real_scene").metrics().score
    console.print(
        f"\n[bold]реальные кропы:[/bold] обучение {train_scene:.5f} против валидации {validation_scene:.5f}"
        f"  → разрыв {train_scene - validation_scene:+.5f}"
    )


if __name__ == "__main__":
    main()
