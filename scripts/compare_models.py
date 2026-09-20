"""Сравнивает несколько обученных моделей на полных срезах валидации.

Обычная оценка во время обучения идёт по нескольким тысячам кропов ради скорости, и при
взвешивании под геометрию теста её эффективный размер падает до пары тысяч. Погрешность при
этом около ±0.005, то есть крупнее, чем разница между последними вариантами модели. Отличить
их можно только на полном срезе, где эффективный размер на порядок больше.

Сравнивать на глаз по разным прогонам обучения нельзя ещё и потому, что температура у каждого
своя: она подбирается на том же срезе, по которому потом сравнивают. Здесь калибровка честно
пересчитывается на половине среза, а метрика считается на другой половине.
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
from avitocv.data.factory import ManifestDatasetFactory, MaterializedDatasetFactory
from avitocv.model.architecture import ModelCostMeter, ModelFactory, TinyNetConfig
from avitocv.model.inference import ProbabilityEnsemble, SymmetricPredictor, TemperatureCalibrator
from avitocv.training.metrics import MetricsAccumulator

console = Console()

REAL_MANIFEST = Path("data/real/validation_manifest.parquet")
CYRILLIC_MANIFEST = Path("data/real/cyrillic_manifest.parquet")
DEFAULT_SYNTHETIC_STORE = Path("data/generated/val")
SLICES = {
    "synthetic": ("store", DEFAULT_SYNTHETIC_STORE, None),
    "real_scene": ("manifest", REAL_MANIFEST, "hiertext_scene"),
    "real_hand": ("manifest", REAL_MANIFEST, "hiertext_handwritten"),
    "cyr_plate": ("manifest", CYRILLIC_MANIFEST, "cyrillic_plate"),
    "cyr_hand": ("manifest", CYRILLIC_MANIFEST, "cyrillic_handwriting"),
}


@dataclass(frozen=True)
class ModelUnderTest:
    """Чекпоинт вместе с геометрией входа, с которой он обучался."""

    name: str
    checkpoint: Path
    input_height: int
    height_downsample: int

    @classmethod
    def parse(cls, specification: str) -> "ModelUnderTest":
        """Разбирает запись вида `имя:путь:высота:прореживание`."""
        parts = specification.split(":")
        if len(parts) != 4:
            raise ValueError(f"ожидается имя:путь:высота:прореживание, получено {specification!r}")
        return cls(parts[0], Path(parts[1]), int(parts[2]), int(parts[3]))


class HeldOutScorer:
    """Калибрует температуру на одной половине среза и меряет метрику на другой.

    Иначе сравнение было бы нечестным: калибровка на тех же кропах, по которым считается
    метрика, даёт каждой модели маленькую фору, и разная у разных моделей.
    """

    def __init__(self, device: torch.device, batch_size: int = 512) -> None:
        self._device = device
        self._batch_size = batch_size
        self.last_temperature = 1.0

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Калиброванные вероятности на отложенной половине вместе с её метками и весами."""
        predictor = SymmetricPredictor(model)
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=0)
        logits, labels, weights = [], [], []
        for batch in loader:
            logits.append(predictor.logits(batch.image.to(self._device)).float().cpu().numpy())
            labels.append(batch.label.numpy())
            weights.append(batch.weight.numpy())
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        weights = np.concatenate(weights)

        middle = len(logits) // 2
        scaler = TemperatureCalibrator().fit(logits[:middle], labels[:middle], weights[:middle])
        self.last_temperature = scaler.temperature
        return scaler.probabilities(logits[middle:]), labels[middle:], weights[middle:]

    @staticmethod
    def score_of(probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        accumulator = MetricsAccumulator()
        accumulator.add(probabilities, labels, weights)
        return accumulator.result().score


def estimate_test_score(scores: dict[str, float]) -> float:
    """Переводит балл на `real_scene` в оценку тестового.

    Пока валидация взвешивалась по одной высоте, разрыв с тестом был 0.023 и выглядел как
    разница доменов. На деле это был перекос метрики: коротким кропам доставалось 30% веса
    против 7% в тесте. После совместного выравнивания по высоте и пропорциям та же модель даёт
    0.9430 на валидации при 0.94726 на тесте, то есть остаток разрыва 0.004.

    Поправка держится на единственной известной точке и не претендует на точность; её смысл в
    том, чтобы не путать порядок величин при выборе модели.
    """
    anchor_validation, anchor_test = 0.9430, 0.94726
    return scores["real_scene"] + (anchor_test - anchor_validation)


def load_model(entry: ModelUnderTest, device: torch.device) -> torch.nn.Module:
    payload = torch.load(entry.checkpoint, map_location=device, weights_only=True)
    config = TinyNetConfig(input_height=entry.input_height, height_downsample=entry.height_downsample)
    model = ModelFactory(config).create(payload["architecture"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнить модели на полных срезах валидации")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model", action="append", required=True,
                        help="имя:путь:высота:прореживание, можно повторять")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--synthetic-store", type=Path, default=DEFAULT_SYNTHETIC_STORE,
                        help="хранилище синтетического среза, если оно не на месте по умолчанию")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    base_config = DataConfig.from_yaml(arguments.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SLICES["synthetic"] = ("store", arguments.synthetic_store, None)
    entries = [ModelUnderTest.parse(item) for item in arguments.model]
    scorer = HeldOutScorer(device, arguments.batch_size)
    predictions: dict[str, list[np.ndarray]] = {}
    truth: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    costs: list[float] = []

    table = Table(title="Полные срезы валидации, калибровка на отложенной половине",
                  header_style="bold magenta")
    table.add_column("модель", style="bold")
    table.add_column("MAC", justify="right")
    for name in SLICES:
        table.add_column(name, justify="right")
    table.add_column("T", justify="right", style="dim")
    table.add_column("оценка теста", justify="right", style="bold cyan")

    for entry in entries:
        model = load_model(entry, device)
        config = replace(base_config, preprocess=replace(base_config.preprocess, height=entry.input_height))
        cost = ModelCostMeter().measure(
            model.cpu(), (1, entry.input_height, config.preprocess.width))
        model.to(device)
        costs.append(cost.multiply_accumulates)
        materialized = MaterializedDatasetFactory(config)
        manifests = ManifestDatasetFactory(config)

        cells, temperatures, scores = [], [], {}
        for name, (kind, path, slice_name) in SLICES.items():
            dataset = (materialized.build_validation(path) if kind == "store"
                       else manifests.build_validation(path, slice_name))
            probabilities, labels, weights = scorer.predict(model, dataset)
            truth.setdefault(name, (labels, weights))
            predictions.setdefault(name, []).append(probabilities)
            score = scorer.score_of(probabilities, labels, weights)
            cells.append(f"{score:.4f}")
            temperatures.append(scorer.last_temperature)
            scores[name] = score
        table.add_row(entry.name, f"{cost.multiply_accumulates / 1e6:.1f}M", *cells,
                      f"{np.mean(temperatures):.2f}", f"{estimate_test_score(scores):.4f}")
        console.print(f"[dim]{entry.name}: посчитано[/dim]")

    if len(entries) > 1:
        ensemble = ProbabilityEnsemble()
        cells, scores = [], {}
        total_cost = sum(costs)
        for name in SLICES:
            labels, weights = truth[name]
            score = HeldOutScorer.score_of(ensemble.combine(predictions[name]), labels, weights)
            cells.append(f"{score:.4f}")
            scores[name] = score
        table.add_row(f"ансамбль ({len(entries)})", f"{total_cost / 1e6:.1f}M", *cells, "—",
                      f"{estimate_test_score(scores):.4f}", style="bold green")

    console.print(table)


if __name__ == "__main__":
    main()
