from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from rich.console import Console
from rich.table import Table

from avitocv.config import DataConfig
from avitocv.data.datasets import OrientationDataset
from avitocv.data.factory import ManifestDatasetFactory, MaterializedDatasetFactory, SyntheticDatasetFactory
from avitocv.data.matching.profile import CropProfile, EmpiricalDistribution
from avitocv.data.matching.profiling import CropMeasurer, CropSetProfiler

REPORTED_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
GOOD_RATIO = 0.15
ACCEPTABLE_RATIO = 0.35


@dataclass(frozen=True)
class MetricComparison:
    name: str
    reference: EmpiricalDistribution
    candidate: EmpiricalDistribution

    def median_gap(self) -> float:
        expected = self.reference.median
        if abs(expected) < 1e-9:
            return float("inf")
        return abs(self.candidate.median - expected) / abs(expected)


class ProfileComparator:
    def compare(self, reference: CropProfile, candidate: CropProfile) -> list[MetricComparison]:
        return [
            MetricComparison("height", reference.crop_height, candidate.crop_height),
            MetricComparison("aspect", reference.aspect_ratio, candidate.aspect_ratio),
            MetricComparison("sharpness", reference.sharpness, candidate.sharpness),
            MetricComparison("contrast", reference.ink_spread, candidate.ink_spread),
        ]


class ComparisonReporter:
    def __init__(self, console: Console) -> None:
        self._console = console

    def report(self, comparisons: list[MetricComparison], reference: CropProfile, candidate: CropProfile) -> None:
        for comparison in comparisons:
            self._console.print(self._build_table(comparison))
        self._console.print(
            f"grayscale share: test {reference.grayscale_share:.3f} vs generated {candidate.grayscale_share:.3f}"
        )

    def _build_table(self, comparison: MetricComparison) -> Table:
        gap = comparison.median_gap()
        table = Table(title=f"{comparison.name}  (median gap {gap:.1%} {self._verdict(gap)})")
        table.add_column("quantile")
        table.add_column("test", justify="right")
        table.add_column("generated", justify="right")
        table.add_column("ratio", justify="right")
        for probability in REPORTED_QUANTILES:
            expected = comparison.reference.quantile(probability)
            actual = comparison.candidate.quantile(probability)
            ratio = actual / expected if abs(expected) > 1e-9 else float("nan")
            table.add_row(f"q{int(probability * 100):02d}", f"{expected:.4f}", f"{actual:.4f}", f"{ratio:.2f}")
        return table

    def _verdict(self, gap: float) -> str:
        if gap <= GOOD_RATIO:
            return "[green]ok[/green]"
        if gap <= ACCEPTABLE_RATIO:
            return "[yellow]close[/yellow]"
        return "[red]off[/red]"


def iterate_training_images(dataset: OrientationDataset, limit: int) -> Iterator[np.ndarray]:
    for index in range(min(limit, len(dataset))):
        crop, _ = dataset.load_crop(index)
        yield np.asarray(crop, dtype=np.uint8)


def build_dataset(config: DataConfig, store: Path | None, manifest: Path | None) -> OrientationDataset:
    if manifest is not None:
        return ManifestDatasetFactory(config).build_validation(manifest)
    if store is None:
        return SyntheticDatasetFactory(config).build_training()
    return MaterializedDatasetFactory(config).build_training(store)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated crops against the test set profile")
    parser.add_argument("--reference", type=Path, default=Path("configs/test_profile.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    reference = CropProfile.load(arguments.reference)
    dataset = build_dataset(DataConfig.from_yaml(arguments.config), arguments.store, arguments.manifest)
    candidate = CropSetProfiler(CropMeasurer()).build(iterate_training_images(dataset, arguments.limit))
    comparisons = ProfileComparator().compare(reference, candidate)
    ComparisonReporter(Console()).report(comparisons, reference, candidate)


if __name__ == "__main__":
    main()
