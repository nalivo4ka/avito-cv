from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, Dataset

from avitocv.config import DataConfig
from avitocv.data.factory import SyntheticDatasetFactory
from avitocv.data.manifest import CropManifest
from avitocv.data.sampling import SeedScheme
from avitocv.data.sources import ManifestTextLineSource, TextLineSource
from avitocv.data.storage import CropIndex, CropShardWriter, encode_crop

MANIFEST_NAME = "manifest.json"
DEFAULT_QUALITY = 88


@dataclass(frozen=True)
class EncodedCrop:
    payload: bytes
    width: int
    height: int


class UprightCropDataset(Dataset):
    def __init__(self, source: TextLineSource, seed_scheme: SeedScheme, quality: int) -> None:
        self._source = source
        self._seed_scheme = seed_scheme
        self._quality = quality

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, index: int) -> EncodedCrop:
        image = self._source.load_upright(index, self._seed_scheme.rng_for(index))
        return EncodedCrop(payload=encode_crop(image, self._quality), width=image.width, height=image.height)


class CropStoreBuilder:
    """Пишет кропы в шарды и периодически сохраняет индекс.

    Индекс сохраняется по ходу дела, потому что генерация идёт десятки минут: без этого
    единственная ошибка в самом конце оставляла бы гигабайты шардов без индекса, то есть
    полностью бесполезными.
    """

    def __init__(self, output_dir: Path, shard_size_bytes: int, checkpoint_every: int = 50_000) -> None:
        self._output_dir = output_dir
        self._shard_size_bytes = shard_size_bytes
        self._checkpoint_every = checkpoint_every

    def build(self, loader: DataLoader, total: int) -> CropIndex:
        records = []
        with CropShardWriter(self._output_dir, self._shard_size_bytes) as writer, self._progress() as progress:
            task = progress.add_task("материализация кропов", total=total)
            for batch in loader:
                for crop in batch:
                    location = writer.write(crop.payload)
                    records.append((location.shard, location.offset, location.length, crop.width, crop.height))
                progress.advance(task, len(batch))
                if len(records) % self._checkpoint_every < len(batch):
                    self._to_index(records).save(self._output_dir)
        return self._to_index(records)

    @staticmethod
    def _to_index(records: list[tuple]) -> CropIndex:
        columns = ("shard", "offset", "length", "width", "height")
        return CropIndex.from_records([dict(zip(columns, row)) for row in records])

    def _progress(self) -> Progress:
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )


def write_manifest(directory: Path, arguments: argparse.Namespace, index: CropIndex) -> None:
    payload = {
        "crop_count": len(index),
        "config": str(arguments.config),
        "seed": arguments.seed,
        "jpeg_quality": arguments.quality,
        "total_bytes": int(index.frame["length"].sum()),
        "median_height": float(index.frame["height"].median()),
        "median_aspect": float((index.frame["width"] / index.frame["height"]).median()),
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def build_source(arguments: argparse.Namespace, config) -> TextLineSource:
    """Источник кропов: манифест реальных фотографий либо генератор синтетики.

    Реальные кропы тоже раскладываются по шардам, а не читаются из манифеста на лету. Причина
    в перемешивании: в манифесте записи идут подряд по снимкам, и загрузчик с shuffle обращался
    бы к случайным снимкам, обнуляя кэш распакованных изображений. Разложенные по шардам кропы
    читаются одинаково быстро в любом порядке.
    """
    if arguments.manifest is None:
        return SyntheticDatasetFactory(config).build_source(arguments.count)
    manifest = CropManifest.load(arguments.manifest)
    if arguments.slice:
        manifest = manifest.take_slice(arguments.slice)
    return ManifestTextLineSource.from_manifest(manifest, Path("."))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Разложить ровные кропы по шардам")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--manifest", type=Path, default=None,
                        help="манифест реальных кропов; без него генерируется синтетика")
    parser.add_argument("--slice", default=None, help="взять только один срез манифеста")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/train"))
    parser.add_argument("--count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = DataConfig.from_yaml(arguments.config)
    source = build_source(arguments, config)
    count = min(arguments.count, len(source)) if arguments.manifest else arguments.count
    dataset = UprightCropDataset(source, SeedScheme(arguments.seed), arguments.quality)
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        collate_fn=list,
        persistent_workers=arguments.workers > 0,
    )
    index = CropStoreBuilder(arguments.output_dir, arguments.shard_size_mb * 1024 * 1024).build(loader, count)
    index.save(arguments.output_dir)
    write_manifest(arguments.output_dir, arguments, index)
    print(f"кропов: {len(index.frame)}, шардов: {index.frame['shard'].nunique()}")


if __name__ == "__main__":
    main()
