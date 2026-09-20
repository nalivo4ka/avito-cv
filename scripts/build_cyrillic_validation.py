"""Собирает валидационный манифест реальной кириллицы.

Слой 1 (HierText) почти весь латинский: он меряет разрыв «синтетика → настоящая съёмка».
Этот слой отвечает на другой вопрос — держится ли модель на реальной кириллице, где заглавные
буквы куда симметричнее латинских и подсказок для определения ориентации меньше.

Взяты два среза, оба из настоящих изображений:

    cyrillic_plate       сфотографированные российские автономера — печатные заглавные и цифры,
                         самый симметричный и потому самый трудный случай
    cyrillic_handwriting отсканированные рукописные слова

Срез `printed` того же датасета отброшен сознательно: это синтетика (видно по смешению разных
гарнитур внутри одной строки и по нарисованному фону бумаги), и валидироваться на чужом
генераторе бессмысленно.

В отличие от слоя 1, метрики здесь считаются **без** выравнивания по высоте. Диапазон высот
покрывает 87% теста, но плотность распределена иначе, и importance-веса сжимают эффективный
размер выборки с 3038 до 60–750 при любом числе бинов, с максимальным весом до 39: метрику
определяли бы единицы кропов. Слой отвечает на вопрос про письменность, а не про геометрию,
поэтому честнее считать его без весов и помнить про смещение (медиана высоты 48–68 против 42
в тесте). Флаг `--weight-by-height` оставлен, чтобы это можно было перепроверить.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import urllib.request
from pathlib import Path

from rich.progress import Progress

from avitocv.data.real.crop_datasets import CropDatasetSpec, CropFileWriter, ParquetCropReader
from avitocv.data.matching.geometry import CropGeometry, JointGeometryWeighter
from avitocv.data.manifest import CropManifest
from avitocv.data.matching.profile import CropProfile

FOXIMAZ_BASE = "https://huggingface.co/api/datasets/Foximaz/russian_ocr_small/parquet"
SPECS = (
    CropDatasetSpec(slice_name="cyrillic_plate", url=f"{FOXIMAZ_BASE}/car_plate/test/0.parquet"),
    CropDatasetSpec(slice_name="cyrillic_handwriting", url=f"{FOXIMAZ_BASE}/handwriting/test/0.parquet"),
)
USER_AGENT = "Mozilla/5.0"
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120


class ParquetDownloader:
    """Качает parquet целиком, чтобы можно было зафиксировать его контрольную сумму."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def fetch(self, spec: CropDatasetSpec) -> tuple[Path, str]:
        target = self._root / f"{spec.slice_name}.parquet"
        if not target.exists():
            socket.setdefaulttimeout(REQUEST_TIMEOUT_SECONDS)
            request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request) as response, open(target, "wb") as handle:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
        return target, _sha256(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_lockfile(path: Path, digests: dict[str, dict], manifest: CropManifest, effective_size: float) -> None:
    counts = manifest.frame["slice_name"].value_counts().to_dict()
    payload = {
        "sources": digests,
        "crop_count": len(manifest),
        "effective_sample_size": round(effective_size),
        "slices": {str(name): int(count) for name, count in counts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать валидационный манифест реальной кириллицы")
    parser.add_argument("--root", type=Path, default=Path("data/real/cyrillic"))
    parser.add_argument("--manifest", type=Path, default=Path("data/real/cyrillic_manifest.parquet"))
    parser.add_argument("--lock-path", type=Path, default=Path("data/real/cyrillic.lock.json"))
    parser.add_argument("--reference-profile", type=Path, default=Path("configs/test_profile.json"))
    parser.add_argument("--weight-by-height", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    downloader = ParquetDownloader(arguments.root / "parquet")
    reader = ParquetCropReader()
    writer = CropFileWriter(arguments.root)

    records, digests = [], {}
    with Progress() as progress:
        task = progress.add_task("извлечение кропов", total=len(SPECS))
        for spec in SPECS:
            path, digest = downloader.fetch(spec)
            digests[spec.slice_name] = {"url": spec.url, "sha256": digest}
            extracted = writer.write(reader.read(path, spec.limit), spec.slice_name)
            records.extend(extracted)
            progress.advance(task)

    manifest = CropManifest.from_records(records)
    weighter = JointGeometryWeighter()
    frame = manifest.frame
    geometry = CropGeometry.of(frame["height"].to_numpy(), frame["width"] / frame["height"])
    weights = weighter.weights_for(geometry, CropProfile.load(arguments.reference_profile).geometry)
    weighted_size = weighter.effective_sample_size(weights)
    if arguments.weight_by_height:
        manifest = manifest.with_weights(weights)
        effective_size = weighted_size
    else:
        effective_size = float(len(manifest))

    manifest.save(arguments.manifest)
    write_lockfile(arguments.lock_path, digests, manifest, effective_size)

    print(f"кропов всего: {len(manifest)}")
    print(f"эффективный размер: {effective_size:.0f}")
    print(f"выравнивание по высоте оставило бы: {weighted_size:.0f} — поэтому оно выключено")
    for name in manifest.slice_names:
        print(f"  {name}: {len(manifest.take_slice(name))}")


if __name__ == "__main__":
    main()
