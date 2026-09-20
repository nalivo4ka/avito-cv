"""Собирает манифест реальных кропов из HierText.

Оба сплита проходят одним кодом, но используются по-разному и не пересекаются:

    validation  1724 снимка -> валидация, взвешенная под геометрию теста
    train       8281 снимок -> обучение, без весов

Разделение строгое: валидационные снимки не должны попасть в обучение, иначе её оценки
перестанут что-либо значить. Веса нужны только валидации — там они приводят распределение
высот к тестовому; в обучении importance-веса лишь увеличили бы дисперсию градиента.

Скрипт занимается загрузкой и CLI; логика разбора разметки живёт в `avitocv.data.real.hiertext`,
взвешивание — в `avitocv.data.matching.geometry.JointGeometryWeighter`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import socket
import tarfile
import urllib.request
from pathlib import Path

from rich.progress import Progress

from avitocv.data.matching.geometry import CropGeometry, JointGeometryWeighter
from avitocv.data.real.hiertext import HierTextManifestBuilder, LineFilter
from avitocv.data.manifest import CropManifest
from avitocv.data.matching.profile import CropProfile

ANNOTATION_URL_TEMPLATE = "https://raw.githubusercontent.com/google-research-datasets/hiertext/main/gt/{split}.jsonl.gz"
IMAGE_ARCHIVE_URL_TEMPLATE = "https://huggingface.co/datasets/1398listener/Hiertext/resolve/main/{split}.tgz"
USER_AGENT = "Mozilla/5.0"
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120


class HierTextDownloader:
    """Качает аннотации и изображения, распаковывает архив и считает контрольные суммы."""

    def __init__(self, root: Path, split: str) -> None:
        self._root = root
        self._split = split
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def annotation_url(self) -> str:
        return ANNOTATION_URL_TEMPLATE.format(split=self._split)

    @property
    def image_url(self) -> str:
        return IMAGE_ARCHIVE_URL_TEMPLATE.format(split=self._split)

    def fetch_annotations(self) -> tuple[list[dict], str]:
        path = self._root / f"{self._split}.jsonl.gz"
        self._download(self.annotation_url, path, "аннотации")
        payload = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        return payload["annotations"], _sha256(path)

    def fetch_images(self) -> tuple[Path, str]:
        archive = self._root / f"{self._split}.tgz"
        self._download(self.image_url, archive, "изображения")
        images_dir = self._root / self._split
        if not images_dir.exists():
            with tarfile.open(archive) as handle:
                handle.extractall(self._root, filter="data")
        return images_dir, _sha256(archive)

    def _download(self, url: str, target: Path, description: str) -> None:
        if target.exists():
            return
        socket.setdefaulttimeout(REQUEST_TIMEOUT_SECONDS)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response, open(target, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            with Progress() as progress:
                task = progress.add_task(f"скачивание: {description}", total=total or None)
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
                    progress.advance(task, len(chunk))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_lockfile(
    path: Path,
    downloader: HierTextDownloader,
    digests: dict[str, str],
    manifest: CropManifest,
    effective_size: float,
) -> None:
    counts = manifest.frame["slice_name"].value_counts().to_dict()
    payload = {
        "annotations": {"url": downloader.annotation_url, "sha256": digests["annotations"]},
        "images": {"url": downloader.image_url, "sha256": digests["images"]},
        "crop_count": len(manifest),
        "effective_sample_size": round(effective_size),
        "slices": {str(name): int(count) for name, count in counts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать манифест реальных кропов из HierText")
    parser.add_argument("--split", default="validation", choices=("validation", "train"))
    parser.add_argument("--root", type=Path, default=Path("data/real/hiertext"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--lock-path", type=Path, default=None)
    parser.add_argument("--reference-profile", type=Path, default=Path("configs/test_profile.json"))
    parser.add_argument("--min-height", type=int, default=12)
    parser.add_argument("--no-weights", action="store_true", help="не считать веса; так собирается обучающий сплит")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    manifest_path = arguments.manifest or Path(f"data/real/{arguments.split}_manifest.parquet")
    lock_path = arguments.lock_path or Path(f"data/real/{arguments.split}.lock.json")

    downloader = HierTextDownloader(arguments.root, arguments.split)
    annotations, annotation_digest = downloader.fetch_annotations()
    images_dir, image_digest = downloader.fetch_images()

    builder = HierTextManifestBuilder(LineFilter(min_height=arguments.min_height))
    manifest = CropManifest.from_records(builder.build(annotations, images_dir))

    if arguments.no_weights:
        effective_size = float(len(manifest))
    else:
        # Веса приводят геометрию к тестовой сразу по двум осям. Одной высоты мало: при
        # взвешивании только по ней короткие кропы получали 30% веса против 7% в тесте, а они
        # же самые трудные, и метрика занижала себя примерно на 0.02.
        weighter = JointGeometryWeighter()
        frame = manifest.frame
        geometry = CropGeometry.of(frame["height"].to_numpy(), frame["width"] / frame["height"])
        weights = weighter.weights_for(geometry, CropProfile.load(arguments.reference_profile).geometry)
        manifest = manifest.with_weights(weights)
        effective_size = weighter.effective_sample_size(weights)

    manifest.save(manifest_path)
    write_lockfile(
        lock_path,
        downloader,
        {"annotations": annotation_digest, "images": image_digest},
        manifest,
        effective_size,
    )

    print(f"строк принято: {builder.report.accepted}")
    print(f"эффективный размер: {effective_size:.0f}")
    print(f"манифест: {manifest_path}")
    print(f"перевёрнутых в кадре, пропущено: {builder.report.upside_down}")
    print(f"отбраковано фильтром: {builder.report.rejected}")
    for name in manifest.slice_names:
        print(f"  {name}: {len(manifest.take_slice(name))}")


if __name__ == "__main__":
    main()
