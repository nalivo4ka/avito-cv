"""Собирает манифест реальных кропов из TextOCR.

Скрипт занимается скачиванием, распаковкой и CLI; разбор разметки живёт в
`avitocv.data.real.textocr`. Оба сплита TextOCR (`train` и `val`) идут в обучение одним манифестом:
валидация у нас своя, на HierText, и трогать её нечем.

Единственное, что нельзя пропустить, — исключение снимков, попавших в валидацию HierText.
Оба датасета выросли из Open Images, и 26 снимков пересекаются.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

from avitocv.data.manifest import CropManifest
from avitocv.data.real.textocr import LineGrouper, RunEmitter, TextOcrManifestBuilder, WordFilter

ANNOTATION_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_{split}.json"
IMAGE_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
READ_CHUNK_BYTES = 4 * 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def extract_images(archive: Path, root: Path) -> Path:
    """Распаковывает архив один раз; повторный запуск ничего не переписывает.

    Оба сплита TextOCR лежат в архиве одной папкой `train_images` — разделение на train и val
    задаётся только разметкой.
    """
    images_dir = root / "train_images"
    if not images_dir.exists():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(root)
    return images_dir


def load_excluded_ids(path: Path) -> set[str]:
    """Идентификаторы снимков валидации HierText — их нельзя брать в обучение."""
    if not path.exists():
        return set()
    payload = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
    return {item["image_id"] for item in payload["annotations"]}


def write_lockfile(path: Path, manifest: CropManifest, digests: dict[str, str], report) -> None:
    payload = {
        "annotations": {
            split: {"url": ANNOTATION_URL_TEMPLATE.format(split=split), "sha256": digest}
            for split, digest in digests.items() if split != "images"
        },
        "images": {"url": IMAGE_ARCHIVE_URL, "sha256": digests["images"]},
        "crop_count": len(manifest),
        "excluded_images": report.excluded_images,
        "upside_down": report.upside_down,
        "rejected": report.rejected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать манифест реальных кропов из TextOCR")
    parser.add_argument("--root", type=Path, default=Path("data/real/textocr"))
    parser.add_argument("--manifest", type=Path, default=Path("data/real/textocr_manifest.parquet"))
    parser.add_argument("--lock-path", type=Path, default=Path("data/real/textocr.lock.json"))
    parser.add_argument("--exclude-from", type=Path, default=Path("data/real/hiertext/validation.jsonl.gz"),
                        help="разметка, снимки которой нельзя брать в обучение")
    parser.add_argument("--min-height", type=int, default=12)
    parser.add_argument("--min-aspect", type=float, default=1.0)
    parser.add_argument("--words-only", action="store_true",
                        help="не склеивать слова в строки; геометрия тогда не совпадёт с тестом")
    parser.add_argument("--max-run-length", type=int, default=3,
                        help="сколько подряд идущих слов даёт отдельный кроп помимо строки целиком")
    parser.add_argument("--skip-hashes", action="store_true", help="не считать sha256 семигигабайтного архива")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    images_root = extract_images(arguments.root / "train_val_images.zip", arguments.root)
    excluded = load_excluded_ids(arguments.exclude_from)
    print(f"снимков в исключении: {len(excluded)}")

    builder = TextOcrManifestBuilder(
        WordFilter(min_height=arguments.min_height, min_aspect_ratio=arguments.min_aspect),
        excluded,
        grouper=None if arguments.words_only else LineGrouper(),
        emitter=RunEmitter(max_run_length=arguments.max_run_length),
    )
    records = []
    digests = {}
    for split in ("train", "val"):
        annotation_path = arguments.root / f"TextOCR_0.1_{split}.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        records.extend(builder.build(payload, images_root))
        digests[split] = "" if arguments.skip_hashes else sha256_of(annotation_path)
        print(f"{split}: накоплено {len(records)} кропов")

    digests["images"] = "" if arguments.skip_hashes else sha256_of(arguments.root / "train_val_images.zip")
    manifest = CropManifest.from_records(records)
    manifest.save(arguments.manifest)
    write_lockfile(arguments.lock_path, manifest, digests, builder.report)

    print(f"принято: {builder.report.accepted}")
    print(f"перевёрнутых в кадре, пропущено: {builder.report.upside_down}")
    print(f"отбраковано фильтром: {builder.report.rejected}")
    print(f"снимков исключено пересечением: {builder.report.excluded_images}")
    for name in manifest.slice_names:
        print(f"  {name}: {len(manifest.take_slice(name))}")
    print(f"манифест: {arguments.manifest}")


if __name__ == "__main__":
    main()
