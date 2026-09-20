"""Извлечение готовых строковых кропов из parquet-датасетов HuggingFace.

Часть OCR-датасетов отдаёт уже вырезанные строки или слова вместе с транскрипцией. Такая
разметка сама гарантирует ориентацию: человек прочитал и записал текст, значит в этом
положении он читается. Распознаватель для получения метки не нужен.

Кропы раскладываются в файлы, а манифест ссылается на них боксом во всю картинку — тогда
они попадают в те же `ManifestTextLineSource` и метрики по срезам, что и боксы из HierText.

Оговорка про геометрию: авторы таких датасетов режут строки плотно, а боксы детектора всегда
неточные и часто задевают соседние строки. Поэтому по форме кропа этот слой отличается
от тестового, и полагаться на него стоит как на проверку письменности, а не геометрии.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from PIL import Image

from avitocv.data.manifest import CropRecord

IMAGE_COLUMN = "image"
TEXT_COLUMN = "text"
MIN_CROP_HEIGHT = 10
MIN_CROP_ASPECT = 1.1
CROP_FILE_TEMPLATE = "{index:05d}.png"


@dataclass(frozen=True)
class CropDatasetSpec:
    """Источник готовых кропов: откуда читать, как назвать срез и сколько брать."""

    slice_name: str
    url: str
    limit: int = 0

    @property
    def is_limited(self) -> bool:
        return self.limit > 0


@dataclass(frozen=True)
class ExtractedCrop:
    """Декодированный кроп вместе с транскрипцией."""

    image: Image.Image
    text: str

    @property
    def is_usable(self) -> bool:
        if self.image.height < MIN_CROP_HEIGHT:
            return False
        return self.image.width / self.image.height >= MIN_CROP_ASPECT


class ParquetCropReader:
    """Читает кропы из локального parquet-файла, декодируя встроенные картинки."""

    def read(self, path: Path, limit: int = 0) -> Iterator[ExtractedCrop]:
        parquet_file = pq.ParquetFile(path)
        produced = 0
        for group_index in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_index)
            texts = table.column(TEXT_COLUMN) if TEXT_COLUMN in table.column_names else None
            for position in range(table.num_rows):
                if limit and produced >= limit:
                    return
                crop = self._decode(table.column(IMAGE_COLUMN)[position].as_py())
                text = str(texts[position].as_py()) if texts is not None else ""
                produced += 1
                yield ExtractedCrop(image=crop, text=text)

    def _decode(self, cell: dict | bytes) -> Image.Image:
        payload = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB").copy()


class CropFileWriter:
    """Раскладывает кропы в файлы и отдаёт записи манифеста с боксом во всю картинку."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def write(self, crops: Iterator[ExtractedCrop], slice_name: str) -> list[CropRecord]:
        directory = self._root / slice_name
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for crop in crops:
            if not crop.is_usable:
                continue
            path = directory / CROP_FILE_TEMPLATE.format(index=len(records))
            crop.image.save(path)
            records.append(
                CropRecord(
                    image_path=path.as_posix(),
                    left=0,
                    top=0,
                    width=crop.image.width,
                    height=crop.image.height,
                    slice_name=slice_name,
                )
            )
        return records
