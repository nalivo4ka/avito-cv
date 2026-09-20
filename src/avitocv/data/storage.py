"""Хранилище заранее сгенерированных кропов.

Миллионы мелких файлов на NTFS читаются медленно, поэтому кропы складываются подряд в несколько
крупных шардов, а parquet-индекс хранит смещение и длину каждого. Сгенерировать набор один раз
и читать из шардов примерно в десять раз быстрее, чем генерировать на лету.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
import pandas as pd
from PIL import Image

SHARD_NAME_TEMPLATE = "shard_{index:05d}.bin"
INDEX_NAME = "index.parquet"
DEFAULT_SHARD_SIZE_BYTES = 512 * 1024 * 1024
INDEX_COLUMNS = ("shard", "offset", "length", "width", "height")
CHROMA_SUBSAMPLING = 2


@dataclass(frozen=True)
class CropLocation:
    """Адрес кропа внутри хранилища: номер шарда, смещение и длина."""

    shard: int
    offset: int
    length: int


class CropShardWriter:
    """Пишет кропы подряд в шарды, начиная новый по достижении предельного размера."""

    def __init__(self, directory: Path, shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._shard_size_bytes = shard_size_bytes
        self._shard_index = 0
        self._handle: BinaryIO | None = None
        self._offset = 0

    def write(self, payload: bytes) -> CropLocation:
        handle = self._ensure_handle(len(payload))
        location = CropLocation(shard=self._shard_index, offset=self._offset, length=len(payload))
        handle.write(payload)
        self._offset += len(payload)
        return location

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "CropShardWriter":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def _ensure_handle(self, payload_size: int) -> BinaryIO:
        # Кроп никогда не разрезается между шардами: если он не влезает в остаток текущего
        # файла, начинается новый. Поэтому шард получается чуть меньше лимита, а не больше.
        if self._handle is not None and self._offset + payload_size <= self._shard_size_bytes:
            return self._handle
        if self._handle is not None:
            self._handle.close()
            self._shard_index += 1
            self._offset = 0
        self._handle = open(self._directory / SHARD_NAME_TEMPLATE.format(index=self._shard_index), "wb")
        return self._handle


class CropShardReader:
    """Читает кроп по адресу; переживает пикление для воркеров DataLoader."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._handles: dict[int, BinaryIO] = {}

    def read(self, location: CropLocation) -> bytes:
        handle = self._handles.get(location.shard)
        if handle is None:
            handle = open(self._directory / SHARD_NAME_TEMPLATE.format(index=location.shard), "rb")
            self._handles[location.shard] = handle
        handle.seek(location.offset)
        return handle.read(location.length)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    # Файловые дескрипторы не пиклятся, а DataLoader копирует объект в каждый воркер.
    # Передаём только путь: воркер откроет шарды сам при первом чтении.
    def __getstate__(self) -> dict:
        return {"_directory": self._directory, "_handles": {}}

    def __setstate__(self, state: dict) -> None:
        self._directory = state["_directory"]
        self._handles = {}


class CropIndex:
    """Таблица адресов и геометрии всех кропов хранилища.

    Хранит две формы одного и того же: таблицу целиком для отчётов и выравнивания геометрии и
    отдельный массив адресов для чтения. В воркеры DataLoader уезжает только массив адресов —
    таблица там не нужна ни одной строчкой кода, а копий столько же, сколько воркеров, и на
    трёх хранилищах это были лишние сотни мегабайт.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = set(INDEX_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"crop index is missing columns: {sorted(missing)}")
        self._frame = frame.reset_index(drop=True)
        self._locations = frame[["shard", "offset", "length"]].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self._locations)

    def location_at(self, index: int) -> CropLocation:
        shard, offset, length = self._locations[index]
        return CropLocation(shard=int(shard), offset=int(offset), length=int(length))

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            raise RuntimeError("таблица индекса не переносится в воркеры: читайте её в главном процессе")
        return self._frame

    def __getstate__(self) -> dict:
        return {"_frame": None, "_locations": self._locations}

    def __setstate__(self, state: dict) -> None:
        self._frame = state["_frame"]
        self._locations = state["_locations"]

    @classmethod
    def from_records(cls, records: Sequence[dict]) -> "CropIndex":
        return cls(pd.DataFrame.from_records(list(records), columns=list(INDEX_COLUMNS)))

    @classmethod
    def load(cls, directory: Path) -> "CropIndex":
        path = Path(directory) / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(f"crop index not found: {path}")
        return cls(pd.read_parquet(path))

    def save(self, directory: Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(Path(directory) / INDEX_NAME, index=False)


def encode_crop(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=CHROMA_SUBSAMPLING)
    return buffer.getvalue()


def decode_crop(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()
