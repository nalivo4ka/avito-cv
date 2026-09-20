"""Манифест реальных кропов: где в какой фотографии лежит ровная строка текста.

Манифест — это способ использовать настоящие фотографии, не таская их копии по репозиторию:
в нём только путь к картинке и координаты бокса. Строки в манифесте по контракту **ровные**,
поэтому он подставляется в `ManifestTextLineSource` наравне с генератором.

Колонка `slice_name` нужна, чтобы считать метрики по срезам отдельно (сцена, рукопись,
кириллица): усреднение в одно число скрывает, где именно модель слабая.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_SLICE = "default"
MANIFEST_COLUMNS = ("slice_name", "image_path", "left", "top", "width", "height")
WEIGHT_COLUMN = "weight"


@dataclass(frozen=True)
class CropRecord:
    """Запись манифеста: файл изображения и координаты бокса в нём."""

    image_path: str
    left: int
    top: int
    width: int
    height: int
    slice_name: str = DEFAULT_SLICE
    weight: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class CropManifest:
    """Таблица записей манифеста с чтением и записью parquet и выборкой по срезам."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = set(MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"манифесту не хватает колонок: {sorted(missing)}")
        self._frame = frame.reset_index(drop=True)
        if WEIGHT_COLUMN not in self._frame.columns:
            self._frame[WEIGHT_COLUMN] = 1.0

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def slice_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._frame["slice_name"].unique()))

    @property
    def weights(self) -> np.ndarray:
        return self._frame[WEIGHT_COLUMN].to_numpy(dtype=float)

    def records(self) -> tuple[CropRecord, ...]:
        columns = list(MANIFEST_COLUMNS) + [WEIGHT_COLUMN]
        return tuple(
            CropRecord(
                slice_name=str(row[0]),
                image_path=str(row[1]),
                left=int(row[2]),
                top=int(row[3]),
                width=int(row[4]),
                height=int(row[5]),
                weight=float(row[6]),
            )
            for row in self._frame[columns].itertuples(index=False, name=None)
        )

    def with_weights(self, weights: np.ndarray) -> "CropManifest":
        if len(weights) != len(self._frame):
            raise ValueError(f"нужно {len(self._frame)} весов, получено {len(weights)}")
        updated = self._frame.copy()
        updated[WEIGHT_COLUMN] = np.asarray(weights, dtype=float)
        return CropManifest(updated)

    def take_slice(self, slice_name: str) -> "CropManifest":
        selected = self._frame[self._frame["slice_name"] == slice_name]
        if selected.empty:
            raise ValueError(f"в манифесте нет среза {slice_name!r}; есть {self.slice_names}")
        return CropManifest(selected)

    @classmethod
    def from_records(cls, records: Sequence[CropRecord]) -> "CropManifest":
        if not records:
            raise ValueError("манифест не может быть пустым")
        frame = pd.DataFrame(
            [[record.slice_name, record.image_path, record.left, record.top,
              record.width, record.height, record.weight]
             for record in records],
            columns=list(MANIFEST_COLUMNS) + [WEIGHT_COLUMN],
        )
        return cls(frame)

    @classmethod
    def load(cls, path: Path) -> "CropManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"манифест не найден: {path}")
        return cls(pd.read_parquet(path))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(Path(path), index=False)

