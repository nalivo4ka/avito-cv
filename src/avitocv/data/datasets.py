"""Разметка и препроцессинг: превращает ровный кроп в обучающий пример.

Здесь вся суть задачи: взять ровный кроп, бросить монетку, повернуть на 180°, если выпала
решка, и вернуть результат вместе с выпавшим числом в качестве метки. Разметка получается
точной по построению и бесплатной.

После поворота применяется кодековая стадия деградаций, затем `ImagePreprocessor` приводит кроп
к тензору фиксированного размера: градации серого, масштаб по высоте с сохранением пропорций
и окно по ширине.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from avitocv.data.degradation import DegradationPipeline
from avitocv.data.sampling import SeedScheme
from avitocv.data.sources import TextLineSource

NORMALIZATION_MEAN = 0.5
NORMALIZATION_STD = 0.5
MAX_CHANNEL_VALUE = 255.0
GRAYSCALE_CHANNELS = 1
COLOR_CHANNELS = 3


class OrientationSample(NamedTuple):
    """Один обучающий пример. Вес нужен там, где выборка смещена относительно теста.

    Именованный кортеж, а не dataclass, потому что default_collate у PyTorch собирает его
    в батч сам, сохраняя имена полей.
    """

    image: torch.Tensor
    label: torch.Tensor
    weight: torch.Tensor


class Orientation(IntEnum):
    """Метка: 0 — кроп ровный, 1 — повёрнут на 180°."""

    UPRIGHT = 0
    ROTATED = 1


class HorizontalWindowFit(ABC):
    """Стратегия выбора окна по ширине при приведении к фиксированному размеру."""

    @abstractmethod
    def offset(self, available: int, rng: np.random.Generator) -> int:
        raise NotImplementedError


class RandomWindowFit(HorizontalWindowFit):
    """Случайное окно: аугментация на обучении."""

    def offset(self, available: int, rng: np.random.Generator) -> int:
        return int(rng.integers(0, available + 1))


class CenterWindowFit(HorizontalWindowFit):
    """Центральное окно: детерминированная валидация и инференс."""

    def offset(self, available: int, rng: np.random.Generator) -> int:
        return available // 2


@dataclass(frozen=True)
class PreprocessConfig:
    """Размер и число каналов тензора, который получает модель."""

    height: int = 32
    width: int = 192
    channel_count: int = GRAYSCALE_CHANNELS

    def __post_init__(self) -> None:
        if self.channel_count not in (GRAYSCALE_CHANNELS, COLOR_CHANNELS):
            raise ValueError(f"channel_count must be 1 or 3, got {self.channel_count}")


class ImagePreprocessor:
    """Приводит кроп произвольного размера к фиксированному нормированному тензору."""

    def __init__(self, config: PreprocessConfig, window_fit: HorizontalWindowFit) -> None:
        self._config = config
        self._window_fit = window_fit

    def to_tensor(self, image: Image.Image, rng: np.random.Generator) -> torch.Tensor:
        pixels = self._to_channels(np.asarray(image.convert("RGB"), dtype=np.uint8))
        scaled = self._scale_to_height(pixels)
        fitted = self._fit_width(scaled, rng)
        return self._to_normalized_tensor(fitted)

    def _to_channels(self, pixels: np.ndarray) -> np.ndarray:
        if self._config.channel_count == COLOR_CHANNELS:
            return pixels
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)[..., None]

    def _scale_to_height(self, pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape[:2]
        target_width = max(1, int(round(width * self._config.height / height)))
        resized = cv2.resize(pixels, (target_width, self._config.height), interpolation=cv2.INTER_AREA)
        return resized if resized.ndim == 3 else resized[..., None]

    def _fit_width(self, pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        width = pixels.shape[1]
        if width == self._config.width:
            return pixels
        if width > self._config.width:
            offset = self._window_fit.offset(width - self._config.width, rng)
            return pixels[:, offset:offset + self._config.width]
        padding = self._config.width - width
        left = self._window_fit.offset(padding, rng)
        # BORDER_REPLICATE растягивает краевые пиксели вместо заливки нулями: чёрная рамка
        # была бы искусственным признаком, которого в тестовых кропах нет.
        # reshape нужен потому, что OpenCV роняет ось каналов у одноканальных картинок.
        return cv2.copyMakeBorder(pixels, 0, 0, left, padding - left, cv2.BORDER_REPLICATE).reshape(
            self._config.height, self._config.width, self._config.channel_count
        )

    def _to_normalized_tensor(self, pixels: np.ndarray) -> torch.Tensor:
        array = pixels.astype(np.float32) / MAX_CHANNEL_VALUE
        normalized = (array - NORMALIZATION_MEAN) / NORMALIZATION_STD
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


class OrientationDataset(Dataset):
    """Ровный кроп -> поворот по метке -> кодековые деградации -> тензор и метка."""

    def __init__(
        self,
        source: TextLineSource,
        codec_stage: DegradationPipeline,
        preprocessor: ImagePreprocessor,
        seed_scheme: SeedScheme,
    ) -> None:
        self._source = source
        self._codec_stage = codec_stage
        self._preprocessor = preprocessor
        self._seed_scheme = seed_scheme
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._source)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def load_crop(self, index: int) -> tuple[Image.Image, Orientation]:
        image, orientation, _ = self._produce(index)
        return image, orientation

    def __getitem__(self, index: int) -> OrientationSample:
        image, orientation, rng = self._produce(index)
        return OrientationSample(
            image=self._preprocessor.to_tensor(image, rng),
            label=torch.tensor(float(orientation), dtype=torch.float32),
            weight=torch.tensor(self._source.weight_of(index), dtype=torch.float32),
        )

    def _produce(self, index: int) -> tuple[Image.Image, Orientation, np.random.Generator]:
        # Порядок принципиален: источник отдаёт заведомо ровный кроп, затем бросается монетка
        # и кроп поворачивается, и только после этого применяются кодековые деградации.
        # Любая несимметричная обработка до поворота стала бы утечкой метки.
        rng = self._seed_scheme.rng_for(index, self._epoch)
        upright = self._source.load_upright(index, rng)
        orientation = Orientation(int(rng.integers(len(Orientation))))
        oriented = self._orient(upright, orientation)
        degraded = self._codec_stage.apply(np.asarray(oriented, dtype=np.uint8), rng)
        return Image.fromarray(degraded, mode="RGB"), orientation, rng

    def _orient(self, image: Image.Image, orientation: Orientation) -> Image.Image:
        if orientation is Orientation.UPRIGHT:
            return image
        return image.transpose(Image.ROTATE_180)


@dataclass(frozen=True)
class DatasetShare:
    """Датасет и сколько примеров брать из него за эпоху."""

    dataset: "OrientationDataset"
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError(f"нужно положительное число примеров, получено {self.sample_count}")


class CombinedOrientationDataset(Dataset):
    """Несколько датасетов как один, у каждого своя стадия деградаций.

    Смешивание живёт здесь, а не на уровне источника кропов, именно из-за деградаций:
    отрисованный текст надо испортить, чтобы он стал похож на съёмку, а настоящий кроп уже
    несёт своё размытие и артефакты сжатия, и та же обработка делает его мутнее тестового.
    Общая стадия на оба вида данных вредила бы одному из них.

    Соответствие «индекс — пример» постоянно, поэтому единственным источником случайности
    порядка остаётся перемешивание в загрузчике.
    """

    def __init__(self, shares: Sequence[DatasetShare]) -> None:
        if not shares:
            raise ValueError("нужен хотя бы один датасет")
        self._shares = tuple(shares)
        self._boundaries = np.cumsum([share.sample_count for share in self._shares])
        self._epoch = 0

    def __len__(self) -> int:
        return int(self._boundaries[-1])

    def describe(self) -> str:
        return " + ".join(f"{share.sample_count} из {len(share.dataset)}" for share in self._shares)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        for share in self._shares:
            share.dataset.set_epoch(epoch)

    def load_crop(self, index: int) -> tuple[Image.Image, Orientation]:
        share, local = self._locate(index)
        return share.dataset.load_crop(local)

    def __getitem__(self, index: int) -> OrientationSample:
        share, local = self._locate(index)
        return share.dataset[local]

    def _locate(self, index: int) -> tuple[DatasetShare, int]:
        position = index % len(self)
        share_index = int(np.searchsorted(self._boundaries, position, side="right"))
        offset = 0 if share_index == 0 else int(self._boundaries[share_index - 1])
        share = self._shares[share_index]
        # Каждая эпоха сдвигает окно по хранилищу. Без сдвига датасет, из которого просят меньше
        # примеров, чем в нём есть, показывал бы всегда одно и то же начало: при 300 тысячах
        # запрошенных из 600 тысяч имеющихся вторая половина не была бы видна никогда.
        window_start = self._epoch * share.sample_count
        return share, (window_start + position - offset) % len(share.dataset)


class CropInferenceDataset(Dataset):
    """Кропы без меток для предсказания: отдаёт идентификатор и готовый тензор.

    Препроцессинг тот же, что на валидации, и детерминированный: `CenterWindowFit` не
    использует случайность, поэтому повторный запуск даёт побитово тот же результат —
    это требование к воспроизводимости отправленного решения.
    """

    def __init__(self, paths: Sequence[Path], preprocessor: ImagePreprocessor) -> None:
        if not paths:
            raise ValueError("не найдено ни одной картинки для предсказания")
        self._paths = tuple(paths)
        self._preprocessor = preprocessor
        self._rng = np.random.default_rng(0)

    @classmethod
    def from_directory(cls, directory: Path, preprocessor: ImagePreprocessor) -> "CropInferenceDataset":
        return cls(sorted(Path(directory).glob("*.png")), preprocessor)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(path.stem for path in self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self._paths[index]) as image:
            return self._preprocessor.to_tensor(image.convert("RGB"), self._rng)
