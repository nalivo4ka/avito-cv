"""Подложка, на которой печатается текст.

Провайдер получает размер холста и цветовую схему и рисует фон. Процедурные провайдеры
работают всегда; `PhotoPatchBackground` подключается автоматически, если в каталоге фонов
лежат фотографии, и даёт самую реалистичную текстуру.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from avitocv.data.synthesis.palettes import MAX_CHANNEL_VALUE, Color, ColorScheme
from avitocv.data.sampling import ValueRange, weighted_choice

Size = tuple[int, int]
PHOTO_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class BackgroundProvider(ABC):
    """Рисует фон заданного размера под заданную цветовую схему."""

    @abstractmethod
    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError


def _jitter(color: Color, amount: int, rng: np.random.Generator) -> Color:
    offsets = rng.integers(-amount, amount + 1, size=3)
    return tuple(int(np.clip(value + offset, 0, MAX_CHANNEL_VALUE)) for value, offset in zip(color, offsets))


@dataclass(frozen=True)
class SolidBackground(BackgroundProvider):
    """Равномерная заливка цветом фона схемы."""

    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        return Image.new("RGB", size, scheme.background)


@dataclass(frozen=True)
class LinearGradientBackground(BackgroundProvider):
    """Линейный градиент произвольного направления."""

    shift_range: ValueRange = ValueRange(20, 90)

    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        width, height = size
        shift = int(self.shift_range.sample(rng)) * (1 if rng.random() < 0.5 else -1)
        start = np.array(scheme.background, dtype=np.float32)
        end = np.clip(start + shift, 0, MAX_CHANNEL_VALUE)
        axis = self._make_axis(width, height, rng)
        gradient = start[None, None, :] + axis[..., None] * (end - start)[None, None, :]
        return Image.fromarray(gradient.astype(np.uint8), mode="RGB")

    def _make_axis(self, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
        horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        weight = float(rng.random())
        return weight * np.broadcast_to(horizontal, (height, width)) + (1.0 - weight) * np.broadcast_to(
            vertical, (height, width)
        )


@dataclass(frozen=True)
class NoiseBackground(BackgroundProvider):
    """Заливка с зерном: бумага, ткань, шумная съёмка."""

    grain_range: ValueRange = ValueRange(4, 28)

    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        width, height = size
        grain = self.grain_range.sample(rng)
        base = np.array(scheme.background, dtype=np.float32)[None, None, :]
        noise = rng.normal(0.0, grain, size=(height, width, 3)).astype(np.float32)
        return Image.fromarray(np.clip(base + noise, 0, MAX_CHANNEL_VALUE).astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class PhotoPatchBackground(BackgroundProvider):
    """Патч из реальной фотографии, слегка подкрашенный под цветовую схему.

    Подкраска нужна, чтобы текст остался читаемым: цвет текста подбирается под фон схемы, и без
    сдвига фотографии в его сторону контраст мог бы пропасть. Но сильная подкраска съедает
    саму фактуру, ради которой патч и берётся, поэтому она держится слабой.
    """

    paths: tuple[Path, ...]
    tint_strength: ValueRange = ValueRange(0.12, 0.55)

    @classmethod
    def from_directory(cls, directory: Path) -> "PhotoPatchBackground | None":
        paths = tuple(
            path for path in sorted(Path(directory).rglob("*")) if path.suffix.lower() in PHOTO_SUFFIXES
        )
        return cls(paths=paths) if paths else None

    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        patch = self._take_patch(size, rng)
        tint = np.array(scheme.background, dtype=np.float32)[None, None, :]
        strength = self.tint_strength.sample(rng)
        blended = np.asarray(patch, dtype=np.float32) * (1.0 - strength) + tint * strength
        return Image.fromarray(blended.astype(np.uint8), mode="RGB")

    def _take_patch(self, size: Size, rng: np.random.Generator) -> Image.Image:
        path = self.paths[int(rng.integers(len(self.paths)))]
        with Image.open(path) as image:
            source = image.convert("RGB")
            scale = max(size[0] / source.width, size[1] / source.height, 1.0)
            if scale > 1.0:
                source = source.resize((int(source.width * scale) + 1, int(source.height * scale) + 1))
            left = int(rng.integers(0, max(1, source.width - size[0] + 1)))
            top = int(rng.integers(0, max(1, source.height - size[1] + 1)))
            return source.crop((left, top, left + size[0], top + size[1]))


class WeightedBackgroundMixture(BackgroundProvider):
    """Взвешенная смесь провайдеров фона."""

    def __init__(self, providers: Sequence[BackgroundProvider], weights: Sequence[float]) -> None:
        if not providers:
            raise ValueError("at least one background provider is required")
        self._providers = tuple(providers)
        self._weights = tuple(float(weight) for weight in weights)

    def render(self, size: Size, scheme: ColorScheme, rng: np.random.Generator) -> Image.Image:
        return weighted_choice(rng, self._providers, self._weights).render(size, scheme, rng)

    @classmethod
    def build_default(cls, photo_directory: Path | None = None) -> "WeightedBackgroundMixture":
        providers: list[BackgroundProvider] = [SolidBackground(), LinearGradientBackground(), NoiseBackground()]
        weights = [3.0, 2.0, 2.0]
        photo_provider = PhotoPatchBackground.from_directory(photo_directory) if photo_directory else None
        if photo_provider is not None:
            providers.append(photo_provider)
            weights.append(3.0)
        return cls(providers, weights)
