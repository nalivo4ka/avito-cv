"""Цветовые схемы: фон, цвет текста и обводка.

Равномерно случайный RGB даёт неоновые сочетания, которых в тесте нет. Поэтому цвет выбирается
одним из пяти режимов со взвешенными вероятностями: документ, инверсия, приглушённый цвет,
вывеска и низкоконтрастная поверхность.
"""

from __future__ import annotations

import colorsys
from abc import ABC, abstractmethod
from dataclasses import dataclass


import numpy as np

from avitocv.data.sampling import ValueRange, happens, weighted_choice

Color = tuple[int, int, int]

MAX_CHANNEL_VALUE = 255
SRGB_LINEAR_THRESHOLD = 0.04045
SRGB_LINEAR_DIVISOR = 12.92
SRGB_GAMMA_OFFSET = 0.055
SRGB_GAMMA_SCALE = 1.055
SRGB_GAMMA_EXPONENT = 2.4
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
CONTRAST_OFFSET = 0.05
CHANNEL_JITTER = 6


def relative_luminance(color: Color) -> float:
    # Формула WCAG 2.1: обратная гамма-коррекция sRGB, затем взвешивание каналов по
    # чувствительности глаза. Нужна, чтобы контраст считался так, как его видит человек,
    # а не как разность байтов.
    channels = []
    for value in color:
        normalized = value / MAX_CHANNEL_VALUE
        if normalized <= SRGB_LINEAR_THRESHOLD:
            channels.append(normalized / SRGB_LINEAR_DIVISOR)
            continue
        channels.append(((normalized + SRGB_GAMMA_OFFSET) / SRGB_GAMMA_SCALE) ** SRGB_GAMMA_EXPONENT)
    return sum(weight * channel for weight, channel in zip(LUMINANCE_WEIGHTS, channels))


def contrast_ratio(first: Color, second: Color) -> float:
    luminances = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (luminances[0] + CONTRAST_OFFSET) / (luminances[1] + CONTRAST_OFFSET)


def _clamp_channel(value: float) -> int:
    return int(np.clip(round(value), 0, MAX_CHANNEL_VALUE))


def _jitter(color: Color, rng: np.random.Generator, amount: int = CHANNEL_JITTER) -> Color:
    offsets = rng.integers(-amount, amount + 1, size=3)
    return tuple(_clamp_channel(value + offset) for value, offset in zip(color, offsets))


def _from_hsv(hue: float, saturation: float, value: float) -> Color:
    return tuple(_clamp_channel(channel * MAX_CHANNEL_VALUE) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def _gray(level: float, rng: np.random.Generator) -> Color:
    return _jitter((_clamp_channel(level),) * 3, rng)


@dataclass(frozen=True)
class ColorScheme:
    """Пара «фон — текст» и необязательная обводка."""

    background: Color
    foreground: Color
    stroke: Color | None

    @property
    def is_light_on_dark(self) -> bool:
        return relative_luminance(self.foreground) > relative_luminance(self.background)

    @property
    def contrast(self) -> float:
        return contrast_ratio(self.foreground, self.background)


class ColorSchemeMode(ABC):
    """Один сценарий окраски кропа."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        raise NotImplementedError


@dataclass(frozen=True)
class DocumentMode(ColorSchemeMode):
    """Тёмный текст на светлом фоне: документы, скриншоты, ценники."""

    background_level: ValueRange = ValueRange(215, 255)
    foreground_level: ValueRange = ValueRange(0, 70)

    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        return _gray(self.background_level.sample(rng), rng), _gray(self.foreground_level.sample(rng), rng)


@dataclass(frozen=True)
class InverseMode(ColorSchemeMode):
    """Светлый текст на тёмном фоне."""

    background_level: ValueRange = ValueRange(0, 70)
    foreground_level: ValueRange = ValueRange(200, 255)

    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        return _gray(self.background_level.sample(rng), rng), _gray(self.foreground_level.sample(rng), rng)


@dataclass(frozen=True)
class MutedColorMode(ColorSchemeMode):
    """Приглушённый цветной фон: упаковка, бумага, стены."""

    saturation: ValueRange = ValueRange(0.04, 0.30)
    background_value: ValueRange = ValueRange(0.45, 1.0)
    foreground_level: ValueRange = ValueRange(0, 90)

    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        background = _from_hsv(rng.random(), self.saturation.sample(rng), self.background_value.sample(rng))
        foreground = _gray(self.foreground_level.sample(rng), rng)
        if relative_luminance(background) < 0.25:
            foreground = _gray(ValueRange(190, 255).sample(rng), rng)
        return background, foreground


@dataclass(frozen=True)
class VividColorMode(ColorSchemeMode):
    """Насыщенный цветной фон: вывески, баннеры, наклейки."""

    saturation: ValueRange = ValueRange(0.55, 1.0)
    value: ValueRange = ValueRange(0.45, 1.0)

    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        background = _from_hsv(rng.random(), self.saturation.sample(rng), self.value.sample(rng))
        is_dark_background = relative_luminance(background) < 0.35
        level = ValueRange(215, 255) if is_dark_background else ValueRange(0, 55)
        return background, _gray(level.sample(rng), rng)


@dataclass(frozen=True)
class SurfaceMode(ColorSchemeMode):
    """Низкий контраст: текст на сфотографированной поверхности."""

    saturation: ValueRange = ValueRange(0.0, 0.35)
    background_value: ValueRange = ValueRange(0.25, 0.85)
    luminance_gap: ValueRange = ValueRange(0.12, 0.45)

    def sample(self, rng: np.random.Generator) -> tuple[Color, Color]:
        background = _from_hsv(rng.random(), self.saturation.sample(rng), self.background_value.sample(rng))
        gap = self.luminance_gap.sample(rng) * MAX_CHANNEL_VALUE
        base = float(np.mean(background))
        # Уводим текст в ту сторону, где остался запас яркости: на светлой поверхности темнее,
        # на тёмной светлее. Иначе обрезка по 0..255 съела бы задуманный контраст.
        level = base - gap if base > MAX_CHANNEL_VALUE / 2 else base + gap
        return background, _gray(level, rng)


@dataclass(frozen=True)
class ColorSchemeSampler:
    """Взвешенная смесь режимов окраски."""

    modes: tuple[ColorSchemeMode, ...] = (
        DocumentMode(),
        InverseMode(),
        MutedColorMode(),
        VividColorMode(),
        SurfaceMode(),
    )
    weights: tuple[float, ...] = (5.0, 1.5, 2.5, 1.5, 2.0)
    stroke_probability: float = 0.10

    def sample(self, rng: np.random.Generator) -> ColorScheme:
        background, foreground = weighted_choice(rng, self.modes, self.weights).sample(rng)
        return ColorScheme(
            background=background,
            foreground=foreground,
            stroke=self._sample_stroke(rng, foreground),
        )

    def _sample_stroke(self, rng: np.random.Generator, foreground: Color) -> Color | None:
        if not happens(rng, self.stroke_probability):
            return None
        is_dark = relative_luminance(foreground) < 0.5
        return (MAX_CHANNEL_VALUE,) * 3 if is_dark else (0,) * 3
