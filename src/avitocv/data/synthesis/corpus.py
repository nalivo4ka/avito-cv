"""Источники текста: что именно будет написано на кропе.

Смысл строки для задачи не важен — важна статистика букв: выносные элементы, заглавные,
пунктуация. Поэтому корпуса Wikipedia дополняются процедурными паттернами: цены, размеры,
артикулы и телефоны в энциклопедическом тексте не встречаются, а в тестовой выборке их много.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from avitocv.data.sampling import ValueRange, weighted_choice

RUBLE_SIGN = "\u20bd"
LATIN_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
MIN_USABLE_LINE_LENGTH = 2


class TextSource(ABC):
    """Поставщик строк текста; единственная операция — выдать случайную строку."""

    @abstractmethod
    def sample_line(self, rng: np.random.Generator) -> str:
        raise NotImplementedError


class LineFileSource(TextSource):
    """Корпус из текстового файла, по одной строке на запись."""

    def __init__(self, path: Path, min_length: int = MIN_USABLE_LINE_LENGTH) -> None:
        self._lines = self._read_lines(Path(path), min_length)

    @staticmethod
    def _read_lines(path: Path, min_length: int) -> tuple[str, ...]:
        if not path.exists():
            raise FileNotFoundError(f"corpus file not found: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        usable = tuple(line for line in lines if len(line) >= min_length)
        if not usable:
            raise ValueError(f"corpus file has no usable lines: {path}")
        return usable

    def __len__(self) -> int:
        return len(self._lines)

    def sample_line(self, rng: np.random.Generator) -> str:
        return self._lines[int(rng.integers(len(self._lines)))]


class TextPattern(ABC):
    """Генератор строки по шаблону."""

    @abstractmethod
    def generate(self, rng: np.random.Generator) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class DimensionPattern(TextPattern):
    """Габариты вида «131x35»."""

    value_range: ValueRange = ValueRange(3, 400)
    separators: tuple[str, ...] = ("x", "*", " x ")

    def generate(self, rng: np.random.Generator) -> str:
        separator = self.separators[int(rng.integers(len(self.separators)))]
        parts = [str(self.value_range.sample_int(rng)) for _ in range(int(rng.integers(2, 4)))]
        return separator.join(parts)


@dataclass(frozen=True)
class PricePattern(TextPattern):
    """Цена с разделителями разрядов и опциональным знаком рубля."""

    value_range: ValueRange = ValueRange(50, 900000)

    def generate(self, rng: np.random.Generator) -> str:
        amount = self.value_range.sample_int(rng)
        grouped = f"{amount:,}".replace(",", " ")
        return f"{grouped} {RUBLE_SIGN}" if rng.random() < 0.5 else grouped


@dataclass(frozen=True)
class ArticleCodePattern(TextPattern):
    """Артикул или маркировка из групп букв и цифр."""

    group_count: ValueRange = ValueRange(2, 5)
    group_length: ValueRange = ValueRange(2, 5)

    def generate(self, rng: np.random.Generator) -> str:
        groups = [self._make_group(rng) for _ in range(self.group_count.sample_int(rng))]
        return " ".join(groups)

    def _make_group(self, rng: np.random.Generator) -> str:
        alphabet = LATIN_UPPERCASE if rng.random() < 0.4 else DIGITS
        length = self.group_length.sample_int(rng)
        return "".join(alphabet[int(index)] for index in rng.integers(len(alphabet), size=length))


@dataclass(frozen=True)
class PhoneNumberPattern(TextPattern):
    """Телефонный номер в российском формате."""

    def generate(self, rng: np.random.Generator) -> str:
        digits = "".join(DIGITS[int(index)] for index in rng.integers(10, size=10))
        return f"+7 {digits[:3]} {digits[3:6]}-{digits[6:8]}-{digits[8:]}"


@dataclass(frozen=True)
class DatePattern(TextPattern):
    """Дата в формате ДД.ММ.ГГГГ."""

    def generate(self, rng: np.random.Generator) -> str:
        day = int(rng.integers(1, 29))
        month = int(rng.integers(1, 13))
        year = int(rng.integers(1990, 2027))
        return f"{day:02d}.{month:02d}.{year}"


@dataclass
class PatternSource(TextSource):
    """Взвешенная смесь шаблонов: числовая часть выборки."""

    patterns: tuple[TextPattern, ...] = field(
        default_factory=lambda: (
            DimensionPattern(),
            PricePattern(),
            ArticleCodePattern(),
            PhoneNumberPattern(),
            DatePattern(),
        )
    )
    weights: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5, 0.5)

    def sample_line(self, rng: np.random.Generator) -> str:
        return weighted_choice(rng, self.patterns, self.weights).generate(rng)


class MixedTextSource(TextSource):
    """Взвешенная смесь нескольких источников текста."""

    def __init__(self, sources: Sequence[TextSource], weights: Sequence[float]) -> None:
        if not sources:
            raise ValueError("at least one text source is required")
        self._sources = tuple(sources)
        self._weights = tuple(float(weight) for weight in weights)

    def sample_line(self, rng: np.random.Generator) -> str:
        return weighted_choice(rng, self._sources, self._weights).sample_line(rng)
