"""Реестр шрифтов и покрытие символов.

Решает два вопроса: каким шрифтом печатать строку (шрифт обязан покрывать нужную письменность)
и какие символы из строки этот шрифт умеет рисовать — иначе вместо глифов в кроп попадают
пустые прямоугольники.

Покрытие кодпоинтов вычисляется один раз при сборке индекса и хранится в нём диапазонами,
поэтому во время обучения fontTools не вызывается вообще.
"""

from __future__ import annotations

import json
import struct
from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont

from avitocv.data.synthesis.writing_systems import DIGIT_CODEPOINTS, SCRIPT_CODEPOINTS, Script

FONT_SUFFIXES = (".ttf", ".otf")
# Не 1.0: часть вполне пригодных шрифтов не содержит редких букв вроде Ё, и требовать
# полного покрытия значило бы выбросить их целиком.
MIN_SCRIPT_COVERAGE = 0.98
# Индекс собирается по сотням скачанных файлов; битый или экзотический шрифт должен просто
# выпасть из выборки, а не обрушить сборку.
FONT_SCAN_ERRORS = (TTLibError, KeyError, OSError, ValueError, struct.error)
# Пробная отрисовка обязана идти в тех же условиях, что и настоящая, иначе она ничего не ловит.
# Обводка нужна потому, что битые контуры проявляются именно через обводчик FreeType. Размер
# важен не меньше: у части декоративных шрифтов глиф спокойно рисуется в 24 px и взрывает
# выделение растра в 64 px с обводкой 3
# Значения должны совпадать с `rendering.NOMINAL_FONT_SIZE` и верхней границей `stroke_width`.
PROBE_FONT_SIZE = 64
PROBE_STROKE_WIDTH = 3
PROBE_CANVAS_SIZE = (1024, 512)


def _to_ranges(codepoints: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(codepoints))
    if not ordered:
        return ()
    ranges = []
    start = previous = ordered[0]
    for codepoint in ordered[1:]:
        if codepoint == previous + 1:
            previous = codepoint
            continue
        ranges.append((start, previous))
        start = previous = codepoint
    ranges.append((start, previous))
    return tuple(ranges)


@dataclass(frozen=True)
class CodepointCoverage:
    """Поддерживаемые кодпоинты, сжатые в диапазоны; проверка бинарным поиском."""

    ranges: tuple[tuple[int, int], ...]
    _starts: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _ends: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Границы разложены в отдельные кортежи под bisect в contains(): он вызывается на
        # каждый символ каждой строки, и распаковка ranges на месте была бы видна в профиле.
        # object.__setattr__ — единственный способ заполнить поле у frozen dataclass.
        object.__setattr__(self, "_starts", tuple(start for start, _ in self.ranges))
        object.__setattr__(self, "_ends", tuple(end for _, end in self.ranges))

    @classmethod
    def from_codepoints(cls, codepoints: Iterable[int]) -> "CodepointCoverage":
        return cls(_to_ranges(codepoints))

    @classmethod
    def from_flat(cls, flat: Sequence[int]) -> "CodepointCoverage":
        return cls(tuple((flat[index], flat[index + 1]) for index in range(0, len(flat), 2)))

    def to_flat(self) -> list[int]:
        return [bound for bounds in self.ranges for bound in bounds]

    def contains(self, codepoint: int) -> bool:
        # Диапазоны отсортированы и не пересекаются, поэтому достаточно взять последний,
        # начинающийся не позже кодпоинта, и проверить его правую границу.
        position = bisect_right(self._starts, codepoint) - 1
        return position >= 0 and codepoint <= self._ends[position]

    def share_of(self, codepoints: Sequence[int]) -> float:
        if not codepoints:
            return 0.0
        return sum(1 for codepoint in codepoints if self.contains(codepoint)) / len(codepoints)


@dataclass(frozen=True)
class FontAsset:
    """Шрифт в реестре: путь, покрываемые письменности, покрытие кодпоинтов."""

    path: Path
    scripts: frozenset[Script]
    coverage: CodepointCoverage

    def can_render(self, script: Script) -> bool:
        return script in self.scripts

    def to_dict(self) -> dict:
        return {
            "path": self.path.as_posix(),
            "scripts": sorted(script.value for script in self.scripts),
            "coverage": self.coverage.to_flat(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FontAsset":
        return cls(
            path=Path(payload["path"]),
            scripts=frozenset(Script(value) for value in payload["scripts"]),
            coverage=CodepointCoverage.from_flat(payload["coverage"]),
        )


class FontCoverageInspector:
    """Читает cmap файла шрифта и решает, годится ли он для генерации."""

    def inspect(self, path: Path) -> FontAsset | None:
        character_map = self._read_character_map(path)
        if character_map is None:
            return None
        coverage = CodepointCoverage.from_codepoints(character_map.keys())
        if coverage.share_of(DIGIT_CODEPOINTS) < MIN_SCRIPT_COVERAGE:
            return None
        scripts = {
            script for script, codepoints in SCRIPT_CODEPOINTS.items()
            if coverage.share_of(codepoints) >= MIN_SCRIPT_COVERAGE
        }
        if not scripts:
            return None
        return FontAsset(path=path, scripts=frozenset(scripts), coverage=coverage)

    def _read_character_map(self, path: Path) -> dict | None:
        try:
            with TTFont(path, fontNumber=0, lazy=True) as font:
                return font.getBestCmap()
        except FONT_SCAN_ERRORS:
            return None


class FontRenderProbe:
    """Проверяет, что шрифт действительно рисуется, а не только заявляет символы в cmap.

    Среди нескольких сотен скачанных шрифтов попадаются экземпляры с изломанными контурами:
    например, в `RubikPixels` глиф `П` при любой ненулевой обводке заставляет FreeType
    выделять гигантский растр, и PIL падает с `OSError`. В cmap такой шрифт выглядит
    совершенно нормальным, поэтому поймать его можно только отрисовкой.

    Проверка делается один раз при сборке индекса: во время обучения падений уже не будет.
    """

    def can_render(self, asset: "FontAsset") -> bool:
        font = ImageFont.truetype(str(asset.path), PROBE_FONT_SIZE)
        draw = ImageDraw.Draw(Image.new("RGBA", PROBE_CANVAS_SIZE, (0, 0, 0, 0)))
        for script in asset.scripts:
            if not self._draws(draw, font, self._alphabet(script)):
                return False
        return self._draws(draw, font, self._alphabet(None))

    @staticmethod
    def _alphabet(script: Script | None) -> str:
        codepoints = DIGIT_CODEPOINTS if script is None else SCRIPT_CODEPOINTS[script]
        return "".join(chr(codepoint) for codepoint in codepoints)

    @staticmethod
    def _draws(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, alphabet: str) -> bool:
        # Посимвольно, потому что генератор рисует строки и целиком, и по буквам при заданном
        # межбуквенном интервале, а падает именно отдельный глиф.
        for character in alphabet:
            try:
                draw.text((0, 0), character, font=font, fill=(255, 255, 255, 255),
                          stroke_width=PROBE_STROKE_WIDTH, stroke_fill=(0, 0, 0, 255))
            except OSError:
                return False
        return True


class FontRegistry:
    """Коллекция пригодных шрифтов с выборкой по письменности; хранится как индекс."""

    def __init__(self, assets: Iterable[FontAsset]) -> None:
        self._assets = tuple(assets)
        if not self._assets:
            raise ValueError("font registry requires at least one usable font")
        self._by_script = {
            script: tuple(asset for asset in self._assets if asset.can_render(script))
            for script in Script
        }

    def __len__(self) -> int:
        return len(self._assets)

    def count_for(self, script: Script) -> int:
        return len(self._by_script[script])

    def sample(self, rng: np.random.Generator, script: Script) -> FontAsset:
        candidates = self._by_script[script]
        if not candidates:
            raise ValueError(f"no font in the registry supports {script.value}")
        return candidates[int(rng.integers(len(candidates)))]

    @classmethod
    def from_directories(cls, directories: Sequence[Path]) -> "FontRegistry":
        inspector = FontCoverageInspector()
        probe = FontRenderProbe()
        assets = []
        for directory in directories:
            for path in sorted(Path(directory).rglob("*")):
                if path.suffix.lower() not in FONT_SUFFIXES:
                    continue
                asset = inspector.inspect(path)
                if asset is not None and probe.can_render(asset):
                    assets.append(asset)
        return cls(assets)

    @classmethod
    def load_index(cls, path: Path) -> "FontRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(FontAsset.from_dict(item) for item in payload["fonts"])

    def save_index(self, path: Path) -> None:
        payload = {"fonts": [asset.to_dict() for asset in self._assets]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


class SupportedTextFilter:
    """Выбрасывает из строки символы, которых нет в выбранном шрифте."""

    def filter(self, text: str, font: FontAsset) -> str:
        kept = "".join(character for character in text if font.coverage.contains(ord(character)))
        return " ".join(kept.split())


@lru_cache(maxsize=512)
def load_truetype_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)
