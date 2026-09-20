"""Нарезает фоновые текстуры из реальных фотографий.

Фотографии берутся из обучающего сплита HierText, который уже скачан. Важно, что его разметка
позволяет вырезать участки **без текста**: патч с чужим текстом внутри был бы источником
ложной метки, потому что этот текст повернулся бы вместе с фоном, а к целевой строке отношения
не имел бы.

Валидационный сплит не используется намеренно: он остаётся чистым для оценки.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rich.progress import Progress

TEXT_MARGIN_RATIO = 0.25
MAX_PLACEMENT_ATTEMPTS = 24
MIN_PATCH_SIDE = 96
PATCH_SUFFIX = ".jpg"
PATCH_QUALITY = 90


@dataclass(frozen=True)
class Rectangle:
    """Прямоугольник со сторонами по осям."""

    left: int
    top: int
    right: int
    bottom: int

    def grown(self, margin: int) -> "Rectangle":
        return Rectangle(self.left - margin, self.top - margin, self.right + margin, self.bottom + margin)

    def overlaps(self, other: "Rectangle") -> bool:
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )


def text_rectangles(annotation: dict) -> list[Rectangle]:
    """Все размеченные строки снимка, расширенные запасом.

    Запас нужен, потому что разметка обтягивает глифы плотно, а рядом остаются их тени,
    обводки и части соседних строк.
    """
    rectangles = []
    for paragraph in annotation["paragraphs"]:
        for line in paragraph["lines"]:
            xs = [point[0] for point in line["vertices"]]
            ys = [point[1] for point in line["vertices"]]
            box = Rectangle(min(xs), min(ys), max(xs), max(ys))
            rectangles.append(box.grown(int((box.bottom - box.top) * TEXT_MARGIN_RATIO) + 2))
    return rectangles


class PatchFinder:
    """Ищет на снимке прямоугольник, не задевающий ни одной размеченной строки."""

    def __init__(self, patch_side: int) -> None:
        self._patch_side = patch_side

    def find(self, size: tuple[int, int], obstacles: list[Rectangle], rng: np.random.Generator) -> Rectangle | None:
        width, height = size
        side = min(self._patch_side, width, height)
        if side < MIN_PATCH_SIDE:
            return None
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            left = int(rng.integers(0, width - side + 1))
            top = int(rng.integers(0, height - side + 1))
            candidate = Rectangle(left, top, left + side, top + side)
            if not any(candidate.overlaps(obstacle) for obstacle in obstacles):
                return candidate
        return None


class BackgroundExtractor:
    """Сохраняет безтекстовые патчи в каталог фонов."""

    def __init__(self, images_dir: Path, output_dir: Path, finder: PatchFinder) -> None:
        self._images_dir = images_dir
        self._output_dir = output_dir
        self._finder = finder
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, annotations: list[dict], limit: int, seed: int) -> int:
        rng = np.random.default_rng(seed)
        saved = 0
        with Progress() as progress:
            task = progress.add_task("нарезка фонов", total=limit)
            for annotation in annotations:
                if saved >= limit:
                    break
                if self._extract_one(annotation, rng, saved):
                    saved += 1
                    progress.advance(task)
        return saved

    def _extract_one(self, annotation: dict, rng: np.random.Generator, index: int) -> bool:
        path = self._images_dir / f"{annotation['image_id']}.jpg"
        if not path.exists():
            return False
        with Image.open(path) as image:
            found = self._finder.find(image.size, text_rectangles(annotation), rng)
            if found is None:
                return False
            patch = image.convert("RGB").crop((found.left, found.top, found.right, found.bottom))
        patch.save(self._output_dir / f"{index:05d}{PATCH_SUFFIX}", quality=PATCH_QUALITY)
        return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Нарезать фоновые текстуры из реальных фотографий")
    parser.add_argument("--annotations", type=Path, default=Path("data/real/hiertext/train.jsonl.gz"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/real/hiertext/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/backgrounds"))
    parser.add_argument("--patch-side", type=int, default=320)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    annotations = json.loads(gzip.open(arguments.annotations, "rt", encoding="utf-8").read())["annotations"]
    extractor = BackgroundExtractor(
        arguments.images_dir,
        arguments.output_dir,
        PatchFinder(arguments.patch_side),
    )
    saved = extractor.extract(annotations, arguments.limit, arguments.seed)
    print(f"сохранено фонов: {saved} в {arguments.output_dir}")


if __name__ == "__main__":
    main()
