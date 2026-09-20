"""Чтение разметки HierText: откуда берутся метки для реальных кропов.

HierText размечает текст на уровне строк четырёхугольниками, и порядок вершин в них задаёт
направление чтения — первая вершина это левый верхний угол *текста*, а не бокса. Поэтому угол
верхнего ребра сам говорит, ровная строка в кадре или перевёрнутая, и распознаватель для
получения метки не нужен вообще. Проверено на контактных листах: строки, признанные ровными,
читаются нормально, признанные перевёрнутыми — действительно вверх ногами.

Кроп берётся как axis-aligned прямоугольник вокруг четырёхугольника, а не выпрямляется
перспективой: в тестовой выборке кропы именно прямоугольные, и слегка наклонённый текст лежит
в них под углом. Выпрямление сделало бы валидацию чище теста.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from avitocv.data.manifest import CropRecord

UPSIDE_DOWN_DEGREES = 180.0
SCENE_SLICE = "hiertext_scene"
HANDWRITTEN_SLICE = "hiertext_handwritten"
QUADRILATERAL_VERTEX_COUNT = 4


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned прямоугольник вокруг размеченного четырёхугольника."""

    left: int
    top: int
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @classmethod
    def around(cls, vertices: list[list[int]]) -> "BoundingBox":
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        return cls(left=min(xs), top=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))


def top_edge_angle(vertices: list[list[int]]) -> float:
    """Угол верхнего ребра текста в градусах: около 0 у ровной строки, около 180 у перевёрнутой."""
    (start_x, start_y), (end_x, end_y) = vertices[0], vertices[1]
    return math.degrees(math.atan2(end_y - start_y, end_x - start_x))


@dataclass(frozen=True)
class LineFilter:
    """Отбор строк, пригодных для валидации: разборчивые, почти горизонтальные, не микроскопические."""

    min_height: int = 12
    min_aspect_ratio: float = 1.5
    # Строки с наклоном сильнее этого не похожи на тестовые кропы: там текст почти горизонтален.
    max_tilt_degrees: float = 35.0

    def is_usable(self, line: dict) -> bool:
        if not line["legible"] or line["vertical"]:
            return False
        if len(line["vertices"]) != QUADRILATERAL_VERTEX_COUNT:
            return False
        box = BoundingBox.around(line["vertices"])
        if box.height < self.min_height or box.aspect_ratio < self.min_aspect_ratio:
            return False
        return self.is_upright(line) is not None

    def is_upright(self, line: dict) -> bool | None:
        """True — строка ровная в кадре, False — перевёрнутая, None — наклон слишком велик."""
        angle = top_edge_angle(line["vertices"])
        if abs(angle) <= self.max_tilt_degrees:
            return True
        if abs(abs(angle) - UPSIDE_DOWN_DEGREES) <= self.max_tilt_degrees:
            return False
        return None


@dataclass
class ManifestBuildReport:
    """Сколько строк принято и по какой причине отброшено."""

    accepted: int = 0
    upside_down: int = 0
    rejected: int = 0


class HierTextManifestBuilder:
    """Превращает аннотации HierText в записи манифеста, оставляя только ровные строки."""

    def __init__(self, line_filter: LineFilter) -> None:
        self._filter = line_filter
        self.report = ManifestBuildReport()

    def build(self, annotations: list[dict], images_dir: Path) -> list[CropRecord]:
        return [record for annotation in annotations for record in self._from_image(annotation, images_dir)]

    def _from_image(self, annotation: dict, images_dir: Path) -> Iterator[CropRecord]:
        relative_path = (images_dir / f"{annotation['image_id']}.jpg").as_posix()
        for paragraph in annotation["paragraphs"]:
            for line in paragraph["lines"]:
                record = self._from_line(line, relative_path)
                if record is not None:
                    yield record

    def _from_line(self, line: dict, relative_path: str) -> CropRecord | None:
        if not self._filter.is_usable(line):
            self.report.rejected += 1
            return None
        # Перевёрнутые в кадре строки пропускаем: источник по контракту отдаёт ровные кропы,
        # а поворачивать их пришлось бы уже после вырезания — лишняя стадия ради 1% данных.
        if not self._filter.is_upright(line):
            self.report.upside_down += 1
            return None
        box = BoundingBox.around(line["vertices"])
        self.report.accepted += 1
        return CropRecord(
            image_path=relative_path,
            left=box.left,
            top=box.top,
            width=box.width,
            height=box.height,
            slice_name=HANDWRITTEN_SLICE if line["handwritten"] else SCENE_SLICE,
        )
