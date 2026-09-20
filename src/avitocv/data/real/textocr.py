"""Чтение разметки TextOCR: второй источник реальных кропов.

TextOCR размечен по словам, HierText — по строкам, и это ровно то, чего нам не хватало.
Слабое место модели — короткие кропы: на срезе с пропорциями до 2.5 она даёт 0.87 против
0.98 на длинных строках, а в тесте такие кропы есть. Строчный HierText их почти не содержит,
зато словарный TextOCR состоит из них на три четверти.

Соглашение о вершинах здесь то же, что в HierText: нулевая вершина — левый верхний угол
*текста*, дальше по часовой стрелке. Проверено по самой разметке: у 94.6% четырёхугольников
первое ребро направлено вправо, у 94.4% второе — вниз; остаток это действительно наклонённый
и вертикальный текст в кадре. Значит метка ориентации снова берётся даром, без распознавателя.

Снимки обоих датасетов взяты из Open Images и частично совпадают: 26 снимков сплита `val`
TextOCR лежат в валидации HierText. Их обязательно выкидывать, иначе одна и та же фотография
окажется и в обучении, и в валидации, и оценка перестанет что-либо значить.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from avitocv.data.real.hiertext import BoundingBox, top_edge_angle
from avitocv.data.manifest import CropRecord

WORD_SLICE = "textocr_word"
LINE_SLICE = "textocr_line"
ILLEGIBLE_MARKER = "."
QUADRILATERAL_COORDINATE_COUNT = 8
UPSIDE_DOWN_DEGREES = 180.0


@dataclass(frozen=True)
class WordFilter:
    """Отбор слов, пригодных для обучения.

    Порог пропорций ниже, чем у строк HierText: короткие кропы здесь не помеха, а сама цель.
    Нижняя граница всё же нужна — при пропорциях меньше единицы это уже не слово, а одиночный
    символ или вертикальная надпись, где понятия «верх строки» просто нет.
    """

    min_height: int = 12
    min_aspect_ratio: float = 1.0
    max_tilt_degrees: float = 35.0

    def is_usable(self, annotation: dict) -> bool:
        if annotation["utf8_string"] == ILLEGIBLE_MARKER:
            return False
        if len(annotation["points"]) != QUADRILATERAL_COORDINATE_COUNT:
            return False
        box = self.box_of(annotation)
        if box.height < self.min_height or box.aspect_ratio < self.min_aspect_ratio:
            return False
        return self.is_upright(annotation) is not None

    def is_upright(self, annotation: dict) -> bool | None:
        """True — слово ровное в кадре, False — перевёрнутое, None — наклон слишком велик."""
        angle = top_edge_angle(self.vertices_of(annotation))
        if abs(angle) <= self.max_tilt_degrees:
            return True
        if abs(abs(angle) - UPSIDE_DOWN_DEGREES) <= self.max_tilt_degrees:
            return False
        return None

    @staticmethod
    def vertices_of(annotation: dict) -> list[list[int]]:
        flat = annotation["points"]
        return [[int(round(flat[index])), int(round(flat[index + 1]))] for index in range(0, len(flat), 2)]

    @classmethod
    def box_of(cls, annotation: dict) -> BoundingBox:
        return BoundingBox.around(cls.vertices_of(annotation))


@dataclass(frozen=True)
class PlacedWord:
    """Слово вместе с геометрией, нужной для сборки строки."""

    box: BoundingBox
    angle: float
    slice_name: str = WORD_SLICE

    @property
    def vertical_center(self) -> float:
        return self.box.top + self.box.height / 2


@dataclass(frozen=True)
class LineGrouper:
    """Собирает слова одного снимка в строки — так, как это сделал бы детектор текста.

    Нужно это из-за геометрии. Тестовые кропы приходят от детектора строк: медиана пропорций
    4.9, коротких кропов всего 7%. Словарный TextOCR устроен наоборот — медиана 2.05, коротких
    65%, и при выравнивании под тест эффективный размер падает с 472 тысяч до 2.5 тысяч, то есть
    датасет превращается в горстку многократно повторённых длинных слов. Склейка соседних слов
    возвращает ему ту же природу, что у строчного HierText, но на 25 тысячах новых фотографий.

    Соседями считаются слова с похожим наклоном, перекрывающиеся по вертикали и разделённые
    горизонтальным промежутком меньше, чем их собственная высота: примерно так строку и
    определяет детектор.
    """

    max_angle_difference: float = 8.0
    min_vertical_overlap: float = 0.5
    max_gap_to_height: float = 1.2

    def group(self, words: list[PlacedWord]) -> list[list[PlacedWord]]:
        lines: list[list[PlacedWord]] = []
        for word in sorted(words, key=lambda item: (item.vertical_center, item.box.left)):
            host = next((line for line in lines if self._joins(line[-1], word)), None)
            if host is None:
                lines.append([word])
            else:
                host.append(word)
        return [sorted(line, key=lambda item: item.box.left) for line in lines]

    def _joins(self, previous: PlacedWord, candidate: PlacedWord) -> bool:
        if abs(previous.angle - candidate.angle) > self.max_angle_difference:
            return False
        if self._vertical_overlap(previous.box, candidate.box) < self.min_vertical_overlap:
            return False
        return self._gap(previous.box, candidate.box) <= self.max_gap_to_height * min(
            previous.box.height, candidate.box.height)

    @staticmethod
    def _vertical_overlap(first: BoundingBox, second: BoundingBox) -> float:
        """Доля перекрытия по вертикали от высоты меньшего бокса."""
        top = max(first.top, second.top)
        bottom = min(first.top + first.height, second.top + second.height)
        return max(bottom - top, 0) / max(min(first.height, second.height), 1)

    @staticmethod
    def _gap(first: BoundingBox, second: BoundingBox) -> float:
        """Горизонтальный зазор между боксами; ноль, если они перекрываются."""
        left, right = (first, second) if first.left <= second.left else (second, first)
        return max(right.left - (left.left + left.width), 0)


@dataclass(frozen=True)
class RunEmitter:
    """Превращает строку из слов в кропы разной длины.

    Одна только целая строка дала бы слишком узкий разброс пропорций, а одни слова — слишком
    короткий. Поэтому берутся все непрерывные отрезки до `max_run_length` слов плюс строка
    целиком: набор покрывает диапазон от одного слова до всей строки, и выравнивание под тест
    выбирает из него то, что нужно, а не повторяет одно и то же.
    """

    max_run_length: int = 3

    def emit(self, line: list[PlacedWord]) -> list[list[PlacedWord]]:
        runs = [
            line[start:start + length]
            for length in range(1, min(self.max_run_length, len(line)) + 1)
            for start in range(len(line) - length + 1)
        ]
        if len(line) > self.max_run_length:
            runs.append(line)
        return runs


def merge_boxes(words: list[PlacedWord]) -> BoundingBox:
    """Axis-aligned прямоугольник вокруг всех слов отрезка."""
    left = min(word.box.left for word in words)
    top = min(word.box.top for word in words)
    right = max(word.box.left + word.box.width for word in words)
    bottom = max(word.box.top + word.box.height for word in words)
    return BoundingBox(left=left, top=top, width=right - left, height=bottom - top)


@dataclass
class WordBuildReport:
    """Сколько слов принято и по какой причине отброшено."""

    accepted: int = 0
    upside_down: int = 0
    rejected: int = 0
    excluded_images: int = 0


class TextOcrManifestBuilder:
    """Превращает аннотации TextOCR в записи манифеста, оставляя только ровные слова.

    Без `grouper` каждое слово становится отдельным кропом; с ним слова снимка сначала
    собираются в строки, и кропом становится непрерывный отрезок строки. Второй режим и нужен
    на практике — он даёт геометрию, похожую на тестовую.
    """

    def __init__(
        self,
        word_filter: WordFilter,
        excluded_image_ids: Iterable[str] = (),
        grouper: LineGrouper | None = None,
        emitter: RunEmitter | None = None,
    ) -> None:
        self._filter = word_filter
        self._excluded = set(excluded_image_ids)
        self._grouper = grouper
        self._emitter = emitter or RunEmitter()
        self.report = WordBuildReport()

    def build(self, payload: dict, images_dir: Path) -> list[CropRecord]:
        return [record for image_id in payload["imgs"] for record in self._from_image(payload, image_id, images_dir)]

    def _from_image(self, payload: dict, image_id: str, images_dir: Path) -> Iterator[CropRecord]:
        if image_id in self._excluded:
            self.report.excluded_images += 1
            return
        # `file_name` в разметке указывает на папку `train/`, а архив распаковывается в
        # `train_images/`, поэтому путь собирается из идентификатора, а не из разметки.
        relative_path = (images_dir / f"{image_id}.jpg").as_posix()
        words = [word for word in (self._to_word(payload["anns"][annotation_id])
                                   for annotation_id in payload["imgToAnns"][image_id]) if word is not None]
        for run in self._runs(words):
            yield self._to_record(run, relative_path)

    def _runs(self, words: list[PlacedWord]) -> list[list[PlacedWord]]:
        if self._grouper is None:
            return [[word] for word in words]
        return [run for line in self._grouper.group(words) for run in self._emitter.emit(line)]

    def _to_word(self, annotation: dict) -> PlacedWord | None:
        if not self._filter.is_usable(annotation):
            self.report.rejected += 1
            return None
        if not self._filter.is_upright(annotation):
            self.report.upside_down += 1
            return None
        self.report.accepted += 1
        return PlacedWord(box=self._filter.box_of(annotation),
                          angle=top_edge_angle(self._filter.vertices_of(annotation)))

    @staticmethod
    def _to_record(run: list[PlacedWord], relative_path: str) -> CropRecord:
        box = merge_boxes(run)
        return CropRecord(
            image_path=relative_path,
            left=box.left,
            top=box.top,
            width=box.width,
            height=box.height,
            slice_name=WORD_SLICE if len(run) == 1 else LINE_SLICE,
        )
