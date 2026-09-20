"""Геометрия кропа: превращает отрисованные строки в бокс, похожий на выход детектора.

Здесь воспроизводятся свойства тестовых кропов, которые видно глазом: неплотные отступы,
иногда срезанные сверху или снизу глифы, обрезки соседних строк сверху и снизу, и произвольное
окно по горизонтали, режущее текст посреди слова.

Ширина холста подбирается так, чтобы окно дало ровно нужный aspect ratio: если строка короче
окна — холст расширяется фоном, если длиннее — окно скользит вдоль строки.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from avitocv.data.synthesis.backgrounds import BackgroundProvider
from avitocv.data.synthesis.corpus import TextSource
from avitocv.data.synthesis.fonts import SupportedTextFilter
from avitocv.data.synthesis.palettes import ColorScheme
from avitocv.data.synthesis.rendering import GlyphLayer, LetterCaseSampler, LineStyle, TextLineRenderer
from avitocv.data.sampling import ValueRange, happens

MAX_FIT_ATTEMPTS = 3
MAX_TEXT_ATTEMPTS = 8
FIT_SAFETY_FACTOR = 1.15     # запас на неравномерную ширину символов при оценке нужной длины
MAX_FITTED_CHARACTERS = 400  # потолок длины строки, чтобы растр не разрастался
MIN_CANVAS_SIDE = 8
MIN_TILT_DEGREES = 0.05


class FittedLineRenderer:
    """Рисует строку, дотягивая её длину до ширины будущего окна обрезки."""

    def __init__(
        self,
        renderer: TextLineRenderer,
        case_sampler: LetterCaseSampler,
        text_filter: SupportedTextFilter,
    ) -> None:
        self._renderer = renderer
        self._case_sampler = case_sampler
        self._text_filter = text_filter

    def render_at_least(
        self,
        source: TextSource,
        style: LineStyle,
        min_aspect: float,
        rng: np.random.Generator,
    ) -> GlyphLayer:
        text = self._take_text(source, style, rng)
        layer = self._renderer.render(text, style)
        for _ in range(MAX_FIT_ATTEMPTS):
            if layer.ink_width >= min_aspect * layer.ink_height:
                return layer
            text = self._grow(text, source, style, layer, min_aspect, rng)
            layer = self._renderer.render(text, style)
        return layer

    def _take_text(self, source: TextSource, style: LineStyle, rng: np.random.Generator) -> str:
        for _ in range(MAX_TEXT_ATTEMPTS):
            candidate = self._text_filter.filter(self._case_sampler.apply(source.sample_line(rng), rng), style.font)
            if candidate:
                return candidate
        raise ValueError(f"text source produced nothing renderable with {style.font.path.name}")

    def _grow(
        self,
        text: str,
        source: TextSource,
        style: LineStyle,
        layer: GlyphLayer,
        min_aspect: float,
        rng: np.random.Generator,
    ) -> str:
        # Наращиваем по измеренной ширине, а не по числу символов: оценка «символов столько же,
        # сколько пикселей в пропорции» промахивалась на пробелах и узких буквах, и строка
        # вырастала в разы длиннее нужного. Лишняя длина потом всё равно обрезается слоем,
        # то есть это была чистая трата растеризации.
        required_width = min_aspect * layer.ink_height * FIT_SAFETY_FACTOR
        grown = text
        while len(grown) < MAX_FITTED_CHARACTERS:
            if self._renderer.measure_width(grown, style) >= required_width:
                break
            grown = f"{grown} {self._take_text(source, style, rng)}"
        return grown[:MAX_FITTED_CHARACTERS]


@dataclass(frozen=True)
class CropLayout:
    """Разыгранные параметры компоновки одного кропа."""

    pad_top_ratio: float
    pad_bottom_ratio: float
    neighbour_gap_ratio: float
    neighbour_visible_ratio: float
    horizontal_slack_ratio: float
    fill_ratio: float
    tilt_degrees: float
    has_neighbour_above: bool
    has_neighbour_below: bool


@dataclass(frozen=True)
class CropLayoutSampler:
    """Разыгрывает компоновку: отступы, соседние строки, заполнение по ширине."""

    # Нижняя граница отрицательная намеренно: часть кропов получает срезанные сверху или
    # снизу глифы, как у неточно поставленного бокса детектора.
    pad_ratio: ValueRange = ValueRange(-0.10, 0.55)
    neighbour_gap_ratio: ValueRange = ValueRange(0.08, 0.45)
    neighbour_visible_ratio: ValueRange = ValueRange(0.05, 0.50)
    horizontal_slack_ratio: ValueRange = ValueRange(0.0, 0.35)
    partial_fill_ratio: ValueRange = ValueRange(0.5, 0.97)
    neighbour_probability: float = 0.35
    partial_fill_probability: float = 0.35
    # Почти все строки близки к горизонтали, но хвост нужен: в разметке HierText у боксов выше
    # 100 px медиана наклона 9.6 градуса, а треть из них наклонена сильнее 15.
    tilt_sigma_degrees: float = 6.0
    max_tilt_degrees: float = 18.0
    # Насколько бокс вправе стать выше строки. В разметке HierText текст занимает не меньше
    # половины высоты бокса, что соответствует удвоению.
    max_box_growth: float = 2.0
    tilt_probability: float = 0.45

    def sample(self, rng: np.random.Generator, target_aspect: float = 1.0) -> CropLayout:
        pad_top = self.pad_ratio.sample(rng)
        pad_bottom = self.pad_ratio.sample(rng)
        return CropLayout(
            pad_top_ratio=pad_top,
            pad_bottom_ratio=pad_bottom,
            neighbour_gap_ratio=self.neighbour_gap_ratio.sample(rng),
            neighbour_visible_ratio=self.neighbour_visible_ratio.sample(rng),
            horizontal_slack_ratio=self.horizontal_slack_ratio.sample(rng),
            fill_ratio=self._sample_fill_ratio(rng),
            tilt_degrees=self._sample_tilt(rng, target_aspect),
            has_neighbour_above=pad_top > 0.0 and happens(rng, self.neighbour_probability),
            has_neighbour_below=pad_bottom > 0.0 and happens(rng, self.neighbour_probability),
        )

    def _sample_fill_ratio(self, rng: np.random.Generator) -> float:
        return self.partial_fill_ratio.sample(rng) if happens(rng, self.partial_fill_probability) else 1.0

    def _sample_tilt(self, rng: np.random.Generator, target_aspect: float) -> float:
        """Наклон строки, ограниченный её пропорциями.

        Длинную строку нельзя сильно наклонить: её бокс со сторонами по осям раздувается как
        `A * sin(a) + cos(a)`, то есть пропорционально длине. При aspect 12 и наклоне 10 градусов
        бокс вырос бы втрое, и от текста осталась бы тонкая диагональ — таких кропов в реальных
        данных нет вовсе, там отношение текста к боксу не опускается ниже 0.51.

        И это не ограничение модели, а физика сцены: длинные строки живут в документах и на
        вывесках, которые снимают почти фронтально, а сильно наклонёнными бывают короткие
        надписи. Поэтому предел наклона выводится из допустимого раздувания бокса.

        Знак симметричен, иначе наклон сам стал бы признаком ориентации.
        """
        if not happens(rng, self.tilt_probability):
            return 0.0
        angle = min(abs(rng.normal(0.0, self.tilt_sigma_degrees)), self._tilt_ceiling(target_aspect))
        return angle if rng.random() < 0.5 else -angle

    def _tilt_ceiling(self, target_aspect: float) -> float:
        """Максимальный наклон, при котором бокс раздувается не сильнее заданного предела."""
        allowed_sine = (self.max_box_growth - 1.0) / max(target_aspect, 1.0)
        if allowed_sine >= 1.0:
            return self.max_tilt_degrees
        return min(self.max_tilt_degrees, math.degrees(math.asin(allowed_sine)))


@dataclass(frozen=True)
class CropRequest:
    """Всё, что нужно композитору: слои строк, компоновка, цвет, целевой aspect ratio."""

    main: GlyphLayer
    above: GlyphLayer | None
    below: GlyphLayer | None
    layout: CropLayout
    color_scheme: ColorScheme
    target_aspect: float


@dataclass(frozen=True)
class _CanvasGeometry:
    """Посчитанные размеры холста и положение текста на нём."""

    width: int
    height: int
    window_width: int
    text_left: float
    pad_top: float


class CropComposer:
    """Собирает холст, кладёт на него строки и вырезает окно нужной пропорции."""

    def __init__(self, background: BackgroundProvider) -> None:
        self._background = background

    def compose(self, request: CropRequest, rng: np.random.Generator) -> Image.Image:
        geometry = self._measure(request, rng)
        canvas = self._background.render((geometry.width, geometry.height), request.color_scheme, rng)
        self._paste_neighbour(canvas, request, geometry, request.above, is_above=True)
        self._paste_neighbour(canvas, request, geometry, request.below, is_above=False)
        self._paste(canvas, request.main, geometry.text_left, geometry.pad_top)
        # Поворот идёт до вырезания окна, потому что детектор обводит уже наклонённый текст:
        # его прямоугольник со сторонами по осям оказывается выше самой строки, и текст
        # занимает лишь часть высоты кропа. Если повернуть после вырезания, бокс не раздуется,
        # и такой кроп ничем не будет похож на настоящий.
        canvas = self._tilt(canvas, request.layout.tilt_degrees)
        return self._cut_window(canvas, request.target_aspect, rng)

    def _measure(self, request: CropRequest, rng: np.random.Generator) -> _CanvasGeometry:
        ink_height = request.main.ink_height
        pad_top = self._padding(request, request.above, request.layout.pad_top_ratio)
        pad_bottom = self._padding(request, request.below, request.layout.pad_bottom_ratio)
        height = max(MIN_CANVAS_SIDE, int(pad_top + ink_height + pad_bottom))
        window_width = max(MIN_CANVAS_SIDE, int(round(height * request.target_aspect)))
        slack = request.layout.horizontal_slack_ratio * ink_height
        content_width = int(request.main.ink_width + 2 * slack)
        # Холст не бывает уже окна: если строка короче нужной ширины, разница добирается
        # фоном, если длиннее — окно скользит вдоль строки. Благодаря этому итоговый
        # aspect ratio равен ровно запрошенному, а не «какой получился».
        width = max(content_width, window_width)
        return _CanvasGeometry(
            width=width,
            height=height,
            window_width=window_width,
            # Свободную ширину распределяем случайно, иначе короткий текст всегда прижимался
            # бы к левому краю кропа.
            text_left=slack + float(rng.integers(0, max(1, width - content_width + 1))),
            pad_top=pad_top,
        )

    def _padding(self, request: CropRequest, neighbour: GlyphLayer | None, pad_ratio: float) -> float:
        ink_height = request.main.ink_height
        padding = pad_ratio * ink_height
        if neighbour is None:
            return padding
        # Соседняя строка задаёт минимальный отступ: она должна попасть в кроп ровно на
        # visible пикселей, а остальная её часть уйти за край холста и оказаться срезанной.
        gap = request.layout.neighbour_gap_ratio * ink_height
        visible = request.layout.neighbour_visible_ratio * neighbour.ink_height
        return max(padding, visible + gap)

    def _paste_neighbour(
        self,
        canvas: Image.Image,
        request: CropRequest,
        geometry: _CanvasGeometry,
        neighbour: GlyphLayer | None,
        is_above: bool,
    ) -> None:
        if neighbour is None:
            return
        gap = request.layout.neighbour_gap_ratio * request.main.ink_height
        # Для строки сверху top уходит в отрицательные значения — так и задумано: видна
        # только её нижняя часть, остальное обрезается краем холста.
        if is_above:
            top = geometry.pad_top - gap - neighbour.ink_height
        else:
            top = geometry.pad_top + request.main.ink_height + gap
        self._paste(canvas, neighbour, geometry.text_left, top)

    def _paste(self, canvas: Image.Image, layer: GlyphLayer, left: float, top: float) -> None:
        # Позиционируем по прямоугольнику чернил, а не по началу слоя: слой содержит
        # технический запас LAYER_MARGIN, который нужно вычесть из смещения.
        offset = (int(left) - layer.ink_box[0], int(top) - layer.ink_box[1])
        canvas.paste(layer.image, offset, layer.image)

    def _tilt(self, canvas: Image.Image, degrees: float) -> Image.Image:
        if abs(degrees) < MIN_TILT_DEGREES:
            return canvas
        # expand=True даёт новый описанный прямоугольник — то самое раздувание бокса.
        return canvas.rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=None)

    def _cut_window(self, canvas: Image.Image, target_aspect: float, rng: np.random.Generator) -> Image.Image:
        # Ширина окна считается уже по повёрнутому холсту, поэтому итоговый aspect ratio
        # остаётся ровно тем, что запросили по профилю теста.
        window_width = max(MIN_CANVAS_SIDE, int(round(canvas.height * target_aspect)))
        width = min(canvas.width, window_width)
        left = int(rng.integers(0, canvas.width - width + 1))
        return canvas.crop((left, 0, left + width, canvas.height)).convert("RGB")
