"""Отрисовка одной строки текста в слой с альфа-каналом.

Результат — `GlyphLayer`: только чернила, без фона, вместе с плотным прямоугольником по
непрозрачным пикселям. Этот прямоугольник и есть «строка» для всей последующей геометрии:
отступы и размеры в `composition` считаются в долях его высоты, потому что реальные боксы
детектора обтягивают именно чернила, а не строчный интерлиньяж шрифта.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from avitocv.data.synthesis.fonts import FontAsset, FontRegistry, load_truetype_font
from avitocv.data.synthesis.palettes import ColorScheme, ColorSchemeSampler
from avitocv.data.sampling import ValueRange, happens
from avitocv.data.synthesis.writing_systems import Script

NOMINAL_FONT_SIZE = 64
TRANSPARENT = (0, 0, 0, 0)
LAYER_MARGIN = 24        # запас вокруг текста под обводку и под сдвиг от наклона
MAX_LAYER_WIDTH = 8192   # предел растра: кропы с aspect ratio под 40 иначе съедают память
# Потолок площади растра под маску текста. Законная строка в 64 px даёт высоту растра около
# 100 px, у самых размашистых декоративных шрифтов с обводкой — до 300; при предельной ширине
# 8192 это 2.5 мегапикселя, так что четыре мегапикселя ничего разумного не режут.
MAX_LAYER_PIXELS = 4 * 1024 * 1024
TRUNCATION_PASSES = 6    # уточнений оценки длины строки, влезающей в предел
LAYER_HEIGHT_SAFETY = 1.3  # ascent+descent покрывает не все глифы — диакритика и росчерки выходят за метрики
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class LineStyle:
    """Полный набор параметров отрисовки: шрифт, кегль, цвет, интервал, наклон."""

    font: FontAsset
    font_size: int
    color_scheme: ColorScheme
    letter_spacing_ratio: float
    stroke_width: int
    skew_degrees: float

    @property
    def has_letter_spacing(self) -> bool:
        return abs(self.letter_spacing_ratio) > 1e-3

    @property
    def has_skew(self) -> bool:
        return abs(self.skew_degrees) > 1e-3


@dataclass(frozen=True)
class GlyphLayer:
    """RGBA-слой с отрисованным текстом и плотный прямоугольник чернил в нём."""

    image: Image.Image
    ink_box: Box

    @property
    def ink_height(self) -> int:
        return self.ink_box[3] - self.ink_box[1]

    @property
    def ink_width(self) -> int:
        return self.ink_box[2] - self.ink_box[0]


class FontRenderError(RuntimeError):
    """Шрифт не может нарисовать даже один символ в пределах допустимого растра.

    Отдельный тип нужен, чтобы падение называло виновника: раньше здесь вылезал `OSError`
    из недр PIL, и на поиск шрифта уходил отдельный прогон по всей выборке.
    """

    def __init__(self, font_path, text: str) -> None:
        super().__init__(f"шрифт {font_path} не укладывается в предел растра на тексте {text[:32]!r}")
        self.font_path = font_path


class TextLineRenderer:
    """Рисует строку по стилю; умеет померить её ширину без растеризации."""

    def measure_width(self, text: str, style: LineStyle) -> float:
        font = load_truetype_font(str(style.font.path), style.font_size)
        if not style.has_letter_spacing:
            return float(font.getlength(text))
        spacing = style.letter_spacing_ratio * style.font_size
        return float(sum(font.getlength(character) + spacing for character in text))

    def measure_raster(self, text: str, style: LineStyle) -> tuple[int, int]:
        """Размер растра, который PIL выделит под маску этой строки.

        Отличается от `measure_width`: та возвращает сумму advance-ширин, а выделяется растр по
        настоящим границам глифов вместе с обводкой. У большинства шрифтов это одно и то же, но
        обрезать строку надо именно по этой величине — advance ничего не гарантирует.
        """
        font = load_truetype_font(str(style.font.path), style.font_size)
        left, top, right, bottom = font.getbbox(text, stroke_width=style.stroke_width)
        return max(right - left, 0), max(bottom - top, 0)

    def render(self, text: str, style: LineStyle) -> GlyphLayer:
        layer = self._draw(self._truncate_to_layer(text, style), style)
        layer = self._apply_skew(layer, style)
        ink_box = layer.getchannel("A").getbbox()
        if ink_box is None:
            raise ValueError(f"text rendered no visible ink: {text[:32]!r}")
        return GlyphLayer(image=layer, ink_box=ink_box)

    def _truncate_to_layer(self, text: str, style: LineStyle) -> str:
        """Возвращает самый длинный префикс, растр под который влезает в предел.

        PIL рендерит строку целиком и только потом она обрезается размером слоя, поэтому на
        длинном тексте растр раздувается до десятков тысяч пикселей: это и лишняя работа, и
        падение с `OSError: array allocation size too large`.

        Поиск именно двоичный, а не пропорциональное ужатие. Ужатие исходит из того, что растр
        растёт линейно по длине, а это неверно: высота растра зависит от того, какие глифы
        попали в префикс, и на части декоративных шрифтов приближение не сходилось за
        отведённые проходы — функция возвращала строку сверх лимита, и PIL всё равно падал.
        Размер растра по длине префикса монотонен, поэтому двоичный поиск даёт точный ответ
        за полтора десятка измерений.
        """
        if not self._fits("" if not text else text[:1], style):
            raise FontRenderError(style.font.path, text)
        low, high = 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self._fits(text[:middle], style):
                low = middle
            else:
                high = middle - 1
        return text[:low]

    def _fits(self, text: str, style: LineStyle) -> bool:
        width, height = self.measure_raster(text, style)
        return width <= MAX_LAYER_WIDTH - 2 * LAYER_MARGIN and width * max(height, 1) <= MAX_LAYER_PIXELS

    def _draw(self, text: str, style: LineStyle) -> Image.Image:
        font = load_truetype_font(str(style.font.path), style.font_size)
        width = min(int(self.measure_width(text, style)) + 2 * LAYER_MARGIN, MAX_LAYER_WIDTH)
        height = self._layer_height(font, style)
        layer = Image.new("RGBA", (width, height), TRANSPARENT)
        draw = ImageDraw.Draw(layer)
        fill = style.color_scheme.foreground + (255,)
        stroke_fill = None if style.color_scheme.stroke is None else style.color_scheme.stroke + (255,)
        if style.has_letter_spacing:
            self._draw_spaced(draw, text, font, style, fill, stroke_fill)
            return layer
        # Без заданного интервала рисуем всю строку одним вызовом: так сохраняется кернинг,
        # который при посимвольной отрисовке теряется.
        try:
            draw.text(
                (LAYER_MARGIN, LAYER_MARGIN),
                text,
                font=font,
                fill=fill,
                stroke_width=style.stroke_width,
                stroke_fill=stroke_fill,
            )
        except OSError as error:
            # Обрезка уже гарантировала, что заявленный размер растра влезает в предел. Если
            # PIL всё-таки не смог, значит шрифт сообщает о себе неправду, и важно узнать какой:
            # без имени виновника поиск занимал отдельный прогон по всей выборке.
            raise FontRenderError(style.font.path, text) from error
        return layer

    def _layer_height(self, font, style: LineStyle) -> int:
        ascent, descent = font.getmetrics()
        return int((ascent + descent) * LAYER_HEIGHT_SAFETY) + 2 * (LAYER_MARGIN + style.stroke_width)

    def _draw_spaced(self, draw, text, font, style, fill, stroke_fill) -> None:
        spacing = style.letter_spacing_ratio * style.font_size
        offset = float(LAYER_MARGIN)
        for character in text:
            draw.text(
                (offset, LAYER_MARGIN),
                character,
                font=font,
                fill=fill,
                stroke_width=style.stroke_width,
                stroke_fill=stroke_fill,
            )
            offset += font.getlength(character) + spacing

    def _apply_skew(self, layer: Image.Image, style: LineStyle) -> Image.Image:
        if not style.has_skew:
            return layer
        # Фальшивый курсив. PIL задаёт обратное отображение (из точки результата в точку
        # источника), поэтому коэффициент при y читает пиксели левее по мере подъёма — верх
        # строки визуально уезжает вправо. Третий член возвращает низ строки на место,
        # а слой расширяется, чтобы наклонённые края не срезались.
        shear = math.tan(math.radians(style.skew_degrees))
        width = layer.width + int(abs(shear) * layer.height) + 1
        matrix = (1.0, shear, -shear * layer.height if shear > 0 else 0.0, 0.0, 1.0, 0.0)
        return layer.transform((width, layer.height), Image.AFFINE, matrix, resample=Image.BICUBIC)


@dataclass(frozen=True)
class LineStyleSampler:
    """Разыгрывает стиль строки для заданной письменности."""

    registry: FontRegistry
    color_sampler: ColorSchemeSampler = ColorSchemeSampler()
    letter_spacing: ValueRange = ValueRange(-0.02, 0.25)
    letter_spacing_probability: float = 0.25
    # Наклон только вперёд, и это принципиально. Курсив в реальности всегда завален вправо;
    # при повороте кропа на 180 градусов он заваливается влево, то есть направление наклона
    # само по себе говорит об ориентации. Симметричный разброс, который стоял здесь раньше,
    # уничтожал этот признак: модель училась, что наклон неинформативен.
    skew_degrees: ValueRange = ValueRange(3.0, 22.0)
    skew_probability: float = 0.20
    stroke_width: ValueRange = ValueRange(1, 3)
    stroke_probability: float = 0.15

    def sample(self, rng: np.random.Generator, script: Script) -> LineStyle:
        scheme = self.color_sampler.sample(rng)
        return LineStyle(
            font=self.registry.sample(rng, script),
            font_size=NOMINAL_FONT_SIZE,
            color_scheme=scheme,
            letter_spacing_ratio=self._sample_spacing(rng),
            stroke_width=self._sample_stroke_width(rng, scheme),
            skew_degrees=self._sample_skew(rng),
        )

    def _sample_spacing(self, rng: np.random.Generator) -> float:
        return self.letter_spacing.sample(rng) if happens(rng, self.letter_spacing_probability) else 0.0

    def _sample_skew(self, rng: np.random.Generator) -> float:
        return self.skew_degrees.sample(rng) if happens(rng, self.skew_probability) else 0.0

    def _sample_stroke_width(self, rng: np.random.Generator, scheme: ColorScheme) -> int:
        if scheme.stroke is None or not happens(rng, self.stroke_probability):
            return 0
        return self.stroke_width.sample_int(rng)


@dataclass(frozen=True)
class LetterCaseSampler:
    """Меняет регистр строки: UPPERCASE, Title или как есть.

    Доля заглавных поднята намеренно. Заглавные буквы почти лишены выносных элементов, и именно
    на них модель чаще всего оказывается неуверенной: в кропах вроде РЕКЛАМЫ или MENS/HOMMES все
    буквы одной высоты, и признаков ориентации остаётся мало. Вывесок в тестовой выборке много,
    поэтому такой текст стоит показывать чаще, чтобы модель научилась вытягивать то немногое,
    что там есть: форму отдельных глифов, знаки препинания, асимметрию отступов.
    """

    uppercase_probability: float = 0.32
    capitalize_probability: float = 0.15

    def apply(self, text: str, rng: np.random.Generator) -> str:
        roll = rng.random()
        if roll < self.uppercase_probability:
            return text.upper()
        if roll < self.uppercase_probability + self.capitalize_probability:
            return text.title()
        return text
