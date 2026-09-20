"""Поставщики **ровных** кропов — вход для `OrientationDataset`.

Контракт `TextLineSource` один: выдать кроп, про который точно известно, что он не перевёрнут.
Откуда он взялся, неважно, поэтому реализации взаимозаменяемы:

    SyntheticTextLineSource     генерирует кроп с нуля
    MaterializedTextLineSource  читает заранее сгенерированный кроп из шардов
    ManifestTextLineSource      вырезает бокс из реальной фотографии по манифесту
    MixedTextLineSource         взвешенно смешивает любые из перечисленных

Поворот и метка живут не здесь, а в `datasets.OrientationDataset`, — так один механизм
разметки работает для любого источника.
"""

from __future__ import annotations

import math

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from avitocv.data.synthesis.composition import CropComposer, CropLayout, CropLayoutSampler, FittedLineRenderer
from avitocv.data.synthesis.composition import CropRequest
from avitocv.data.synthesis.corpus import TextSource
from avitocv.data.degradation import DegradationPipeline, ResizeToHeight
from avitocv.data.matching.geometry import (
    DEFAULT_BIN_COUNT,
    DEFAULT_MAX_REPEATS,
    DEFAULT_REFERENCE_SAMPLE_SIZE,
    CropGeometry,
    GeometryDensityRatio,
    TruncatedImportanceSampler,
)
from avitocv.data.manifest import CropManifest, CropRecord
from avitocv.data.matching.profile import CropProfile, GeometryReference
from avitocv.data.synthesis.rendering import FontRenderError, GlyphLayer, LineStyle, LineStyleSampler
from avitocv.data.sampling import ValueRange, happens, weighted_choice
from avitocv.data.storage import CropIndex, CropShardReader, decode_crop
from avitocv.data.synthesis.writing_systems import Script

MAX_FONT_ATTEMPTS = 8  # замен шрифта на один кроп, прежде чем признать выборку безнадёжной
MIN_CROP_HEIGHT = 10
MIN_CROP_ASPECT = 1.1
MIN_CANVAS_TO_INK_RATIO = 0.6
# Грубая оценка высоты чернил относительно кегля. Точность не важна: кроп всё равно
# приводится к целевой высоте финальным ресайзом, а оценка нужна лишь чтобы выбрать кегль
# в правильном порядке величины.
INK_TO_FONT_RATIO = 0.8
NEIGHBOUR_SIZE_RATIO = ValueRange(0.7, 1.15)
# Потолок удлинения строки ради компенсации наклона: дальше растёт только стоимость отрисовки.
MAX_TILT_COMPENSATION = 1.5
# Бокс детектора никогда не обтягивает строку идеально, поэтому реальные кропы тоже
# берутся с небольшим случайным запасом вокруг размеченного прямоугольника.
DEFAULT_MANIFEST_JITTER = ValueRange(0.0, 0.12)
# Сколько распакованных снимков держать: записи идут подряд по снимкам, так что хватает пары.
DECODED_PHOTO_CACHE_SIZE = 4
# Нижняя граница для логарифмической шкалы при выравнивании высот.


class TextLineSource(ABC):
    """Поставщик заведомо ровных кропов с текстом."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError

    def weight_of(self, index: int) -> float:
        """Вклад кропа в метрику. Переопределяется там, где выборка смещена относительно теста."""
        return 1.0


@dataclass(frozen=True)
class ScriptedTextSource:
    """Источник текста вместе с письменностью, которой он написан, и его весом."""

    text_source: TextSource
    script: Script
    weight: float


@dataclass(frozen=True)
class SyntheticSourceConfig:
    """Числовые настройки генерации: длина эпохи, суперсэмплинг, пределы кегля."""

    virtual_length: int = 1_000_000
    supersample: ValueRange = ValueRange(1.15, 2.0)
    font_size_limits: ValueRange = ValueRange(14, 140)
    neighbour_font_change_probability: float = 0.4


@dataclass(frozen=True)
class SyntheticComponents:
    """Собранные зависимости генератора; передаются одним объектом."""

    text_sources: tuple[ScriptedTextSource, ...]
    style_sampler: LineStyleSampler
    line_renderer: FittedLineRenderer
    layout_sampler: CropLayoutSampler
    composer: CropComposer
    profile: CropProfile
    capture_stage: DegradationPipeline


class SyntheticTextLineSource(TextLineSource):
    """Генерирует ровный кроп с нуля: текст, стиль, компоновка, съёмочные деградации."""

    def __init__(self, components: SyntheticComponents, config: SyntheticSourceConfig) -> None:
        self._components = components
        self._config = config
        self._resizer = ResizeToHeight()
        self.fallback_count = 0

    def __len__(self) -> int:
        return self._config.virtual_length

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        """Собирает кроп; неудачный шрифт заменяет другим, а не роняет весь прогон.

        Среди полутора тысяч шрифтов попадаются такие, где PIL сообщает один размер растра, а
        выделяет другой, и падает с `OSError` на конкретном сочетании шрифта, кегля и текста.
        Поймать их заранее не получается: обе пробы — поглифовая и построчная — их пропускают.
        А падение дорогое: генерация идёт около часа, и одна такая пара обнуляла всю работу.

        Повтор берёт новый стиль из того же `rng`, то есть остаётся детерминированным и не
        выходит за пределы собственного распределения генератора. Сколько раз это понадобилось,
        видно в `fallback_count` — если счётчик заметный, значит проблема не в единичном шрифте
        и её надо разбирать, а не заметать.
        """
        target_height = self._sample_height(rng)
        target_aspect = self._sample_aspect(rng)
        layout = self._components.layout_sampler.sample(rng, target_aspect)
        for _ in range(MAX_FONT_ATTEMPTS):
            scripted = self._pick_text_source(rng)
            style = self._make_style(rng, scripted.script, layout, target_height)
            try:
                request = self._build_request(rng, scripted, style, layout, target_aspect)
                canvas = self._components.composer.compose(request, rng)
            except FontRenderError:
                self.fallback_count += 1
                continue
            return self._finalize(canvas, target_height, rng)
        raise FontRenderError(style.font.path, f"индекс {index}")

    def _sample_height(self, rng: np.random.Generator) -> int:
        return max(MIN_CROP_HEIGHT, int(round(float(self._components.profile.crop_height.sample(rng)))))

    def _sample_aspect(self, rng: np.random.Generator) -> float:
        return max(MIN_CROP_ASPECT, float(self._components.profile.aspect_ratio.sample(rng)))

    def _pick_text_source(self, rng: np.random.Generator) -> ScriptedTextSource:
        sources = self._components.text_sources
        return weighted_choice(rng, sources, [item.weight for item in sources])

    def _make_style(
        self,
        rng: np.random.Generator,
        script: Script,
        layout: CropLayout,
        target_height: int,
    ) -> LineStyle:
        style = self._components.style_sampler.sample(rng, script)
        # Кегль выводится из целевой высоты кропа, а не берётся фиксированным. Тогда финальный
        # ресайз всегда идёт вниз, и мелкий текст получает сглаживание от даунскейла — как
        # в реальных кропах, а не рисуется сразу в 12 пикселей с рваными краями.
        ink_height = target_height / self._canvas_to_ink_ratio(layout)
        raw_size = ink_height * self._config.supersample.sample(rng) / INK_TO_FONT_RATIO
        clamped = int(np.clip(raw_size, self._config.font_size_limits.low, self._config.font_size_limits.high))
        return replace(style, font_size=clamped)

    @staticmethod
    def _tilt_compensation(tilt_degrees: float, target_aspect: float) -> float:
        """Во сколько раз длиннее должна быть строка, чтобы после наклона получить нужный aspect.

        Наклон раздувает высоту бокса сильнее, чем ширину, поэтому пропорции падают: без
        компенсации медиана aspect выходила 4.25 вместо 4.89 у теста. Из соотношения сторон
        описанного прямоугольника нужный коэффициент выражается точно.

        При большом наклоне решения не существует: у строки, наклонённой сильнее чем arctan(1/A),
        бокс физически не может быть настолько широким. В этом случае берём потолок — такие
        кропы и в реальных данных получаются более квадратными.
        """
        radians = abs(math.radians(tilt_degrees))
        cosine, sine = math.cos(radians), math.sin(radians)
        denominator = cosine - target_aspect * sine
        if denominator <= 0.0:
            return MAX_TILT_COMPENSATION
        needed = (target_aspect * cosine - sine) / denominator
        return float(np.clip(needed / target_aspect, 1.0, MAX_TILT_COMPENSATION))

    def _canvas_to_ink_ratio(self, layout: CropLayout) -> float:
        # Клампим снизу, потому что отступы бывают отрицательными: без ограничения отношение
        # ушло бы к нулю и раздуло кегль до предела.
        return max(MIN_CANVAS_TO_INK_RATIO, 1.0 + layout.pad_top_ratio + layout.pad_bottom_ratio)

    def _build_request(
        self,
        rng: np.random.Generator,
        scripted: ScriptedTextSource,
        style: LineStyle,
        layout: CropLayout,
        target_aspect: float,
    ) -> CropRequest:
        # Строка должна быть не уже окна обрезки; fill_ratio < 1 оставляет часть кропов
        # с фоновыми полями по краям, как у бокса вокруг короткой надписи.
        min_ink_aspect = (
            target_aspect
            * self._canvas_to_ink_ratio(layout)
            * layout.fill_ratio
            * self._tilt_compensation(layout.tilt_degrees, target_aspect)
        )
        main = self._components.line_renderer.render_at_least(scripted.text_source, style, min_ink_aspect, rng)
        return CropRequest(
            main=main,
            above=self._render_neighbour(rng, scripted, style, min_ink_aspect, layout.has_neighbour_above),
            below=self._render_neighbour(rng, scripted, style, min_ink_aspect, layout.has_neighbour_below),
            layout=layout,
            color_scheme=style.color_scheme,
            target_aspect=target_aspect,
        )

    def _render_neighbour(
        self,
        rng: np.random.Generator,
        scripted: ScriptedTextSource,
        style: LineStyle,
        min_ink_aspect: float,
        is_present: bool,
    ) -> GlyphLayer | None:
        if not is_present:
            return None
        neighbour_style = self._vary_style(rng, style, scripted.script)
        return self._components.line_renderer.render_at_least(
            scripted.text_source, neighbour_style, min_ink_aspect, rng
        )

    def _vary_style(self, rng: np.random.Generator, style: LineStyle, script: Script) -> LineStyle:
        size = max(8, int(style.font_size * NEIGHBOUR_SIZE_RATIO.sample(rng)))
        if not happens(rng, self._config.neighbour_font_change_probability):
            return replace(style, font_size=size)
        font = self._components.style_sampler.registry.sample(rng, script)
        return replace(style, font_size=size, font=font)

    def _finalize(self, canvas: Image.Image, target_height: int, rng: np.random.Generator) -> Image.Image:
        pixels = np.asarray(canvas, dtype=np.uint8)
        degraded = self._components.capture_stage.apply(pixels, rng)
        resized = self._resizer.apply_to(degraded, target_height, rng)
        return Image.fromarray(resized, mode="RGB")


class ManifestTextLineSource(TextLineSource):
    """Вырезает боксы из реальных фотографий по манифесту.

    Снимок декодируется один раз на несколько кропов. Без этого валидация была неприлично
    медленной: в одной фотографии HierText размечено в среднем 25 строк, а PIL декодирует
    целое изображение даже когда нужен маленький прямоугольник из него, — то есть мегапиксельный
    снимок разжимался по два десятка раз подряд. Записи манифеста идут подряд по снимкам,
    поэтому кэша на несколько последних хватает.
    """

    def __init__(
        self,
        records: Sequence[CropRecord],
        root: Path,
        padding_jitter: ValueRange,
        cache_size: int = DECODED_PHOTO_CACHE_SIZE,
    ) -> None:
        if not records:
            raise ValueError("manifest source requires at least one record")
        # Записи хранятся колонками, а не кортежем датаклассов. При 780 тысячах кропов TextOCR
        # кортеж занимал сотни мегабайт и, главное, пиклился в каждый воркер целиком — загрузчик
        # падал с MemoryError ещё до первого кропа. Пути повторяются (25 тысяч снимков на 780
        # тысяч кропов), поэтому они лежат таблицей, а в колонке только её индексы.
        self._paths, codes = self._encode_paths(records)
        self._path_codes = codes
        self._boxes = np.array([[record.left, record.top, record.width, record.height]
                                for record in records], dtype=np.int32)
        self._weights = np.array([record.weight for record in records], dtype=np.float32)
        self._root = Path(root)
        self._padding_jitter = padding_jitter
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()

    @staticmethod
    def _encode_paths(records: Sequence[CropRecord]) -> tuple[tuple[str, ...], np.ndarray]:
        lookup: dict[str, int] = {}
        codes = np.empty(len(records), dtype=np.int32)
        for position, record in enumerate(records):
            codes[position] = lookup.setdefault(record.image_path, len(lookup))
        return tuple(lookup), codes

    @classmethod
    def from_manifest(
        cls,
        manifest: CropManifest,
        root: Path,
        padding_jitter: ValueRange = DEFAULT_MANIFEST_JITTER,
    ) -> "ManifestTextLineSource":
        return cls(manifest.records(), root, padding_jitter)

    def __len__(self) -> int:
        return len(self._path_codes)

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        position = index % len(self)
        photo = self._photo(self._paths[self._path_codes[position]])
        return photo.crop(self._jittered_box(self._boxes[position], photo.size, rng))

    def weight_of(self, index: int) -> float:
        return float(self._weights[index % len(self)])

    def _photo(self, relative_path: str) -> Image.Image:
        cached = self._cache.get(relative_path)
        if cached is not None:
            self._cache.move_to_end(relative_path)
            return cached
        with Image.open(self._root / relative_path) as image:
            decoded = image.convert("RGB").copy()
        self._cache[relative_path] = decoded
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return decoded

    def _jittered_box(
        self,
        box: np.ndarray,
        image_size: tuple[int, int],
        rng: np.random.Generator,
    ) -> tuple[int, int, int, int]:
        # Расширенный бокс обрезается границами картинки: PIL.crop за пределами изображения
        # заливает выход нулями, и у кропов у края фотографии появилась бы чёрная рамка,
        # которой не бывает в настоящих боксах детектора.
        box_left, box_top, box_width, box_height = (int(value) for value in box)
        margin = self._padding_jitter.sample(rng) * box_height
        width, height = image_size
        left, top = box_left, box_top
        right, bottom = box_left + box_width, box_top + box_height
        return (
            max(0, int(left - margin)),
            max(0, int(top - margin)),
            min(width, int(right + margin)),
            min(height, int(bottom + margin)),
        )

    def __getstate__(self) -> dict:
        # Декодированные снимки не пиклятся в воркеры: каждый наполнит свой кэш сам.
        return {key: value for key, value in self.__dict__.items() if key != "_cache"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache = OrderedDict()


class MixedTextLineSource(TextLineSource):
    """Взвешенно смешивает несколько источников кропов."""

    def __init__(self, sources: Sequence[TextLineSource], weights: Sequence[float], length: int) -> None:
        if not sources:
            raise ValueError("at least one line source is required")
        self._sources = tuple(sources)
        self._weights = tuple(float(weight) for weight in weights)
        self._length = length

    def __len__(self) -> int:
        return self._length

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        source = weighted_choice(rng, self._sources, self._weights)
        return source.load_upright(int(rng.integers(len(source))), rng)


class MaterializedTextLineSource(TextLineSource):
    """Читает заранее сгенерированные кропы из шардового хранилища."""

    def __init__(self, directory: Path, length: int | None = None) -> None:
        self._directory = Path(directory)
        self._index = CropIndex.load(self._directory)
        self._reader = CropShardReader(self._directory)
        if len(self._index) == 0:
            raise ValueError(f"хранилище кропов пусто: {directory}")
        # Ограничение длины нужно для коротких прогонов: сами кропы остаются доступны все,
        # просто эпоха заканчивается раньше.
        self._length = min(length, len(self._index)) if length else len(self._index)

    def __len__(self) -> int:
        return self._length

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        return decode_crop(self._reader.read(self._index.location_at(index % len(self._index))))


class GeometryMatchedResampler:
    """Строит таблицу индексов, выравнивающую геометрию обучающей выборки под тестовую.

    Выравнивается пара величин сразу: высота кропа и его пропорции. По отдельности этого мало,
    и вторая половина выяснилась дорого. Выравнивание одной высоты дало прибавку, но усилило
    перекос по пропорциям: в HierText 23% строк короче aspect 2.5 против 7% в тесте, а после
    подгонки высот доля выросла до 33%. Короткий кроп это мало символов, то есть мало
    независимых свидетельств об ориентации, и именно там модель слабее всего — 0.88 против
    0.98 на длинных строках. Треть обучения уходила в режим, которого в тесте почти нет.

    Выравнивание идёт повторами, а не отсевом: набор без повторов, точно совпадающий с эталоном,
    ограничен самой редкой ячейкой сетки. Насколько дорого обходятся повторы, показывает
    `effective_sample_size`: у HierText это 31 тысяча из 300 тысяч вытянутых, у словарного
    TextOCR — всего 2.5 тысячи, потому что длинных строк там почти нет.
    """

    def __init__(self, bin_count: int = DEFAULT_BIN_COUNT,
                 reference_sample_size: int = DEFAULT_REFERENCE_SAMPLE_SIZE, seed: int = 0,
                 max_repeats: int = DEFAULT_MAX_REPEATS) -> None:
        self._ratio = GeometryDensityRatio(bin_count, reference_sample_size, seed)
        self._sampler = TruncatedImportanceSampler(max_repeats)
        self._seed = seed

    def build(
        self,
        heights: np.ndarray,
        aspects: np.ndarray,
        reference: GeometryReference,
        count: int,
        table_size: int | None = None,
    ) -> np.ndarray:
        """Таблица индексов: `count` — спрос одной эпохи, `table_size` — длина таблицы.

        Разделять их обязательно. Таблица ровно на одну эпоху означает, что обучение каждую
        эпоху видит один и тот же набор: сдвиг окна в `CombinedOrientationDataset` берётся по
        модулю длины набора, а она тогда равна спросу, и сдвиг тождественно нулевой. Из 990
        тысяч реальных кропов в обучение попадало около 180 тысяч, остальные не участвовали.

        Потолок повторов при этом считается от спроса эпохи, а не от длины таблицы: смысл у
        него «сколько раз кроп покажут за эпоху», и от того, на сколько эпох вперёд построена
        таблица, он зависеть не должен.
        """
        geometry = CropGeometry.of(heights, aspects)
        ratio = self._ratio.compute(geometry, reference)
        probability = self._sampler.probabilities(ratio, count)
        rng = np.random.default_rng(self._seed)
        return rng.choice(len(geometry), size=table_size or count, replace=True, p=probability)

    @staticmethod
    def effective_sample_size(indices: np.ndarray) -> float:
        counts = np.bincount(indices)
        counts = counts[counts > 0]
        return float(counts.sum() ** 2 / np.sum(counts ** 2))


class ResampledTextLineSource(TextLineSource):
    """Источник, читающий кропы через таблицу индексов.

    Сама таблица решает, что и сколько раз попадёт в эпоху; источник об этом ничего не знает.
    Соответствие «индекс — кроп» остаётся постоянным, поэтому единственным источником
    случайности порядка остаётся перемешивание в загрузчике.
    """

    def __init__(self, source: TextLineSource, indices: np.ndarray) -> None:
        if len(indices) == 0:
            raise ValueError("таблица индексов пуста")
        self._source = source
        # int32 вместо int64: индекс кропа не превышает миллионов, а таблица на весь прогон
        # это миллионы элементов в каждом воркере.
        self._indices = np.asarray(indices, dtype=np.int32)

    def __len__(self) -> int:
        return len(self._indices)

    def load_upright(self, index: int, rng: np.random.Generator) -> Image.Image:
        return self._source.load_upright(int(self._indices[index % len(self._indices)]), rng)

    def weight_of(self, index: int) -> float:
        return self._source.weight_of(int(self._indices[index % len(self._indices)]))
