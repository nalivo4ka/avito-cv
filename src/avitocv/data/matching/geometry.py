"""Сопоставление геометрии кропов с тестовой: одна сетка на взвешивание и на пересэмплинг.

Геометрия здесь — пара «высота кропа, его пропорции». Обе величины нужны вместе, и это выяснилось
дорого. Пока сходилась одна высота, валидация отдавала диапазону пропорций до 2.5 тридцать
процентов веса, тогда как в тесте его доля 7%. Именно там модель слабее всего, так что метрика
занижала себя: по валидации выходило 0.9242, на тесте — 0.9473, и почти весь разрыв объяснялся
этим перекосом, а не разницей доменов.

Взвешивание и пересэмплинг решают одну задачу и отличаются только тем, что делают с отношением
плотностей: метрика домножает на него веса, обучающая выборка вытягивает по нему индексы. Поэтому
сетка и отношение живут здесь, а не дублируются в двух местах.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avitocv.data.matching.profile import GeometryReference

MIN_GEOMETRY_HEIGHT = 8.0
MIN_GEOMETRY_ASPECT = 1.05
DEFAULT_BIN_COUNT = 10
DEFAULT_REFERENCE_SAMPLE_SIZE = 200_000
DEFAULT_MAX_REPEATS = 10
CLIPPING_PASSES = 6


@dataclass(frozen=True)
class CropGeometry:
    """Геометрия набора кропов в том виде, в каком её сравнивают с эталонной."""

    heights: np.ndarray
    aspects: np.ndarray

    @classmethod
    def of(cls, heights, aspects) -> "CropGeometry":
        return cls(np.asarray(heights, dtype=float), np.asarray(aspects, dtype=float))

    @classmethod
    def sampled_from(cls, reference: GeometryReference, rng: np.random.Generator, count: int) -> "CropGeometry":
        return cls.of(reference.crop_height.sample(rng, count), reference.aspect_ratio.sample(rng, count))

    def __len__(self) -> int:
        return len(self.heights)


class LogGeometryGrid:
    """Двумерная логарифмическая сетка по высоте и пропорциям.

    Логарифм потому, что оба распределения тяжелохвостые: в линейной шкале почти всё попало бы
    в первые корзины и сетка перестала бы что-либо различать.
    """

    def __init__(self, height_edges: np.ndarray, aspect_edges: np.ndarray) -> None:
        self._height_edges = height_edges
        self._aspect_edges = aspect_edges

    @classmethod
    def spanning(cls, parts: list[CropGeometry], bin_count: int = DEFAULT_BIN_COUNT) -> "LogGeometryGrid":
        """Сетка накрывает объединение диапазонов, а не только эталонный.

        Если строить её по одному эталону, всё выходящее за его пределы прижимается к крайней
        корзине и наследует её плотность — то есть кроп, которого в эталоне быть не может,
        считается допустимым. При объединённой сетке плотность эталона в таких ячейках нулевая,
        и кроп честно получает нулевой вес.
        """
        heights = np.concatenate([_logarithm(part.heights, MIN_GEOMETRY_HEIGHT) for part in parts])
        aspects = np.concatenate([_logarithm(part.aspects, MIN_GEOMETRY_ASPECT) for part in parts])
        return cls(np.histogram_bin_edges(heights, bins=bin_count),
                   np.histogram_bin_edges(aspects, bins=bin_count))

    def density(self, geometry: CropGeometry) -> np.ndarray:
        counts, _, _ = np.histogram2d(
            _logarithm(geometry.heights, MIN_GEOMETRY_HEIGHT),
            _logarithm(geometry.aspects, MIN_GEOMETRY_ASPECT),
            bins=[self._height_edges, self._aspect_edges],
        )
        return counts / max(counts.sum(), 1.0)

    def cells(self, geometry: CropGeometry) -> tuple[np.ndarray, np.ndarray]:
        return (
            _bin_of(geometry.heights, self._height_edges, MIN_GEOMETRY_HEIGHT),
            _bin_of(geometry.aspects, self._aspect_edges, MIN_GEOMETRY_ASPECT),
        )


class GeometryDensityRatio:
    """Отношение эталонной плотности к фактической в ячейке каждого кропа.

    Это importance-отношение: кроп из ячейки, которая в тесте встречается вдвое чаще, чем у нас,
    получает двойку. Ноль означает, что в тесте таких кропов не бывает вовсе.
    """

    def __init__(self, bin_count: int = DEFAULT_BIN_COUNT,
                 reference_sample_size: int = DEFAULT_REFERENCE_SAMPLE_SIZE, seed: int = 0) -> None:
        self._bin_count = bin_count
        self._reference_sample_size = reference_sample_size
        self._seed = seed

    def compute(self, geometry: CropGeometry, reference: GeometryReference) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        target = CropGeometry.sampled_from(reference, rng, self._reference_sample_size)
        grid = LogGeometryGrid.spanning([target, geometry], self._bin_count)
        target_density = grid.density(target)
        actual_density = grid.density(geometry)
        cells = grid.cells(geometry)
        ratio = np.where(
            actual_density[cells] > 0.0,
            target_density[cells] / np.maximum(actual_density[cells], 1e-12),
            0.0,
        )
        if ratio.sum() <= 0.0:
            raise ValueError("распределения геометрии не пересекаются")
        return ratio


class JointGeometryWeighter:
    """Importance-веса, приводящие геометрию манифеста к тестовой.

    Подвыборка с точным совпадением оставила бы горстку кропов — её размер упирается в самую
    редкую ячейку сетки. Взвешивание сохраняет все кропы, платя за это падением эффективного
    размера, который и печатается рядом с метрикой.
    """

    def __init__(self, ratio: GeometryDensityRatio | None = None) -> None:
        self._ratio = ratio or GeometryDensityRatio()

    def weights_for(self, geometry: CropGeometry, reference: GeometryReference) -> np.ndarray:
        raw = self._ratio.compute(geometry, reference)
        return raw / raw.sum() * len(raw)

    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        return float(weights.sum() ** 2 / np.sum(weights ** 2))


def _logarithm(values: np.ndarray, floor: float) -> np.ndarray:
    return np.log(np.clip(np.asarray(values, dtype=float), floor, None))


def _bin_of(values: np.ndarray, edges: np.ndarray, floor: float) -> np.ndarray:
    return np.clip(np.digitize(_logarithm(values, floor), edges) - 1, 0, len(edges) - 2)


class TruncatedImportanceSampler:
    """Вероятности выбора с потолком на число повторов одного кропа.

    Точное выравнивание геометрии платит тем, чего у нас мало, — независимыми снимками. В
    редких ячейках сетки отношение плотностей доходит до тысяч, и весь вес садится на горстку
    кропов: у словарного TextOCR 782 тысячи кропов вырождались в эффективные 7 тысяч. Кроп,
    вытянутый четыре тысячи раз, новой информации не несёт, как его ни аугментируй.

    Поэтому вероятность обрезается так, чтобы ожидаемое число повторов не превышало потолок, а
    высвободившаяся масса раздаётся тем, кто до потолка не дотянул. Это усечённое
    importance-семплирование (Ionides, 2008): смещение в обмен на дисперсию. Цена измерена —
    при потолке 10 расхождение с тестовым распределением пропорций растёт с 0.04 до 0.08 по
    полной вариации, а эффективный размер TextOCR поднимается с 7 до 100 тысяч.

    Несколько проходов нужны потому, что раздача высвободившейся массы сама может вытолкнуть
    соседей за потолок.
    """

    def __init__(self, max_repeats: int = DEFAULT_MAX_REPEATS, passes: int = CLIPPING_PASSES) -> None:
        self._max_repeats = max_repeats
        self._passes = passes

    def probabilities(self, ratio: np.ndarray, draw_count: int) -> np.ndarray:
        probability = ratio / ratio.sum()
        if self._max_repeats <= 0:
            return probability
        cap = self._max_repeats / draw_count
        for _ in range(self._passes):
            excess = probability > cap
            if not excess.any():
                break
            released = float((probability[excess] - cap).sum())
            probability = np.where(excess, cap, probability)
            headroom = (cap - probability)[~excess]
            if headroom.sum() <= 0.0:
                break
            probability[~excess] += headroom / headroom.sum() * min(released, float(headroom.sum()))
        return probability / probability.sum()
