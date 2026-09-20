"""Деградации изображения, разложенные на две стадии относительно поворота.

`build_capture_stage` — то, что происходит при съёмке и до сжатия: перспектива, размытие,
яркость. Все эти операции симметричны относительно поворота на 180°, поэтому применяются
к ещё ровному кропу.

`build_codec_stage` — то, что делает кодек: джиттер разрешения, шум, JPEG. JPEG режет картинку
на блоки 8x8 от левого верхнего угла, поэтому у повёрнутого кропа сетка блоков легла бы иначе,
чем у ровного. Эта стадия обязана идти строго **после** поворота, иначе метку можно было бы
угадать по артефактам сжатия, а не по тексту.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from avitocv.data.sampling import ValueRange, happens

MAX_CHANNEL_VALUE = 255
MIN_SIDE = 4
INTERPOLATIONS = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST)
# Запасное значение на случай вызова без параметра: вдвое выше входа 32 px. Рабочие стадии
# получают высоту от `OrientationDatasetAssembler`, и при входе 48 она равна 96.
DEFAULT_WORK_HEIGHT = 64


class ImageDegradation(ABC):
    """Одно преобразование изображения."""

    @abstractmethod
    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianBlurDegradation(ImageDegradation):
    """Расфокус."""

    sigma: ValueRange = ValueRange(0.4, 2.2)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=self.sigma.sample(rng))


@dataclass(frozen=True)
class MotionBlurDegradation(ImageDegradation):
    """Смаз от движения камеры под случайным углом."""

    length: ValueRange = ValueRange(3, 11)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Ядро — отрезок, повёрнутый на случайный угол: так смаз получает произвольное
        # направление. `| 1` делает размер нечётным, иначе у ядра нет центрального пикселя.
        size = self.length.sample_int(rng) | 1
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0 / size
        rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        rotated = cv2.warpAffine(kernel, rotation, (size, size))
        total = rotated.sum()
        return cv2.filter2D(image, -1, rotated / total if total > 0 else kernel)


@dataclass(frozen=True)
class BrightnessContrastDegradation(ImageDegradation):
    """Сдвиг яркости и сжатие контраста."""

    contrast: ValueRange = ValueRange(0.80, 1.40)
    brightness: ValueRange = ValueRange(-28.0, 28.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        adjusted = image.astype(np.float32) * self.contrast.sample(rng) + self.brightness.sample(rng)
        return np.clip(adjusted, 0, MAX_CHANNEL_VALUE).astype(np.uint8)


@dataclass(frozen=True)
class PerspectiveDegradation(ImageDegradation):
    """Лёгкая перспектива от съёмки под углом."""

    corner_shift: ValueRange = ValueRange(0.0, 0.06)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        offsets = rng.uniform(-1.0, 1.0, size=(4, 2)) * self.corner_shift.sample(rng) * height
        matrix = cv2.getPerspectiveTransform(source, (source + offsets).astype(np.float32))
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class GaussianNoiseDegradation(ImageDegradation):
    """Сенсорный шум."""

    sigma: ValueRange = ValueRange(1.0, 14.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # float32, а не умолчательный float64: на шум уходит вдвое меньше памяти, а разницы
        # для восьмибитной картинки никакой.
        noise = rng.normal(0.0, self.sigma.sample(rng), size=image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, MAX_CHANNEL_VALUE).astype(np.uint8)


@dataclass(frozen=True)
class JpegDegradation(ImageDegradation):
    """Повторное сжатие JPEG; применяется только после поворота."""

    quality: ValueRange = ValueRange(28, 92)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=self.quality.sample_int(rng))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


@dataclass(frozen=True)
class ResolutionJitterDegradation(ImageDegradation):
    """Пережатие через меньшее разрешение и обратно."""

    scale: ValueRange = ValueRange(0.45, 0.95)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        factor = self.scale.sample(rng)
        shrunk = _resize(image, (max(MIN_SIDE, int(width * factor)), max(MIN_SIDE, int(height * factor))), rng)
        return _resize(shrunk, (width, height), rng)


@dataclass(frozen=True)
class ArcDegradation(ImageDegradation):
    """Изгиб строки по дуге.

    Текст на округлых предметах — банках, бутылках, печатях, круглых вывесках — идёт дугой,
    и прямых строк там нет. Реализовано смещением по вертикали, квадратичным по горизонтали:
    середина строки уходит вверх или вниз относительно краёв.

    Знак кривизны разыгрывается симметрично, иначе выпуклость сама стала бы признаком
    ориентации: у перевёрнутого кропа дуга смотрит в другую сторону.
    """

    curvature: ValueRange = ValueRange(0.05, 0.30)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        if width < MIN_SIDE or height < MIN_SIDE:
            return image
        amplitude = self.curvature.sample(rng) * height * (1.0 if rng.random() < 0.5 else -1.0)
        columns = np.arange(width, dtype=np.float32)
        # Парабола с нулями на краях и вершиной в середине строки.
        normalized = (columns - width / 2.0) / (width / 2.0)
        shift = amplitude * (1.0 - normalized ** 2)
        map_x = np.tile(columns, (height, 1))
        map_y = np.arange(height, dtype=np.float32)[:, None] - shift[None, :]
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class ResolutionCapDegradation(ImageDegradation):
    """Ограничивает рабочее разрешение перед остальными деградациями.

    Кроп приходит в нативном размере, вплоть до 477x1592, а модель получает 48x192. Деградации
    же применялись к полному разрешению: на крупном кропе это 15 мс против 0.35 мс у медианного,
    и вся разница выбрасывается финальным ужатием. Обучение из-за этого упиралось в подготовку
    данных, а видеокарта простаивала на три четверти.

    Ограничение вдвое выше входа сети выбрано не случайно: всё, что мельче двух пикселей на
    пиксель входа, после ужатия неразличимо, поэтому артефакты сжатия и шума сохраняют свой
    видимый масштаб. Метрика резкости это подтверждает: нормированная к высоте входа, она после
    ограничения не меняется.

    Конкретное значение приходит от `OrientationDatasetAssembler` и всегда равно удвоенной
    высоте входа сети; значение по умолчанию отвечает входу 32 и используется только в тестах.
    """

    max_height: int = 64

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        if height <= self.max_height:
            return image
        scale = self.max_height / height
        return _resize(image, (max(MIN_SIDE, int(round(width * scale))), self.max_height), rng)


@dataclass(frozen=True)
class ResizeToHeight:
    """Приводит кроп к целевой высоте, сохраняя пропорции."""

    def apply_to(self, image: np.ndarray, target_height: int, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        scale = target_height / height
        target_width = max(MIN_SIDE, int(round(width * scale)))
        return _resize(image, (target_width, max(MIN_SIDE, target_height)), rng)


def _resize(image: np.ndarray, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    # Метод интерполяции выбирается случайно: картинки на площадку попадают через разные
    # пайплайны обработки, и модель не должна привязываться к артефактам одного ресайзера.
    interpolation = INTERPOLATIONS[int(rng.integers(len(INTERPOLATIONS)))]
    return cv2.resize(image, size, interpolation=interpolation)


@dataclass(frozen=True)
class ProbabilisticDegradation:
    """Деградация, применяемая с заданной вероятностью."""

    degradation: ImageDegradation
    probability: float

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not happens(rng, self.probability):
            return image
        return self.degradation.apply(image, rng)


class DegradationPipeline:
    """Упорядоченная цепочка вероятностных деградаций."""

    def __init__(self, steps: Sequence[ProbabilisticDegradation]) -> None:
        self._steps = tuple(steps)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = image
        for step in self._steps:
            result = step.apply(result, rng)
        return result

    @classmethod
    def build_capture_stage(cls) -> "DegradationPipeline":
        # Применяется до поворота. Все операции здесь симметричны относительно поворота
        # на 180°, поэтому не оставляют следа, по которому можно угадать метку.
        return cls([
            ProbabilisticDegradation(ArcDegradation(), 0.12),
            ProbabilisticDegradation(PerspectiveDegradation(), 0.25),
            ProbabilisticDegradation(BrightnessContrastDegradation(), 0.45),
            ProbabilisticDegradation(GaussianBlurDegradation(), 0.30),
            ProbabilisticDegradation(MotionBlurDegradation(), 0.10),
        ])

    @classmethod
    def build_real_codec_stage(cls, max_height: int = DEFAULT_WORK_HEIGHT) -> "DegradationPipeline":
        # Для кропов из настоящих фотографий. Они уже несут собственные размытие, шум и
        # артефакты сжатия, поэтому синтетические деградации только сместили бы распределение:
        # замеры показали падение резкости вдвое относительно тестовой выборки.
        # Остаётся одно лёгкое перекодирование — оно нужно, чтобы у повёрнутых кропов сетка
        # блоков JPEG не отличалась от неповёрнутых, и не является аугментацией.
        return cls([
            ProbabilisticDegradation(ResolutionCapDegradation(max_height), 1.0),
            ProbabilisticDegradation(JpegDegradation(ValueRange(94, 96)), 1.0),
        ])

    @classmethod
    def build_real_training_stage(cls, max_height: int = DEFAULT_WORK_HEIGHT) -> "DegradationPipeline":
        """Для реальных кропов в обучении: аугментация, но не имитация.

        Обучающая стадия рассчитана на отрисованный текст, который надо испортить до похожести
        на съёмку. Настоящий кроп уже несёт своё размытие, шум и артефакты сжатия, и та же
        обработка делает его заметно мутнее тестового — на контактном листе это видно сразу.
        Поэтому размытия здесь нет вовсе, а остальное ослаблено: цель — разнообразие, а не
        воспроизведение условий съёмки, которые уже воспроизведены самой фотографией.
        """
        return cls([
            ProbabilisticDegradation(ResolutionCapDegradation(max_height), 1.0),
            ProbabilisticDegradation(ResolutionJitterDegradation(ValueRange(0.7, 0.95)), 0.12),
            ProbabilisticDegradation(BrightnessContrastDegradation(), 0.25),
            ProbabilisticDegradation(GaussianNoiseDegradation(ValueRange(1.0, 7.0)), 0.20),
            ProbabilisticDegradation(JpegDegradation(ValueRange(55, 95)), 0.70),
        ])

    @classmethod
    def build_codec_stage(cls, max_height: int = DEFAULT_WORK_HEIGHT) -> "DegradationPipeline":
        # Применяется только после поворота. JPEG режет картинку на блоки 8x8 от левого
        # верхнего угла, поэтому до поворота он оставил бы в перевёрнутых кропах другую сетку
        # блоков — готовый признак метки, никак не связанный с текстом.
        return cls([
            ProbabilisticDegradation(ResolutionCapDegradation(max_height), 1.0),
            ProbabilisticDegradation(ResolutionJitterDegradation(), 0.30),
            ProbabilisticDegradation(GaussianBlurDegradation(ValueRange(0.35, 0.85)), 0.55),
            ProbabilisticDegradation(GaussianNoiseDegradation(), 0.35),
            ProbabilisticDegradation(JpegDegradation(), 0.80),
        ])
