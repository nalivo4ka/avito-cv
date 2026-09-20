"""Архитектуры классификатора ориентации и подсчёт их стоимости.

Различающий сигнал в задаче — вертикальная асимметрия строки: выносные элементы букв, положение
базовой линии, высота точек и запятых. Отсюда два решения, определяющих форму сети.

**Высота сворачивается в каналы, а не усредняется.** Обычное глобальное усреднение по всей
картинке уничтожило бы информацию «на какой высоте находится признак» — ровно ту, которая
отличает `р` от `ь`. Поэтому после свёрток остаток высоты переносится в измерение каналов.

**Усреднение идёт только по ширине.** Ориентация — свойство всей строки, и каждая буква даёт
независимое свидетельство; агрегирование по ширине их складывает. Дополнительно берётся максимум:
одного отчётливого выносного элемента достаточно, чтобы снять неопределённость, и среднее такой
одиночный признак размывает.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum

import torch
from torch import nn

LOGIT_OUTPUT_SIZE = 1
CONVOLUTION_KERNEL_SIZE = 3


class WidthPooling(nn.Module, ABC):
    """Сворачивает измерение ширины, оставляя вектор признаков на строку."""

    @property
    @abstractmethod
    def output_multiplier(self) -> int:
        """Во сколько раз вырастает размер признака после агрегирования."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeanWidthPooling(WidthPooling):
    """Среднее по ширине: складывает свидетельства всех букв строки."""

    @property
    def output_multiplier(self) -> int:
        return 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=-1)


class MeanMaxWidthPooling(WidthPooling):
    """Среднее вместе с максимумом: максимум ловит одиночный отчётливый выносной элемент."""

    @property
    def output_multiplier(self) -> int:
        return 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)


class BlockKind(Enum):
    """Тип свёрточного блока.

    Плотная свёртка 3x3 стоит in*out*9 умножений на позицию, раздельная — in*9 + in*out.
    На наших размерах это разница в 5-7 раз по MAC при почти той же точности, поэтому
    раздельные блоки взяты по умолчанию: задача отдельно оценивает производительность.
    """

    DENSE = "dense"
    SEPARABLE = "separable"


@dataclass(frozen=True)
class TinyNetConfig:
    """Размеры своей сети: каналы блоков, высота входа и тип блока."""

    channels: tuple[int, ...] = (24, 48, 96, 128)
    input_height: int = 32
    dropout: float = 0.0
    block_kind: BlockKind = BlockKind.SEPARABLE
    # Во сколько раз прореживается высота до головы сети. Ширину блоки прореживают всегда,
    # высоту — только пока не достигнут этого множителя.
    #
    # Параметр не косметический. Признаки, различающие ориентацию у почти симметричных букв,
    # очень тонкие: перекладина Н сидит выше центра, верхняя чаша В и S меньше нижней
    # (типографская оптическая коррекция), точка пересечения X тоже выше середины. Это единицы
    # процентов высоты глифа. При прореживании в восемь раз голова видит шесть строк на вход
    # 48 px, и такие различия там уже не разрешаются — а именно они остаются единственной
    # зацепкой на коротких кропах, где букв мало и брать больше неоткуда.
    height_downsample: int = 8

    def __post_init__(self) -> None:
        if len(self.channels) < 2:
            raise ValueError("нужно хотя бы два блока свёрток")
        if self.height_downsample < 2 or self.height_downsample & (self.height_downsample - 1):
            raise ValueError(f"прореживание высоты должно быть степенью двойки, получено {self.height_downsample}")
        if self.height_downsample > 2 ** (len(self.channels) - 1):
            raise ValueError("прореживание высоты не может превышать число прореживающих ступеней")
        if self.input_height % self.height_downsample != 0:
            raise ValueError(f"высота входа {self.input_height} не делится на {self.height_downsample}")

    @property
    def final_height(self) -> int:
        return self.input_height // self.height_downsample


class DenseConvolutionBlock(nn.Sequential):
    """Плотная свёртка, нормализация, нелинейность и прореживание."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, CONVOLUTION_KERNEL_SIZE, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )


class SeparableConvolutionBlock(nn.Sequential):
    """Раздельная свёртка: пространственная по каналам отдельно, затем смешивание каналов."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: tuple[int, int]) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                CONVOLUTION_KERNEL_SIZE,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )


class ConvolutionStem(nn.Sequential):
    """Первый слой: плотная свёртка с шагом 2. На одном входном канале она почти бесплатна,
    а дальше вся сеть работает на вдвое меньшем разрешении."""

    def __init__(self, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(1, out_channels, CONVOLUTION_KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TinyOrientationNet(nn.Module):
    """Своя компактная сеть: свёртки, высота в каналы, агрегирование по ширине, линейный выход."""

    def __init__(self, config: TinyNetConfig = TinyNetConfig(), pooling: WidthPooling | None = None) -> None:
        super().__init__()
        self._config = config
        self._pooling = pooling or MeanMaxWidthPooling()
        self.features = self._build_features(config)
        feature_size = config.channels[-1] * config.final_height * self._pooling.output_multiplier
        self.head = nn.Sequential(
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
            nn.Linear(feature_size, LOGIT_OUTPUT_SIZE),
        )

    @staticmethod
    def _build_features(config: TinyNetConfig) -> nn.Sequential:
        block_type = (
            SeparableConvolutionBlock if config.block_kind is BlockKind.SEPARABLE else DenseConvolutionBlock
        )
        # Стем уже прореживает высоту вдвое, поэтому счётчик начинается с двойки.
        blocks: list[nn.Module] = [ConvolutionStem(config.channels[0])]
        height_factor = 2
        in_channels = config.channels[0]
        for out_channels in config.channels[1:]:
            pools_height = height_factor < config.height_downsample
            blocks.append(block_type(in_channels, out_channels, (2, 2) if pools_height else (1, 2)))
            height_factor *= 2 if pools_height else 1
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        # (B, C, H, W) -> (B, C*H, W): высота переносится в каналы, чтобы вертикальное
        # положение признака дошло до головы сети, а не усреднилось.
        batch_size, channels, height, width = features.shape
        merged = features.reshape(batch_size, channels * height, width)
        return self.head(self._pooling(merged)).squeeze(-1)


@dataclass(frozen=True)
class ModelCost:
    """Стоимость модели: то, по чему задача отдельно сравнивает решения."""

    parameter_count: int
    multiply_accumulates: int

    @property
    def megabytes_fp32(self) -> float:
        return self.parameter_count * 4 / 1024 ** 2

    def describe(self) -> str:
        return (
            f"параметров {self.parameter_count:,}"
            f" ({self.megabytes_fp32:.2f} МБ fp32),"
            f" MAC на кроп {self.multiply_accumulates / 1e6:.2f}M"
        ).replace(",", " ")


class ModelCostMeter:
    """Считает параметры и умножения-накопления одним прогоном с хуками."""

    _COUNTED_TYPES = (nn.Conv2d, nn.Linear)

    def measure(self, model: nn.Module, input_shape: tuple[int, int, int]) -> ModelCost:
        totals: list[int] = []
        handles = [
            module.register_forward_hook(self._make_hook(totals))
            for module in model.modules()
            if isinstance(module, self._COUNTED_TYPES)
        ]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
        for handle in handles:
            handle.remove()
        model.train(was_training)
        return ModelCost(
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            multiply_accumulates=sum(totals),
        )

    def _make_hook(self, totals: list[int]):
        def hook(module: nn.Module, inputs, output) -> None:
            if isinstance(module, nn.Conv2d):
                output_positions = output.shape[-2] * output.shape[-1]
                per_position = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
                totals.append(module.out_channels * output_positions * per_position)
                return
            totals.append(module.in_features * module.out_features)

        return hook


@dataclass(frozen=True)
class ModelFactory:
    """Создаёт модель по имени из конфига."""

    tiny_config: TinyNetConfig = field(default_factory=TinyNetConfig)

    def create(self, name: str) -> nn.Module:
        if name == "tiny":
            return TinyOrientationNet(self.tiny_config)
        if name == "tiny_mean":
            return TinyOrientationNet(self.tiny_config, MeanWidthPooling())
        if name == "tiny_dense":
            return TinyOrientationNet(replace(self.tiny_config, block_kind=BlockKind.DENSE))
        if name == "tiny_tall":
            return TinyOrientationNet(replace(self.tiny_config, height_downsample=4))
        raise ValueError(f"неизвестная архитектура: {name!r}")
