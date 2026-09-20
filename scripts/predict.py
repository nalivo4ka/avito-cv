"""Считает предсказания для тестовой выборки и пишет submission.csv.

Помимо самого сабмишена скрипт печатает безметочную диагностику: для любого кропа ровно одна
из ориентаций верна, поэтому у согласованной модели p(x) + p(rot180(x)) = 1. Отклонение
считается прямо на выданных 20 000 кропов, без единой метки, и показывает, насколько модель
чувствует себя дома на реальном домене. Это главный инструмент, чтобы не тратить вслепую
попытки отправки, которых всего семь.

Сами предсказания берутся с симметризацией, у которой это соотношение выполняется тождественно,
поэтому диагностика считается по прямому прогону — иначе она измеряла бы ноль по построению.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress
from torch.utils.data import DataLoader

from avitocv.config import DataConfig
from avitocv.data.datasets import CenterWindowFit, CropInferenceDataset, ImagePreprocessor
from avitocv.model.architecture import ModelCostMeter, ModelFactory, TinyNetConfig
from avitocv.model.inference import (
    DirectPredictor,
    ProbabilityEnsemble,
    TemperatureScaler,
    rotate_half_turn,
)
from avitocv.training.metrics import ConsistencyReport

SUBMISSION_COLUMNS = ("image_id", "p_180")

# Состав, которым получен отправленный submission.csv: путь:высота_входа:прореживание_высоты.
# Он же значение по умолчанию, чтобы запуск без единого аргумента воспроизводил отправленное.
SUBMITTED_ENSEMBLE = (
    "artifacts/tall_v2/best.pt:48:4",
    "artifacts/tiny_v2/best.pt:48:8",
)


@dataclass(frozen=True)
class EnsembleMember:
    """Чекпоинт вместе с геометрией входа, с которой он обучался.

    Геометрия входит в описание участника, потому что у моделей она разная: `tiny_tall` видит
    вдвое более подробную высоту, чем `tiny`, и каждому нужен свой препроцессинг. Поэтому
    усреднять приходится уже вероятности, а не признаки.
    """

    checkpoint: Path
    input_height: int
    height_downsample: int

    @classmethod
    def parse(cls, specification: str) -> "EnsembleMember":
        parts = specification.split(":")
        if len(parts) != 3:
            raise ValueError(f"ожидается путь:высота:прореживание, получено {specification!r}")
        return cls(Path(parts[0]), int(parts[1]), int(parts[2]))


class CheckpointLoader:
    """Восстанавливает модель и калибровку из сохранённого чекпоинта."""

    def load(
        self,
        path: Path,
        architecture: str | None,
        device: torch.device,
        input_height: int,
        height_downsample: int,
    ) -> tuple[torch.nn.Module, TemperatureScaler]:
        payload = torch.load(path, map_location=device, weights_only=True)
        # Архитектура записана в чекпоинт, поэтому её не надо помнить и передавать руками.
        # А вот геометрия входа в чекпоинт не пишется, и при несовпадении torch сообщает
        # только про размеры тензора головы — по такому сообщению не догадаться, что надо
        # поправить. Поэтому ошибка перехватывается и переводится на человеческий язык.
        model = ModelFactory(TinyNetConfig(input_height=input_height,
                                           height_downsample=height_downsample)).create(
            architecture or payload["architecture"])
        try:
            model.load_state_dict(payload["state_dict"])
        except RuntimeError as error:
            raise SystemExit(
                f"веса {path} не подходят под заданную геометрию входа "
                f"(высота {input_height}, прореживание {height_downsample}).\n"
                f"Укажите ту, с которой модель обучалась, например:\n"
                f"  python scripts/predict.py --member {path}:48:4\n"
                f"Исходная ошибка: {error}"
            ) from error
        model.to(device).eval()
        return model, TemperatureScaler(float(payload.get("temperature", 1.0)))


class LoadedMember:
    """Участник ансамбля с загруженной моделью и своей калибровкой."""

    def __init__(self, model: torch.nn.Module, scaler: TemperatureScaler, cost: float) -> None:
        self.predictor = DirectPredictor(model)
        self.scaler = scaler
        self.cost = cost
        self.forward: list[np.ndarray] = []
        self.flipped: list[np.ndarray] = []

    def observe(self, images: torch.Tensor) -> None:
        self.forward.append(self.predictor.logits(images).float().cpu().numpy())
        self.flipped.append(self.predictor.logits(rotate_half_turn(images)).float().cpu().numpy())

    def probabilities(self) -> np.ndarray:
        # Симметризация: логит антисимметричен по построению, поэтому p(x) + p(rot180 x) = 1.
        forward, flipped = np.concatenate(self.forward), np.concatenate(self.flipped)
        return self.scaler.probabilities((forward - flipped) / 2.0)

    def consistency(self) -> ConsistencyReport:
        """Безметочная диагностика считается по прямому прогону: у симметризованного она ноль."""
        scaler = TemperatureScaler(self.scaler.temperature)
        return ConsistencyReport.from_pairs(
            scaler.probabilities(np.concatenate(self.forward)),
            scaler.probabilities(np.concatenate(self.flipped)),
        )


class TestSetPredictor:
    """Прогоняет тест один раз на всю группу участников с общей геометрией входа.

    Разделять по геометрии приходится потому, что у моделей разный вход, и каждой нужен свой
    препроцессинг. Но у участников одного размера он общий, а именно на нём и уходит время:
    чтение и подготовка 20 000 PNG занимают 15 секунд против 1.9 секунды на прогон сети. Читать
    тест заново под каждого участника означало бы удваивать самую дорогую часть ради самой
    дешёвой — ансамбль из двух стоил бы 31 секунду вместо 18.
    """

    def __init__(self, device: torch.device, batch_size: int, worker_count: int) -> None:
        self._device = device
        self._batch_size = batch_size
        self._worker_count = worker_count

    @torch.no_grad()
    def run(self, members: list[LoadedMember], dataset: CropInferenceDataset) -> None:
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=self._worker_count)
        with Progress() as progress:
            task = progress.add_task("предсказание", total=len(loader))
            for images in loader:
                images = images.to(self._device, non_blocking=True)
                for member in members:
                    member.observe(images)
                progress.advance(task)


def write_submission(path: Path, image_ids: tuple[str, ...], probabilities: np.ndarray) -> None:
    frame = pd.DataFrame({SUBMISSION_COLUMNS[0]: image_ids, SUBMISSION_COLUMNS[1]: probabilities})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Посчитать предсказания для тестовой выборки")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="одиночная модель вместо ансамбля по умолчанию; "
                             "не забудьте --input-height и --height-downsample, с которыми она обучалась")
    parser.add_argument("--member", action="append", default=None,
                        help="участник ансамбля путь:высота:прореживание, можно повторять; "
                             f"по умолчанию {' и '.join(SUBMITTED_ENSEMBLE)}")
    parser.add_argument("--architecture", default=None, help="по умолчанию берётся из чекпоинта")
    parser.add_argument("--images-dir", type=Path, default=Path("test/images"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--height-downsample", type=int, default=8,
                        help="прореживание высоты внутри сети, как при обучении")
    parser.add_argument("--input-height", type=int, default=0,
                        help="высота входа сети, если отличается от конфига")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_members(arguments: argparse.Namespace) -> list[EnsembleMember]:
    """Участники ансамбля.

    По умолчанию берётся тот же состав, которым получен отправленный `submission.csv`, поэтому
    запуск вообще без аргументов воспроизводит именно его. Геометрия входа в чекпоинт не
    пишется, и без этих значений по умолчанию такой запуск падал бы на несовпадении форм.
    """
    if arguments.checkpoint is not None:
        return [EnsembleMember(arguments.checkpoint, arguments.input_height or 0, arguments.height_downsample)]
    return [EnsembleMember.parse(item) for item in (arguments.member or SUBMITTED_ENSEMBLE)]


def group_by_geometry(members: list[EnsembleMember]) -> dict[int, list[EnsembleMember]]:
    """Участники с одинаковой высотой входа делят чтение и препроцессинг теста."""
    groups: dict[int, list[EnsembleMember]] = {}
    for member in members:
        groups.setdefault(member.input_height, []).append(member)
    return groups


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(0)
    device = resolve_device(arguments.device)
    base_config = DataConfig.from_yaml(arguments.config)
    loader = CheckpointLoader()
    runner = TestSetPredictor(device, arguments.batch_size, arguments.workers)

    predictions, image_ids, total_cost = [], None, 0.0
    for input_height, group in group_by_geometry(build_members(arguments)).items():
        config = base_config
        if input_height:
            config = replace(config, preprocess=replace(config.preprocess, height=input_height))
        preprocessor = ImagePreprocessor(config.preprocess, CenterWindowFit())
        dataset = CropInferenceDataset.from_directory(arguments.images_dir, preprocessor)
        if image_ids is not None and dataset.image_ids != image_ids:
            raise ValueError("участники ансамбля прошли по разным наборам картинок")
        image_ids = dataset.image_ids

        loaded = []
        shape = (config.preprocess.channel_count, config.preprocess.height, config.preprocess.width)
        for member in group:
            model, scaler = loader.load(member.checkpoint, arguments.architecture, device,
                                        config.preprocess.height, member.height_downsample)
            cost = ModelCostMeter().measure(model.cpu(), shape)
            model.to(device)
            print(f"{member.checkpoint}: {cost.describe()}, температура {scaler.temperature:.3f}")
            loaded.append(LoadedMember(model, scaler, cost.multiply_accumulates))

        runner.run(loaded, dataset)
        for member, entry in zip(group, loaded):
            print(f"  {member.checkpoint.parent.name}: {entry.consistency().describe()}")
            predictions.append(entry.probabilities())
            total_cost += entry.cost

    # Усредняются вероятности, а не логиты: Brier строго выпукла, поэтому по неравенству
    # Йенсена ошибка среднего не превышает среднюю ошибку участников.
    probabilities = ProbabilityEnsemble().combine(predictions)
    write_submission(arguments.output, image_ids, probabilities)

    print(f"участников {len(predictions)}, суммарно {total_cost / 1e6:.1f}M MAC на кроп")
    print(f"средняя вероятность {probabilities.mean():.4f}")
    print(f"уверенных (|p-0.5|>0.4): {np.mean(np.abs(probabilities - 0.5) > 0.4):.1%}")
    print(f"записано: {arguments.output}")


if __name__ == "__main__":
    main()
