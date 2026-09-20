"""Конфигурация данных, читаемая из YAML (`configs/data.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from avitocv.data.datasets import PreprocessConfig


@dataclass(frozen=True)
class DataPaths:
    """Пути к ассетам: индекс шрифтов, корпуса, профиль теста, каталог фонов."""

    font_index: Path
    russian_corpus: Path
    english_corpus: Path
    test_profile: Path
    background_dir: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "DataPaths":
        background = payload.get("background_dir")
        return cls(
            font_index=Path(payload["font_index"]),
            russian_corpus=Path(payload["russian_corpus"]),
            english_corpus=Path(payload["english_corpus"]),
            test_profile=Path(payload["test_profile"]),
            background_dir=Path(background) if background else None,
        )


@dataclass(frozen=True)
class SourceWeights:
    """Соотношение русского, английского и процедурного текста в выборке."""

    russian: float = 6.0
    english: float = 2.0
    patterns: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict) -> "SourceWeights":
        return cls(**{key: float(value) for key, value in payload.items()})


@dataclass(frozen=True)
class DataConfig:
    """Полная конфигурация данных: пути, веса, препроцессинг, длины эпох и сиды."""

    paths: DataPaths
    weights: SourceWeights = field(default_factory=SourceWeights)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train_length: int = 1_000_000
    validation_length: int = 20_000
    train_seed: int = 1234
    validation_seed: int = 9999

    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            paths=DataPaths.from_dict(payload["paths"]),
            weights=SourceWeights.from_dict(payload.get("weights", {})),
            preprocess=PreprocessConfig(**payload.get("preprocess", {})),
            train_length=int(payload.get("train_length", cls.train_length)),
            validation_length=int(payload.get("validation_length", cls.validation_length)),
            train_seed=int(payload.get("train_seed", cls.train_seed)),
            validation_seed=int(payload.get("validation_seed", cls.validation_seed)),
        )
