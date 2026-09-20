from __future__ import annotations

import numpy as np
import pytest
import torch

from avitocv.model.architecture import (
    BlockKind,
    MeanWidthPooling,
    ModelCostMeter,
    ModelFactory,
    TinyNetConfig,
    TinyOrientationNet,
)
from avitocv.model.inference import (
    CalibratedPredictor,
    DirectPredictor,
    SymmetricPredictor,
    TemperatureCalibrator,
    TemperatureScaler,
    rotate_half_turn,
)
from avitocv.training.metrics import ConsistencyReport, MetricsAccumulator

INPUT_SHAPE = (1, 32, 192)
BATCH_SIZE = 8


def _batch(seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(BATCH_SIZE, *INPUT_SHAPE)).astype(np.float32))


class TestTinyOrientationNet:
    def test_output_is_one_logit_per_crop(self) -> None:
        assert TinyOrientationNet()(_batch()).shape == (BATCH_SIZE,)

    def test_network_stays_small(self) -> None:
        cost = ModelCostMeter().measure(TinyOrientationNet(), INPUT_SHAPE)
        assert cost.parameter_count < 60_000
        assert cost.multiply_accumulates < 10_000_000

    def test_separable_blocks_are_cheaper_than_dense_ones(self) -> None:
        meter = ModelCostMeter()
        separable = meter.measure(TinyOrientationNet(TinyNetConfig()), INPUT_SHAPE)
        dense = meter.measure(TinyOrientationNet(TinyNetConfig(block_kind=BlockKind.DENSE)), INPUT_SHAPE)
        assert separable.multiply_accumulates < dense.multiply_accumulates / 3

    def test_vertical_position_reaches_the_head(self) -> None:
        """Высота сворачивается в каналы, поэтому сдвиг признака по вертикали обязан менять ответ."""
        model = TinyOrientationNet().eval()
        top, bottom = torch.zeros(1, *INPUT_SHAPE), torch.zeros(1, *INPUT_SHAPE)
        top[:, :, 4:8, :] = 1.0
        bottom[:, :, 24:28, :] = 1.0
        with torch.no_grad():
            assert not torch.allclose(model(top), model(bottom))

    def test_mean_pooling_reduces_the_head_size(self) -> None:
        mean_only = TinyOrientationNet(TinyNetConfig(), MeanWidthPooling())
        mean_max = TinyOrientationNet(TinyNetConfig())
        assert sum(p.numel() for p in mean_only.parameters()) < sum(p.numel() for p in mean_max.parameters())

    def test_height_that_does_not_divide_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TinyNetConfig(input_height=30)


class TestModelFactory:
    def test_known_architectures_are_built(self) -> None:
        factory = ModelFactory()
        for name in ("tiny", "tiny_tall", "tiny_mean", "tiny_dense"):
            assert factory.create(name)(_batch()).shape == (BATCH_SIZE,)

    def test_unknown_architecture_is_reported(self) -> None:
        with pytest.raises(ValueError):
            ModelFactory().create("resnet1000")


class TestHalfTurn:
    def test_rotating_twice_returns_the_original(self) -> None:
        images = _batch()
        assert torch.equal(rotate_half_turn(rotate_half_turn(images)), images)

    def test_rotation_flips_both_axes(self) -> None:
        images = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        assert torch.equal(rotate_half_turn(images).flatten(), torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]))


class TestSymmetricPredictor:
    """Симметризация навязывает точное соотношение p(x) + p(rot180 x) = 1."""

    def test_logits_are_exactly_antisymmetric(self) -> None:
        model = TinyOrientationNet().eval()
        predictor = SymmetricPredictor(model)
        images = _batch(1)
        with torch.no_grad():
            forward = predictor.logits(images)
            flipped = predictor.logits(rotate_half_turn(images))
        assert torch.allclose(forward, -flipped, atol=1e-5)

    def test_direct_predictor_is_not_antisymmetric(self) -> None:
        model = TinyOrientationNet().eval()
        images = _batch(2)
        with torch.no_grad():
            forward = DirectPredictor(model).logits(images)
            flipped = DirectPredictor(model).logits(rotate_half_turn(images))
        assert not torch.allclose(forward, -flipped, atol=1e-3)

    def test_mean_probability_is_exactly_one_half(self) -> None:
        model = TinyOrientationNet().eval()
        probabilities = CalibratedPredictor(SymmetricPredictor(model)).probabilities(_batch(3))
        flipped = CalibratedPredictor(SymmetricPredictor(model)).probabilities(rotate_half_turn(_batch(3)))
        assert np.allclose(probabilities + flipped, 1.0, atol=1e-5)


class TestTemperatureCalibration:
    def _overconfident(self, size: int = 4000) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, size=size).astype(np.float64)
        true_logits = rng.normal(0.0, 1.2, size=size) + (labels * 2.0 - 1.0) * 1.5
        return true_logits * 4.0, labels

    def test_calibration_lowers_brier(self) -> None:
        logits, labels = self._overconfident()
        scaler = TemperatureCalibrator().fit(logits, labels)
        before = np.mean((TemperatureScaler().probabilities(logits) - labels) ** 2)
        after = np.mean((scaler.probabilities(logits) - labels) ** 2)
        assert after < before

    def test_overconfident_logits_get_temperature_above_one(self) -> None:
        logits, labels = self._overconfident()
        assert TemperatureCalibrator().fit(logits, labels).temperature > 1.0

    def test_weights_shift_the_fitted_temperature(self) -> None:
        logits, labels = self._overconfident()
        weights = np.where(labels > 0.5, 5.0, 0.2)
        plain = TemperatureCalibrator().fit(logits, labels).temperature
        weighted = TemperatureCalibrator().fit(logits, labels, weights).temperature
        assert plain != pytest.approx(weighted, rel=1e-3)


class TestMetrics:
    def test_perfect_predictions_score_one(self) -> None:
        accumulator = MetricsAccumulator()
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        accumulator.add(labels, labels, np.ones(4))
        assert accumulator.result().score == pytest.approx(1.0)

    def test_constant_half_scores_three_quarters(self) -> None:
        accumulator = MetricsAccumulator()
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        accumulator.add(np.full(4, 0.5), labels, np.ones(4))
        assert accumulator.result().score == pytest.approx(0.75)

    def test_weights_change_the_result(self) -> None:
        labels = np.array([0.0, 1.0])
        probabilities = np.array([0.0, 0.0])
        plain, weighted = MetricsAccumulator(), MetricsAccumulator()
        plain.add(probabilities, labels, np.ones(2))
        weighted.add(probabilities, labels, np.array([9.0, 1.0]))
        assert weighted.result().brier < plain.result().brier

    def test_effective_count_drops_with_uneven_weights(self) -> None:
        accumulator = MetricsAccumulator()
        accumulator.add(np.full(100, 0.5), np.zeros(100), np.array([100.0] + [0.01] * 99))
        result = accumulator.result()
        assert result.sample_count == 100 and result.effective_count < 2.0

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            MetricsAccumulator().add(np.zeros(3), np.zeros(2), np.ones(3))

    def test_empty_accumulator_is_reported(self) -> None:
        with pytest.raises(ValueError):
            MetricsAccumulator().result()


class TestConsistencyReport:
    def test_consistent_pairs_have_no_deviation(self) -> None:
        forward = np.array([0.9, 0.2, 0.5])
        report = ConsistencyReport.from_pairs(forward, 1.0 - forward)
        assert report.mean_absolute_deviation == pytest.approx(0.0)

    def test_inconsistent_pairs_are_detected(self) -> None:
        report = ConsistencyReport.from_pairs(np.array([0.9, 0.9]), np.array([0.9, 0.9]))
        assert report.mean_absolute_deviation == pytest.approx(0.8)

    def test_bimodality_counts_confident_predictions(self) -> None:
        forward = np.array([0.99, 0.01, 0.5, 0.5])
        report = ConsistencyReport.from_pairs(forward, 1.0 - forward)
        assert report.bimodality == pytest.approx(0.5)


class TestHeightDownsampling:
    """Вертикальное разрешение у головы — рычаг для коротких кропов.

    У почти симметричных букв ориентацию выдают доли процента высоты глифа: перекладина Н выше
    центра, верхняя чаша В меньше нижней. При прореживании в восемь раз голова видит шесть строк
    и таких различий не разрешает, а на коротких кропах других зацепок нет.
    """

    def test_less_downsampling_keeps_more_rows(self) -> None:
        coarse = TinyNetConfig(input_height=48, height_downsample=8)
        fine = TinyNetConfig(input_height=48, height_downsample=4)
        assert (coarse.final_height, fine.final_height) == (6, 12)

    def test_feature_map_matches_the_declared_height(self) -> None:
        config = TinyNetConfig(input_height=48, height_downsample=4)
        model = TinyOrientationNet(config).eval()
        with torch.no_grad():
            features = model.features(torch.zeros(1, 1, 48, 192))
        assert features.shape[2] == config.final_height

    def test_finer_vertical_detail_is_cheaper_than_a_taller_input(self) -> None:
        """Поднимать вход дороже: платишь и за ширину, которая коротким кропам не нужна."""
        meter = ModelCostMeter()
        finer = meter.measure(TinyOrientationNet(TinyNetConfig(input_height=48, height_downsample=4)), (1, 48, 192))
        taller = meter.measure(TinyOrientationNet(TinyNetConfig(input_height=64, height_downsample=8)), (1, 64, 192))
        assert finer.multiply_accumulates < taller.multiply_accumulates

    def test_output_is_still_one_logit_per_crop(self) -> None:
        model = TinyOrientationNet(TinyNetConfig(input_height=48, height_downsample=4))
        assert model(torch.zeros(BATCH_SIZE, 1, 48, 192)).shape == (BATCH_SIZE,)

    def test_non_power_of_two_downsampling_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TinyNetConfig(input_height=48, height_downsample=6)

    def test_downsampling_beyond_available_stages_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TinyNetConfig(input_height=64, height_downsample=16)

    def test_tall_variant_is_available_from_the_factory(self) -> None:
        factory = ModelFactory(TinyNetConfig(input_height=48, height_downsample=8))
        assert factory.create("tiny_tall")(torch.zeros(2, 1, 48, 192)).shape == (2,)
