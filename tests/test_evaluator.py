"""Tests for the Evaluator and EvaluationResult."""

from __future__ import annotations

import pytest

from llm_calibration_eval.evaluator import EvaluationResult, Evaluator
from llm_calibration_eval.providers.base import CalibrationSample
from llm_calibration_eval.providers.mock_provider import MockProvider
from llm_calibration_eval.datasets.trivia import TRIVIA_DATASET, load_dataset
from llm_calibration_eval.calibration.temperature_scaling import TemperatureScaling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_samples(n: int = 10) -> list[CalibrationSample]:
    return [CalibrationSample(f"Question {i}?", f"Answer{i}", "test") for i in range(n)]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_run_returns_evaluation_result(self) -> None:
        provider = MockProvider(accuracy=1.0, seed=0)
        evaluator = Evaluator(provider)
        samples = make_samples(5)
        result = evaluator.run(samples)
        assert isinstance(result, EvaluationResult)

    def test_perfect_accuracy_provider(self) -> None:
        provider = MockProvider(accuracy=1.0, seed=0)
        samples = make_samples(10)
        result = Evaluator(provider).run(samples)
        assert result.n_correct == 10
        assert result.accuracy == pytest.approx(1.0)

    def test_zero_accuracy_provider(self) -> None:
        provider = MockProvider(accuracy=0.0, seed=0)
        samples = make_samples(10)
        result = Evaluator(provider).run(samples)
        assert result.n_correct == 0
        assert result.accuracy == pytest.approx(0.0)

    def test_sample_count(self) -> None:
        provider = MockProvider(seed=0)
        samples = make_samples(15)
        result = Evaluator(provider).run(samples)
        assert result.n_samples == 15

    def test_metrics_are_numbers(self) -> None:
        provider = MockProvider(seed=1)
        result = Evaluator(provider).run(make_samples(10))
        assert isinstance(result.brier_score, float)
        assert isinstance(result.ece, float)
        assert isinstance(result.mean_confidence, float)

    def test_brier_score_in_range(self) -> None:
        provider = MockProvider(seed=2)
        result = Evaluator(provider).run(make_samples(20))
        assert 0.0 <= result.brier_score <= 1.0

    def test_ece_non_negative(self) -> None:
        provider = MockProvider(seed=3)
        result = Evaluator(provider).run(make_samples(20))
        assert result.ece >= 0.0

    def test_mean_confidence_in_range(self) -> None:
        provider = MockProvider(seed=4)
        result = Evaluator(provider).run(make_samples(20))
        assert 0.0 <= result.mean_confidence <= 1.0

    def test_reliability_bins_present(self) -> None:
        provider = MockProvider(seed=5)
        result = Evaluator(provider).run(make_samples(20))
        assert len(result.reliability_bins) > 0

    def test_responses_list_length(self) -> None:
        provider = MockProvider(seed=6)
        samples = make_samples(8)
        result = Evaluator(provider).run(samples)
        assert len(result.responses) == 8

    def test_custom_judge(self) -> None:
        # Judge that always says correct
        provider = MockProvider(accuracy=0.0, seed=0)
        evaluator = Evaluator(provider, judge=lambda r, s: True)
        result = evaluator.run(make_samples(5))
        assert result.n_correct == 5

    def test_dataset_name_stored(self) -> None:
        provider = MockProvider(seed=0)
        result = Evaluator(provider).run(make_samples(5), dataset_name="my-dataset")
        assert result.dataset_name == "my-dataset"

    def test_empty_samples_raises(self) -> None:
        provider = MockProvider(seed=0)
        with pytest.raises(ValueError, match="empty"):
            Evaluator(provider).run([])

    def test_provider_name_in_result(self) -> None:
        provider = MockProvider(model="my-model", seed=0)
        result = Evaluator(provider).run(make_samples(3))
        assert result.provider_name == "my-model"

    def test_elapsed_seconds_positive(self) -> None:
        provider = MockProvider(seed=0)
        result = Evaluator(provider).run(make_samples(5))
        assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def _make_result(self) -> EvaluationResult:
        provider = MockProvider(seed=42)
        samples = make_samples(20)
        return Evaluator(provider).run(samples, dataset_name="test")

    def test_summary_table_is_string(self) -> None:
        result = self._make_result()
        table = result.summary_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_summary_table_contains_metrics(self) -> None:
        result = self._make_result()
        table = result.summary_table()
        assert "Brier" in table
        assert "ECE" in table
        assert "Accuracy" in table


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_trivia_dataset_has_50_items(self) -> None:
        assert len(TRIVIA_DATASET) == 50

    def test_load_dataset_trivia(self) -> None:
        ds = load_dataset("trivia")
        assert len(ds) == 50

    def test_load_dataset_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")

    def test_all_samples_have_answer(self) -> None:
        for sample in TRIVIA_DATASET:
            assert sample.correct_answer.strip() != ""

    def test_all_samples_have_question(self) -> None:
        for sample in TRIVIA_DATASET:
            assert sample.question.strip() != ""


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    def test_fit_returns_self(self) -> None:
        ts = TemperatureScaling()
        result = ts.fit([0.7, 0.8, 0.6, 0.9], [1, 1, 0, 1])
        assert result is ts

    def test_temperature_positive_after_fit(self) -> None:
        ts = TemperatureScaling()
        ts.fit([0.9] * 20, [0] * 20)  # overconfident → T > 1
        assert ts.temperature_ > 0.0

    def test_overconfident_increases_temperature(self) -> None:
        ts = TemperatureScaling()
        ts.fit([0.9] * 20, [0] * 20)
        assert ts.temperature_ > 1.0  # should soften probabilities

    def test_transform_returns_same_length(self) -> None:
        ts = TemperatureScaling()
        ts.fit([0.8, 0.6, 0.7], [1, 0, 1])
        confs = [0.8, 0.6, 0.7, 0.5]
        result = ts.transform(confs)
        assert len(result) == 4

    def test_transform_values_in_range(self) -> None:
        ts = TemperatureScaling()
        ts.fit([0.8, 0.6, 0.7], [1, 0, 1])
        result = ts.transform([0.2, 0.5, 0.8, 0.99])
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_fit_transform(self) -> None:
        ts = TemperatureScaling()
        confs = [0.9, 0.85, 0.8]
        outcomes = [1, 1, 0]
        result = ts.fit_transform(confs, outcomes)
        assert len(result) == 3

    def test_invalid_init_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaling(init_temperature=0.0)

    def test_empty_fit_raises(self) -> None:
        ts = TemperatureScaling()
        with pytest.raises(ValueError):
            ts.fit([], [])

    def test_temperature_one_no_change(self) -> None:
        """With T=1 and no fitting, transform should not change values much."""
        ts = TemperatureScaling(init_temperature=1.0)
        ts.temperature_ = 1.0
        result = ts.transform([0.7])
        # logit(0.7) = 0.847; sigmoid(0.847) = 0.7
        assert result[0] == pytest.approx(0.7, abs=0.01)

    def test_run_full_pipeline(self) -> None:
        """Integration: evaluate, fit temperature, re-evaluate."""
        provider = MockProvider.overconfident(accuracy=0.5, seed=0)
        samples = make_samples(30)
        evaluator = Evaluator(provider)
        result = evaluator.run(samples)

        confs = [r.confidence for _, r, _ in result.responses]
        outcomes = [int(correct) for _, _, correct in result.responses]

        ts = TemperatureScaling()
        calibrated_confs = ts.fit_transform(confs, outcomes)
        assert len(calibrated_confs) == 30
