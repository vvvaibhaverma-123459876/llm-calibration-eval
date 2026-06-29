"""Tests for calibration metrics."""

from __future__ import annotations

import math

import pytest

from llm_calibration_eval.metrics.brier import brier_score, brier_skill_score
from llm_calibration_eval.metrics.ece import expected_calibration_error, maximum_calibration_error
from llm_calibration_eval.metrics.reliability import ReliabilityBin, reliability_diagram_data


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_confidence_all_correct(self) -> None:
        bs = brier_score([1.0, 1.0, 1.0], [1, 1, 1])
        assert bs == pytest.approx(0.0)

    def test_perfect_confidence_all_wrong(self) -> None:
        bs = brier_score([1.0, 1.0, 1.0], [0, 0, 0])
        assert bs == pytest.approx(1.0)

    def test_uniform_confidence_half_correct(self) -> None:
        # confidence=0.5 for all, half correct → BS = 0.25
        bs = brier_score([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
        assert bs == pytest.approx(0.25)

    def test_known_values(self) -> None:
        # (0.9 - 1)^2 + (0.3 - 0)^2 = 0.01 + 0.09 = 0.10; mean = 0.05
        bs = brier_score([0.9, 0.3], [1, 0])
        assert bs == pytest.approx(0.05)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            brier_score([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            brier_score([0.5, 0.5], [1])

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            brier_score([1.5], [1])

    def test_non_binary_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score([0.5], [2])

    def test_single_sample(self) -> None:
        bs = brier_score([0.8], [1])
        assert bs == pytest.approx(0.04)


class TestBrierSkillScore:
    def test_perfect_forecaster(self) -> None:
        # Always predicts 1 when correct, 0 when wrong → BS=0 → BSS=1
        bss = brier_skill_score([1.0, 1.0, 0.0], [1, 1, 0])
        assert bss == pytest.approx(1.0)

    def test_climatological_baseline_returns_zero(self) -> None:
        outcomes = [1, 1, 0, 0]
        base_rate = 0.5
        bss = brier_skill_score([base_rate] * 4, outcomes)
        assert bss == pytest.approx(0.0)

    def test_worse_than_baseline_negative(self) -> None:
        # Systematically wrong predictions
        bss = brier_skill_score([0.0, 0.0, 1.0], [1, 1, 0])
        assert bss < 0


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


class TestECE:
    def test_perfect_calibration(self) -> None:
        # 10 bins, each with one sample at its midpoint
        confs = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        # outcomes set so accuracy == confidence in each bin
        # Each bin has 1 sample; accuracy = outcome in that bin
        # For ECE to be 0, outcome must equal confidence → not possible with
        # binary outcomes exactly, so we test near-zero instead.
        confs_simple = [0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2]
        outcomes_simple = [1, 1, 1, 0, 0, 0, 0, 1]
        ece = expected_calibration_error(confs_simple, outcomes_simple, n_bins=2)
        # bin [0,0.5): conf=0.2, acc=0.25 → |gap|=0.05, weight=0.5
        # bin [0.5,1]: conf=0.8, acc=0.75 → |gap|=0.05, weight=0.5
        # ECE = 0.05
        assert ece == pytest.approx(0.05, abs=0.01)

    def test_completely_overconfident(self) -> None:
        # Always 0.9 confidence, always wrong
        confs = [0.9] * 20
        outcomes = [0] * 20
        ece = expected_calibration_error(confs, outcomes)
        assert ece == pytest.approx(0.9, abs=0.05)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error([], [])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error([0.5], [1, 0])

    def test_non_negative(self) -> None:
        import random
        rng = random.Random(0)
        confs = [rng.random() for _ in range(100)]
        outcomes = [rng.randint(0, 1) for _ in range(100)]
        ece = expected_calibration_error(confs, outcomes)
        assert ece >= 0.0

    def test_n_bins_one(self) -> None:
        confs = [0.7, 0.8]
        outcomes = [1, 0]
        ece = expected_calibration_error(confs, outcomes, n_bins=1)
        # one bin: avg_conf=0.75, avg_acc=0.5 → ECE=0.25
        assert ece == pytest.approx(0.25)


class TestMCE:
    def test_basic(self) -> None:
        confs = [0.9] * 5 + [0.1] * 5
        outcomes = [0] * 5 + [1] * 5
        mce = maximum_calibration_error(confs, outcomes)
        assert mce > 0.0


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    def test_returns_bins(self) -> None:
        confs = [0.1, 0.2, 0.8, 0.9]
        outcomes = [0, 0, 1, 1]
        bins = reliability_diagram_data(confs, outcomes, n_bins=2)
        assert len(bins) == 2
        assert all(isinstance(b, ReliabilityBin) for b in bins)

    def test_bin_counts_correct(self) -> None:
        confs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        outcomes = [0, 1, 0, 1, 1, 0]
        bins = reliability_diagram_data(confs, outcomes, n_bins=2)
        total = sum(b.count for b in bins)
        assert total == 6

    def test_avg_confidence_in_range(self) -> None:
        import random
        rng = random.Random(42)
        confs = [rng.random() for _ in range(50)]
        outcomes = [rng.randint(0, 1) for _ in range(50)]
        bins = reliability_diagram_data(confs, outcomes)
        for b in bins:
            assert 0.0 <= b.avg_confidence <= 1.0
            assert 0.0 <= b.avg_accuracy <= 1.0

    def test_calibration_gap(self) -> None:
        # overconfident: high conf, all wrong
        b = ReliabilityBin(0.8, 0.9, 0.85, 0.2, 10)
        assert b.calibration_gap == pytest.approx(0.65)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            reliability_diagram_data([], [])

    def test_single_sample(self) -> None:
        bins = reliability_diagram_data([0.7], [1], n_bins=10)
        assert len(bins) == 1
        assert bins[0].avg_accuracy == pytest.approx(1.0)
        assert bins[0].avg_confidence == pytest.approx(0.7)
