"""Main evaluation orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from llm_calibration_eval.metrics.brier import brier_score
from llm_calibration_eval.metrics.ece import expected_calibration_error
from llm_calibration_eval.metrics.reliability import ReliabilityBin, reliability_diagram_data
from llm_calibration_eval.providers.base import CalibrationSample, LLMProvider, ProviderResponse


@dataclass
class EvaluationResult:
    """Results from a calibration evaluation run.

    Attributes:
        provider_name: Name/model of the LLM provider used.
        dataset_name: Name of the dataset evaluated.
        n_samples: Total number of questions evaluated.
        n_correct: Number of questions answered correctly.
        accuracy: Fraction of correct answers (n_correct / n_samples).
        brier_score: Mean squared error between confidences and outcomes.
        ece: Expected Calibration Error.
        mean_confidence: Average stated confidence across all samples.
        reliability_bins: Binned data for the reliability diagram.
        responses: Raw list of (sample, response, is_correct) triples.
        elapsed_seconds: Wall-clock time for the evaluation run.
    """

    provider_name: str
    dataset_name: str
    n_samples: int
    n_correct: int
    accuracy: float
    brier_score: float
    ece: float
    mean_confidence: float
    reliability_bins: list[ReliabilityBin]
    responses: list[tuple[CalibrationSample, ProviderResponse, bool]] = field(
        default_factory=list, repr=False
    )
    elapsed_seconds: float = 0.0

    def summary_table(self) -> str:
        """Return a formatted text table of key metrics.

        Returns:
            Multi-line string with the evaluation summary.
        """
        lines = [
            f"{'='*52}",
            f"  LLM Calibration Evaluation Summary",
            f"{'='*52}",
            f"  Provider       : {self.provider_name}",
            f"  Dataset        : {self.dataset_name}",
            f"  Samples        : {self.n_samples}",
            f"{'─'*52}",
            f"  Accuracy       : {self.accuracy:.3f} ({self.n_correct}/{self.n_samples})",
            f"  Mean Confidence: {self.mean_confidence:.3f}",
            f"  Brier Score    : {self.brier_score:.4f}  (lower = better)",
            f"  ECE            : {self.ece:.4f}  (lower = better)",
            f"{'─'*52}",
            f"  Elapsed        : {self.elapsed_seconds:.1f}s",
            f"{'='*52}",
        ]
        return "\n".join(lines)


def _default_judge(response: ProviderResponse, sample: CalibrationSample) -> bool:
    """Case-insensitive exact match judge."""
    return response.answer.strip().lower() == sample.correct_answer.strip().lower()


class Evaluator:
    """Orchestrates calibration evaluation of an LLM provider.

    Example::

        from llm_calibration_eval import Evaluator
        from llm_calibration_eval.providers import MockProvider
        from llm_calibration_eval.datasets import load_dataset

        provider = MockProvider.perfectly_calibrated(accuracy=0.8)
        evaluator = Evaluator(provider)
        result = evaluator.run(load_dataset("trivia"))
        print(result.summary_table())

    Args:
        provider: An :class:`LLMProvider` instance to evaluate.
        judge: Optional callable ``(response, sample) -> bool`` used to
            decide if the model's answer is correct.  Defaults to a
            case-insensitive exact string match.
        n_bins: Number of bins for ECE / reliability diagram (default 10).
        verbose: If *True*, print progress to stdout.
    """

    def __init__(
        self,
        provider: LLMProvider,
        judge: Callable[[ProviderResponse, CalibrationSample], bool] | None = None,
        n_bins: int = 10,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        self._judge = judge or _default_judge
        self._n_bins = n_bins
        self._verbose = verbose

    def run(
        self,
        samples: list[CalibrationSample],
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """Run the evaluation against a list of calibration samples.

        Args:
            samples: Questions with known answers.
            dataset_name: Label for the dataset (used in reporting).

        Returns:
            :class:`EvaluationResult` with all metrics populated.

        Raises:
            ValueError: If *samples* is empty.
        """
        if not samples:
            raise ValueError("samples must not be empty.")

        start = time.monotonic()
        responses: list[tuple[CalibrationSample, ProviderResponse, bool]] = []
        confidences: list[float] = []
        outcomes: list[int] = []

        for i, sample in enumerate(samples):
            if self._verbose:
                print(f"  [{i+1}/{len(samples)}] {sample.question[:60]}...")

            response = self._provider.query(sample)
            is_correct = self._judge(response, sample)
            responses.append((sample, response, is_correct))
            confidences.append(response.confidence)
            outcomes.append(int(is_correct))

        elapsed = time.monotonic() - start

        n_correct = sum(outcomes)
        accuracy = n_correct / len(samples)

        return EvaluationResult(
            provider_name=self._provider.model,
            dataset_name=dataset_name,
            n_samples=len(samples),
            n_correct=n_correct,
            accuracy=accuracy,
            brier_score=brier_score(confidences, outcomes),
            ece=expected_calibration_error(confidences, outcomes, n_bins=self._n_bins),
            mean_confidence=sum(confidences) / len(confidences),
            reliability_bins=reliability_diagram_data(confidences, outcomes, n_bins=self._n_bins),
            responses=responses,
            elapsed_seconds=elapsed,
        )
