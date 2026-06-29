"""Mock LLM provider for deterministic testing without real API keys."""

from __future__ import annotations

import random
from typing import Callable

from llm_calibration_eval.providers.base import (
    CalibrationSample,
    LLMProvider,
    ProviderResponse,
)


class MockProvider(LLMProvider):
    """Deterministic mock provider for unit tests and offline demos.

    The mock simulates a model with configurable accuracy and calibration
    behaviour.  Three behaviour modes are available:

    * **perfect** — always answers correctly with confidence 1.0.
    * **random** — answers randomly with calibrated confidence equal to
      the actual accuracy rate.
    * **overconfident** — often wrong but reports high confidence
      (useful for testing calibration metrics).

    Args:
        model: Model identifier string (e.g. ``"mock-perfect"``).
        accuracy: Probability [0, 1] of returning the correct answer.
        confidence_fn: Optional callable ``(is_correct: bool) -> float``
            that returns the stated confidence for a response.
            Defaults to a mildly miscalibrated function.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: str = "mock-v1",
        accuracy: float = 0.8,
        confidence_fn: Callable[[bool], float] | None = None,
        seed: int | None = 42,
    ) -> None:
        super().__init__(model)
        self._accuracy = accuracy
        self._rng = random.Random(seed)

        if confidence_fn is not None:
            self._confidence_fn = confidence_fn
        else:
            self._confidence_fn = self._default_confidence_fn

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def query(self, sample: CalibrationSample) -> ProviderResponse:
        """Return a mock response for *sample*.

        Args:
            sample: The calibration question.

        Returns:
            A :class:`ProviderResponse` with a deterministic answer and
            confidence drawn from ``confidence_fn``.
        """
        is_correct = self._rng.random() < self._accuracy
        answer = sample.correct_answer if is_correct else f"Wrong({sample.correct_answer})"
        confidence = self._confidence_fn(is_correct)

        raw = f"ANSWER: {answer}\nCONFIDENCE: {confidence:.4f}"
        return ProviderResponse(
            answer=answer,
            confidence=confidence,
            raw_text=raw,
            model=self._model,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_confidence_fn(self, is_correct: bool) -> float:
        """Slightly over-confident confidence function.

        Correct answers get confidence in [0.65, 0.99].
        Wrong answers get confidence in [0.35, 0.75].
        """
        if is_correct:
            return round(self._rng.uniform(0.65, 0.99), 4)
        return round(self._rng.uniform(0.35, 0.75), 4)

    @classmethod
    def perfectly_calibrated(cls, accuracy: float = 0.7, seed: int = 0) -> "MockProvider":
        """Factory: returns a perfectly-calibrated mock.

        The confidence always equals the accuracy rate.

        Args:
            accuracy: The fixed accuracy / confidence value.
            seed: Random seed.

        Returns:
            A :class:`MockProvider` instance.
        """
        def confidence_fn(is_correct: bool) -> float:  # noqa: ARG001
            return accuracy

        return cls(
            model=f"mock-calibrated-{accuracy}",
            accuracy=accuracy,
            confidence_fn=confidence_fn,
            seed=seed,
        )

    @classmethod
    def overconfident(cls, accuracy: float = 0.5, seed: int = 1) -> "MockProvider":
        """Factory: returns a systematically over-confident mock.

        Reports confidence ~0.9 regardless of actual accuracy.

        Args:
            accuracy: Actual accuracy of the mock.
            seed: Random seed.

        Returns:
            A :class:`MockProvider` instance.
        """
        rng = random.Random(seed + 999)

        def confidence_fn(is_correct: bool) -> float:  # noqa: ARG001
            return round(rng.uniform(0.85, 0.99), 4)

        return cls(
            model="mock-overconfident",
            accuracy=accuracy,
            confidence_fn=confidence_fn,
            seed=seed,
        )
