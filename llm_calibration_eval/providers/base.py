"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class ProviderResponse:
    """Response from an LLM provider for a calibration query.

    Attributes:
        answer: The predicted answer (e.g. "Paris").
        confidence: Stated confidence in [0, 1].
        raw_text: Full raw response text from the model.
        model: Model identifier that produced the response.
        prompt_tokens: Number of prompt tokens used (if available).
        completion_tokens: Number of completion tokens used (if available).
    """

    answer: str
    confidence: float
    raw_text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class CalibrationSample:
    """A single question/answer pair for calibration evaluation.

    Attributes:
        question: The question posed to the model.
        correct_answer: The ground-truth correct answer.
        category: Optional topic category (e.g. "geography").
    """

    question: str
    correct_answer: str
    category: str = "general"


class LLMProvider(abc.ABC):
    """Abstract base class that all LLM providers must implement.

    Subclasses must override :meth:`query` to call the underlying model
    and return a :class:`ProviderResponse` with the model's stated answer
    and confidence score.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        """The model identifier used by this provider."""
        return self._model

    @abc.abstractmethod
    def query(self, sample: CalibrationSample) -> ProviderResponse:
        """Query the LLM with a calibration sample.

        The prompt must ask the model for:
        1. A direct answer to the question.
        2. A confidence score between 0 and 1.

        Args:
            sample: The question and expected answer.

        Returns:
            A :class:`ProviderResponse` containing the model's answer
            and stated confidence.
        """

    # ------------------------------------------------------------------
    # Shared prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(question: str) -> str:
        """Build the standard calibration prompt for a question.

        Args:
            question: The question to ask.

        Returns:
            A prompt string that instructs the model to reply with an
            answer and a numeric confidence score.
        """
        return (
            f"Answer the following question as accurately as possible.\n\n"
            f"Question: {question}\n\n"
            "Respond in exactly this format (two lines, nothing else):\n"
            "ANSWER: <your answer>\n"
            "CONFIDENCE: <number between 0 and 1>\n\n"
            "The CONFIDENCE value should reflect how likely your answer is correct. "
            "Use 1.0 if you are certain, 0.5 if you are guessing, etc."
        )

    @staticmethod
    def parse_response(raw: str) -> tuple[str, float]:
        """Parse model output into (answer, confidence).

        Expects the model to have responded with lines like::

            ANSWER: Paris
            CONFIDENCE: 0.95

        Args:
            raw: Raw text from the model.

        Returns:
            Tuple of (answer_string, confidence_float).

        Raises:
            ValueError: If the response cannot be parsed.
        """
        answer = ""
        confidence = 0.5  # sane default if missing

        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                answer = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                raw_conf = line.split(":", 1)[1].strip()
                try:
                    confidence = float(raw_conf)
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5

        if not answer:
            # Fallback: use the entire response trimmed.
            answer = raw.strip()

        return answer, confidence

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model!r})"
