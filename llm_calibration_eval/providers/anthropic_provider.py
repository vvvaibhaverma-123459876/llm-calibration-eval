"""Anthropic provider (claude-haiku-4-5 by default)."""

from __future__ import annotations

import os

from llm_calibration_eval.providers.base import (
    CalibrationSample,
    LLMProvider,
    ProviderResponse,
)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider.

    Reads ``ANTHROPIC_API_KEY`` from the environment unless an explicit
    ``api_key`` is supplied.

    Args:
        model: Anthropic model name (default ``"claude-haiku-4-5"``).
        api_key: API key; defaults to ``ANTHROPIC_API_KEY`` env var.
        max_tokens: Maximum tokens in the response (default 256).
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        super().__init__(model)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens

    def query(self, sample: CalibrationSample) -> ProviderResponse:
        """Call the Anthropic Messages API.

        Args:
            sample: Calibration question with a known correct answer.

        Returns:
            :class:`ProviderResponse` with the model's answer and
            stated confidence.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
            anthropic.APIError: On API errors.
        """
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)
        prompt = self.build_prompt(sample.question)

        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from the first content block.
        raw = ""
        for block in message.content:
            if block.type == "text":
                raw = block.text
                break

        answer, confidence = self.parse_response(raw)

        usage = message.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0

        return ProviderResponse(
            answer=answer,
            confidence=confidence,
            raw_text=raw,
            model=message.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
