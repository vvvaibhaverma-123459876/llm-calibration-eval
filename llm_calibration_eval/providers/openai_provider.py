"""OpenAI provider (gpt-4o-mini by default)."""

from __future__ import annotations

import os

from llm_calibration_eval.providers.base import (
    CalibrationSample,
    LLMProvider,
    ProviderResponse,
)


class OpenAIProvider(LLMProvider):
    """OpenAI chat completion provider.

    Reads ``OPENAI_API_KEY`` from the environment unless an explicit
    ``api_key`` is supplied.

    Args:
        model: OpenAI model name (default ``"gpt-4o-mini"``).
        api_key: API key; defaults to ``OPENAI_API_KEY`` env var.
        temperature: Sampling temperature (default ``0.0`` for
            determinism).
        max_tokens: Maximum tokens in the completion (default 256).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        super().__init__(model)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def query(self, sample: CalibrationSample) -> ProviderResponse:
        """Call the OpenAI chat completions API.

        Args:
            sample: Calibration question with a known correct answer.

        Returns:
            :class:`ProviderResponse` with the model's answer and
            stated confidence.

        Raises:
            ImportError: If the ``openai`` package is not installed.
            openai.OpenAIError: On API errors.
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key)
        prompt = self.build_prompt(sample.question)

        completion = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = completion.choices[0].message.content or ""
        answer, confidence = self.parse_response(raw)

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ProviderResponse(
            answer=answer,
            confidence=confidence,
            raw_text=raw,
            model=completion.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
