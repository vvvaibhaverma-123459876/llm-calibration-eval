"""Tests for LLM provider implementations."""

from __future__ import annotations

import pytest

from llm_calibration_eval.providers.base import (
    CalibrationSample,
    LLMProvider,
    ProviderResponse,
)
from llm_calibration_eval.providers.mock_provider import MockProvider


# ---------------------------------------------------------------------------
# ProviderResponse
# ---------------------------------------------------------------------------


class TestProviderResponse:
    def test_valid_creation(self) -> None:
        r = ProviderResponse(
            answer="Paris", confidence=0.9, raw_text="ANSWER: Paris\nCONFIDENCE: 0.9",
            model="mock"
        )
        assert r.answer == "Paris"
        assert r.confidence == 0.9

    def test_confidence_boundary_zero(self) -> None:
        r = ProviderResponse(answer="x", confidence=0.0, raw_text="", model="m")
        assert r.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        r = ProviderResponse(answer="x", confidence=1.0, raw_text="", model="m")
        assert r.confidence == 1.0

    def test_confidence_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            ProviderResponse(answer="x", confidence=1.5, raw_text="", model="m")

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            ProviderResponse(answer="x", confidence=-0.1, raw_text="", model="m")


# ---------------------------------------------------------------------------
# CalibrationSample
# ---------------------------------------------------------------------------


class TestCalibrationSample:
    def test_default_category(self) -> None:
        s = CalibrationSample("What?", "Answer")
        assert s.category == "general"

    def test_custom_category(self) -> None:
        s = CalibrationSample("Q", "A", "science")
        assert s.category == "science"


# ---------------------------------------------------------------------------
# LLMProvider (via build_prompt / parse_response)
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_build_prompt_contains_question(self) -> None:
        prompt = LLMProvider.build_prompt("What is 2+2?")
        assert "What is 2+2?" in prompt
        assert "ANSWER" in prompt
        assert "CONFIDENCE" in prompt

    def test_parse_response_normal(self) -> None:
        raw = "ANSWER: Paris\nCONFIDENCE: 0.85"
        answer, confidence = LLMProvider.parse_response(raw)
        assert answer == "Paris"
        assert confidence == pytest.approx(0.85)

    def test_parse_response_missing_confidence_defaults(self) -> None:
        raw = "ANSWER: Tokyo"
        answer, confidence = LLMProvider.parse_response(raw)
        assert answer == "Tokyo"
        assert 0.0 <= confidence <= 1.0

    def test_parse_response_missing_answer_defaults(self) -> None:
        raw = "CONFIDENCE: 0.7"
        answer, confidence = LLMProvider.parse_response(raw)
        assert answer != ""  # some default
        assert confidence == pytest.approx(0.7)

    def test_parse_response_empty_string_defaults(self) -> None:
        answer, confidence = LLMProvider.parse_response("")
        assert isinstance(answer, str)
        assert 0.0 <= confidence <= 1.0

    def test_parse_response_confidence_clamped(self) -> None:
        raw = "ANSWER: Foo\nCONFIDENCE: 2.0"
        _, confidence = LLMProvider.parse_response(raw)
        assert confidence <= 1.0

    def test_parse_response_extra_whitespace(self) -> None:
        raw = "ANSWER:   Berlin  \nCONFIDENCE:  0.92  "
        answer, confidence = LLMProvider.parse_response(raw)
        assert answer == "Berlin"
        assert confidence == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def _sample(self) -> CalibrationSample:
        return CalibrationSample("What is 2+2?", "4", "math")

    def test_correct_answer_when_seed_fixed(self) -> None:
        provider = MockProvider(accuracy=1.0, seed=0)
        response = provider.query(self._sample())
        assert response.answer == "4"
        assert 0.0 <= response.confidence <= 1.0

    def test_wrong_answer_when_accuracy_zero(self) -> None:
        provider = MockProvider(accuracy=0.0, seed=0)
        response = provider.query(self._sample())
        assert response.answer != "4"

    def test_model_name_stored(self) -> None:
        provider = MockProvider(model="my-mock")
        assert provider.model == "my-mock"

    def test_response_has_required_fields(self) -> None:
        provider = MockProvider(seed=1)
        response = provider.query(self._sample())
        assert isinstance(response.answer, str)
        assert isinstance(response.confidence, float)
        assert isinstance(response.raw_text, str)
        assert isinstance(response.model, str)

    def test_deterministic_with_same_seed(self) -> None:
        s = self._sample()
        r1 = MockProvider(accuracy=0.6, seed=7).query(s)
        r2 = MockProvider(accuracy=0.6, seed=7).query(s)
        assert r1.answer == r2.answer
        assert r1.confidence == r2.confidence

    def test_perfectly_calibrated_factory(self) -> None:
        provider = MockProvider.perfectly_calibrated(accuracy=0.7, seed=0)
        assert "calibrated" in provider.model

    def test_perfectly_calibrated_confidence_equals_accuracy(self) -> None:
        accuracy = 0.7
        provider = MockProvider.perfectly_calibrated(accuracy=accuracy, seed=0)
        for _ in range(20):
            r = provider.query(self._sample())
            assert r.confidence == pytest.approx(accuracy)

    def test_overconfident_factory(self) -> None:
        provider = MockProvider.overconfident(accuracy=0.3, seed=0)
        assert "overconfident" in provider.model

    def test_overconfident_high_confidence(self) -> None:
        provider = MockProvider.overconfident(accuracy=0.3, seed=0)
        confidences = [provider.query(self._sample()).confidence for _ in range(30)]
        assert all(c >= 0.8 for c in confidences)

    def test_multiple_samples(self) -> None:
        provider = MockProvider(accuracy=0.8, seed=42)
        samples = [
            CalibrationSample(f"Q{i}", f"A{i}") for i in range(20)
        ]
        responses = [provider.query(s) for s in samples]
        assert len(responses) == 20

    def test_confidence_always_in_range(self) -> None:
        provider = MockProvider(seed=0)
        for _ in range(50):
            r = provider.query(self._sample())
            assert 0.0 <= r.confidence <= 1.0

    def test_custom_confidence_fn(self) -> None:
        provider = MockProvider(confidence_fn=lambda _: 0.5, seed=0)
        r = provider.query(self._sample())
        assert r.confidence == pytest.approx(0.5)
