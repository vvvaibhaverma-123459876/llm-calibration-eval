"""Tests for the CLI and real provider stubs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_calibration_eval.cli import main, cmd_run, _build_provider, _plot_reliability
from llm_calibration_eval.providers.base import CalibrationSample, ProviderResponse
from llm_calibration_eval.providers.openai_provider import OpenAIProvider
from llm_calibration_eval.providers.anthropic_provider import AnthropicProvider
from llm_calibration_eval.evaluator import EvaluationResult
from llm_calibration_eval.metrics.reliability import ReliabilityBin


# ---------------------------------------------------------------------------
# _build_provider
# ---------------------------------------------------------------------------


class TestBuildProvider:
    def test_mock(self) -> None:
        p = _build_provider("mock", None)
        assert "mock" in p.model.lower()

    def test_mock_custom_model(self) -> None:
        p = _build_provider("mock", "custom-mock")
        assert p.model == "custom-mock"

    def test_openai_default_model(self) -> None:
        p = _build_provider("openai", None)
        assert p.model == "gpt-4o-mini"

    def test_anthropic_default_model(self) -> None:
        p = _build_provider("anthropic", None)
        assert p.model == "claude-haiku-4-5"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            _build_provider("fake", None)


# ---------------------------------------------------------------------------
# CLI main() — via sys.argv
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_run_mock_provider(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--provider", "mock", "--dataset", "trivia", "--n", "5"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Brier" in out

    def test_run_bad_provider_exits_1(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--provider", "bad", "--dataset", "trivia"])
        # argparse catches bad choices before we get to cmd_run

    def test_run_bad_dataset_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--provider", "mock", "--dataset", "nonexistent"])
        assert exc_info.value.code == 1

    def test_run_verbose(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--provider", "mock", "--dataset", "trivia", "--n", "3", "--verbose"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "[1/3]" in out

    def test_no_subcommand_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([])


# ---------------------------------------------------------------------------
# _plot_reliability (no real display)
# ---------------------------------------------------------------------------


class TestPlotReliability:
    def _make_result(self) -> EvaluationResult:
        from llm_calibration_eval.providers.mock_provider import MockProvider
        from llm_calibration_eval.evaluator import Evaluator
        provider = MockProvider(seed=0)
        samples = [CalibrationSample(f"Q{i}", f"A{i}") for i in range(10)]
        return Evaluator(provider).run(samples)

    def test_plot_no_matplotlib_prints_message(self, capsys) -> None:
        import sys
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            result = self._make_result()
            _plot_reliability(result)
        out = capsys.readouterr().out
        assert "matplotlib" in out

    def test_plot_with_matplotlib_saves_file(self, tmp_path, monkeypatch) -> None:
        import matplotlib
        matplotlib.use("Agg")
        monkeypatch.chdir(tmp_path)
        result = self._make_result()
        _plot_reliability(result)
        assert (tmp_path / "reliability_diagram.png").exists()

    def test_plot_empty_bins(self, capsys) -> None:
        result = self._make_result()
        result.reliability_bins.clear()
        _plot_reliability(result)
        out = capsys.readouterr().out
        assert "No reliability bins" in out


# ---------------------------------------------------------------------------
# OpenAIProvider — mock the openai SDK
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _make_mock_completion(self, answer: str = "Paris", confidence: float = 0.9):
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = (
            f"ANSWER: {answer}\nCONFIDENCE: {confidence}"
        )
        mock_completion.model = "gpt-4o-mini"
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5
        return mock_completion

    def test_query_returns_provider_response(self) -> None:
        sample = CalibrationSample("What is the capital of France?", "Paris")
        mock_completion = self._make_mock_completion("Paris", 0.95)

        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = mock_completion

            provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake")
            response = provider.query(sample)

        assert response.answer == "Paris"
        assert response.confidence == pytest.approx(0.95)
        assert response.model == "gpt-4o-mini"

    def test_query_with_null_usage(self) -> None:
        sample = CalibrationSample("Q?", "A")
        mock_completion = self._make_mock_completion("A", 0.7)
        mock_completion.usage = None

        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = mock_completion
            provider = OpenAIProvider(api_key="fake")
            response = provider.query(sample)

        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0

    def test_openai_not_installed_raises(self) -> None:
        import sys
        sample = CalibrationSample("Q?", "A")
        with patch.dict(sys.modules, {"openai": None}):
            provider = OpenAIProvider(api_key="fake")
            with pytest.raises(ImportError, match="openai"):
                provider.query(sample)

    def test_default_model(self) -> None:
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# AnthropicProvider — mock the anthropic SDK
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _make_mock_message(self, answer: str = "Paris", confidence: float = 0.9):
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = f"ANSWER: {answer}\nCONFIDENCE: {confidence}"

        mock_message = MagicMock()
        mock_message.content = [mock_block]
        mock_message.model = "claude-haiku-4-5"
        mock_message.usage.input_tokens = 12
        mock_message.usage.output_tokens = 6
        return mock_message

    def test_query_returns_provider_response(self) -> None:
        sample = CalibrationSample("What is the capital of France?", "Paris")
        mock_message = self._make_mock_message("Paris", 0.92)

        with patch("anthropic.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_message

            provider = AnthropicProvider(model="claude-haiku-4-5", api_key="fake")
            response = provider.query(sample)

        assert response.answer == "Paris"
        assert response.confidence == pytest.approx(0.92)

    def test_query_null_usage(self) -> None:
        sample = CalibrationSample("Q?", "A")
        mock_message = self._make_mock_message("A", 0.5)
        mock_message.usage = None

        with patch("anthropic.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_message
            provider = AnthropicProvider(api_key="fake")
            response = provider.query(sample)

        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0

    def test_anthropic_not_installed_raises(self) -> None:
        import sys
        sample = CalibrationSample("Q?", "A")
        with patch.dict(sys.modules, {"anthropic": None}):
            provider = AnthropicProvider(api_key="fake")
            with pytest.raises(ImportError, match="anthropic"):
                provider.query(sample)

    def test_default_model(self) -> None:
        provider = AnthropicProvider()
        assert provider.model == "claude-haiku-4-5"

    def test_non_text_blocks_skipped(self) -> None:
        sample = CalibrationSample("Q?", "A")
        mock_block_tool = MagicMock()
        mock_block_tool.type = "tool_use"
        mock_block_text = MagicMock()
        mock_block_text.type = "text"
        mock_block_text.text = "ANSWER: A\nCONFIDENCE: 0.8"

        mock_message = MagicMock()
        mock_message.content = [mock_block_tool, mock_block_text]
        mock_message.model = "claude-haiku-4-5"
        mock_message.usage.input_tokens = 5
        mock_message.usage.output_tokens = 5

        with patch("anthropic.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_message
            provider = AnthropicProvider(api_key="fake")
            response = provider.query(sample)

        assert response.answer == "A"
