"""Command-line interface for llm-calibration-eval.

Usage::

    python -m llm_calibration_eval run --provider mock --dataset trivia
    python -m llm_calibration_eval run --provider openai --dataset trivia
    python -m llm_calibration_eval run --provider anthropic --dataset trivia --plot
"""

from __future__ import annotations

import argparse
import sys


def _build_provider(provider_name: str, model: str | None) -> "LLMProvider":  # type: ignore[name-defined]  # noqa: F821
    from llm_calibration_eval.providers import (
        AnthropicProvider,
        MockProvider,
        OpenAIProvider,
    )

    if provider_name == "mock":
        return MockProvider(model=model or "mock-v1")
    if provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini")
    if provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-haiku-4-5")
    raise ValueError(
        f"Unknown provider '{provider_name}'. "
        "Choose from: mock, openai, anthropic"
    )


def _plot_reliability(result: "EvaluationResult") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Plot a reliability diagram and display it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. pip install matplotlib")
        return

    bins = result.reliability_bins
    if not bins:
        print("No reliability bins to plot.")
        return

    x = [b.avg_confidence for b in bins]
    y = [b.avg_accuracy for b in bins]
    sizes = [b.count * 3 for b in bins]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    ax.scatter(x, y, s=sizes, alpha=0.8, label="Observed")
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(f"Reliability Diagram — {result.provider_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("reliability_diagram.png", dpi=150)
    print("Reliability diagram saved to reliability_diagram.png")
    plt.show()


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    from llm_calibration_eval.datasets import load_dataset
    from llm_calibration_eval.evaluator import Evaluator

    try:
        provider = _build_provider(args.provider, args.model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        samples = load_dataset(args.dataset)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.n is not None and args.n > 0:
        samples = samples[: args.n]

    print(f"Running calibration eval: {args.provider} on {len(samples)} samples...")
    evaluator = Evaluator(provider, verbose=args.verbose)
    result = evaluator.run(samples, dataset_name=args.dataset)

    print()
    print(result.summary_table())

    if args.plot:
        _plot_reliability(result)

    return 0


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``llm-calibration-eval`` CLI."""
    parser = argparse.ArgumentParser(
        prog="llm-calibration-eval",
        description="Measure LLM confidence calibration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run", help="Run a calibration evaluation."
    )
    run_parser.add_argument(
        "--provider",
        required=True,
        choices=["mock", "openai", "anthropic"],
        help="LLM provider to evaluate.",
    )
    run_parser.add_argument(
        "--dataset",
        default="trivia",
        help="Dataset to use (default: trivia).",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Override the default model for the provider.",
    )
    run_parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N samples.",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a reliability diagram after evaluation.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each sample.",
    )

    parsed = parser.parse_args(argv)

    if parsed.command == "run":
        sys.exit(cmd_run(parsed))


if __name__ == "__main__":
    main()
