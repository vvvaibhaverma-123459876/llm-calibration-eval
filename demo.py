"""Demo: Run calibration evaluation with mock provider."""
from llm_calibration_eval.providers.mock_provider import MockProvider
from llm_calibration_eval.evaluator import Evaluator
from llm_calibration_eval.datasets.trivia import load_dataset

print("=== LLM Calibration Evaluator Demo ===\n")
provider = MockProvider(seed=42)
dataset = load_dataset("trivia")[:20]
evaluator = Evaluator(provider)
result = evaluator.run(dataset)

print(result.summary_table())
print("\nDemo complete.")
