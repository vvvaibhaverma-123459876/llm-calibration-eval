"""LLM Calibration Evaluator.

Measures how well-calibrated LLMs are — do they say "90% confident"
when they're right 90% of the time?
"""

__version__ = "0.1.0"
__all__ = ["Evaluator", "EvaluationResult"]

from llm_calibration_eval.evaluator import Evaluator, EvaluationResult
