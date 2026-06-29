"""Brier score implementation.

The Brier score is the mean squared error between stated confidence and
the binary correctness outcome.  A perfectly calibrated model achieves
a Brier score of 0; always-wrong + fully-confident scores 1.
"""

from __future__ import annotations

import numpy as np


def brier_score(
    confidences: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
) -> float:
    """Compute the Brier score for a set of predictions.

    .. math::

        BS = \\frac{1}{N} \\sum_{i=1}^{N} (p_i - o_i)^2

    where :math:`p_i` is the stated confidence and :math:`o_i \\in \\{0, 1\\}`
    is the binary correctness outcome (1 = correct).

    Args:
        confidences: Sequence of confidence values in [0, 1].
        outcomes: Sequence of binary outcomes (1 = correct, 0 = incorrect).
            Must be the same length as *confidences*.

    Returns:
        Mean squared error (Brier score) as a float in [0, 1].

    Raises:
        ValueError: If *confidences* and *outcomes* have different lengths
            or are empty, or if any confidence is outside [0, 1].

    Examples:
        >>> brier_score([0.9, 0.8, 0.3], [1, 1, 0])
        0.02...
    """
    p = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)

    if p.ndim != 1 or o.ndim != 1:
        raise ValueError("confidences and outcomes must be 1-D sequences.")
    if len(p) == 0:
        raise ValueError("confidences and outcomes must not be empty.")
    if len(p) != len(o):
        raise ValueError(
            f"confidences and outcomes must have the same length "
            f"(got {len(p)} and {len(o)})."
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("All confidence values must be in [0, 1].")
    if not np.all(np.isin(o, [0.0, 1.0])):
        raise ValueError("All outcome values must be 0 or 1.")

    return float(np.mean((p - o) ** 2))


def brier_skill_score(
    confidences: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
) -> float:
    """Brier Skill Score relative to a no-skill (climatological) baseline.

    .. math::

        BSS = 1 - \\frac{BS}{BS_{\\text{ref}}}

    where :math:`BS_{\\text{ref}}` is the Brier score of a forecaster who
    always predicts the base-rate probability.  BSS = 1 is perfect;
    BSS = 0 matches the climatological baseline; BSS < 0 is worse.

    Args:
        confidences: Sequence of confidence values in [0, 1].
        outcomes: Sequence of binary outcomes (1 = correct, 0 = incorrect).

    Returns:
        Brier Skill Score as a float.
    """
    o = np.asarray(outcomes, dtype=float)
    base_rate = float(np.mean(o))
    bs_ref = brier_score([base_rate] * len(o), outcomes)
    bs = brier_score(confidences, outcomes)
    if bs_ref == 0.0:
        return 0.0
    return float(1.0 - bs / bs_ref)
