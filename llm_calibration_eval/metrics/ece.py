"""Expected Calibration Error (ECE) implementation."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidences: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE partitions predictions into *n_bins* equally-spaced confidence bins
    and computes the weighted average gap between mean confidence and mean
    accuracy within each bin:

    .. math::

        \\text{ECE} = \\sum_{b=1}^{B} \\frac{|B_b|}{N}
                      \\left| \\overline{p}_b - \\overline{o}_b \\right|

    Args:
        confidences: Sequence of confidence values in [0, 1].
        outcomes: Sequence of binary outcomes (1 = correct, 0 = incorrect).
        n_bins: Number of equally-spaced bins (default 10).

    Returns:
        ECE as a non-negative float (lower is better; 0 is perfect).

    Raises:
        ValueError: If inputs are invalid.
    """
    p = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)

    if len(p) == 0:
        raise ValueError("confidences and outcomes must not be empty.")
    if len(p) != len(o):
        raise ValueError(
            f"confidences and outcomes must have the same length "
            f"(got {len(p)} and {len(o)})."
        )
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("All confidence values must be in [0, 1].")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    n = len(p)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include right edge in last bin.
        if i < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        avg_conf = float(p[mask].mean())
        avg_acc = float(o[mask].mean())
        ece += (n_bin / n) * abs(avg_conf - avg_acc)

    return float(ece)


def maximum_calibration_error(
    confidences: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE) — the worst-bin calibration gap.

    Args:
        confidences: Sequence of confidence values in [0, 1].
        outcomes: Sequence of binary outcomes.
        n_bins: Number of bins.

    Returns:
        MCE as a non-negative float.
    """
    p = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)

    if len(p) == 0:
        raise ValueError("confidences and outcomes must not be empty.")
    if len(p) != len(o):
        raise ValueError(
            f"confidences and outcomes must have the same length "
            f"(got {len(p)} and {len(o)})."
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)

        if mask.sum() == 0:
            continue

        gap = abs(float(p[mask].mean()) - float(o[mask].mean()))
        mce = max(mce, gap)

    return float(mce)
