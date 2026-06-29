"""Reliability diagram data computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBin:
    """Data for a single bin in a reliability diagram.

    Attributes:
        bin_lower: Lower edge of confidence bin.
        bin_upper: Upper edge of confidence bin.
        avg_confidence: Mean stated confidence in this bin.
        avg_accuracy: Mean accuracy (fraction correct) in this bin.
        count: Number of samples in this bin.
    """

    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int

    @property
    def calibration_gap(self) -> float:
        """Signed gap: confidence minus accuracy (positive = overconfident)."""
        return self.avg_confidence - self.avg_accuracy


def reliability_diagram_data(
    confidences: list[float] | np.ndarray,
    outcomes: list[int] | np.ndarray,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Compute binned data for a reliability diagram.

    Args:
        confidences: Sequence of confidence values in [0, 1].
        outcomes: Sequence of binary outcomes (1 = correct, 0 = incorrect).
        n_bins: Number of equally-spaced bins.

    Returns:
        List of :class:`ReliabilityBin` objects, one per non-empty bin.

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

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        bins.append(
            ReliabilityBin(
                bin_lower=float(low),
                bin_upper=float(high),
                avg_confidence=float(p[mask].mean()),
                avg_accuracy=float(o[mask].mean()),
                count=n_bin,
            )
        )

    return bins
