"""Temperature scaling for post-hoc LLM calibration.

Temperature scaling divides log-odds by a learned scalar T before applying
the sigmoid.  T > 1 softens probabilities (reduces overconfidence); T < 1
sharpens them.
"""

from __future__ import annotations

import math

import numpy as np


def _logit(p: float) -> float:
    """Numerically-stable logit (log-odds)."""
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class TemperatureScaling:
    """Post-hoc calibration via temperature scaling.

    Learns a scalar temperature T such that
    ``p_calibrated = sigmoid(logit(p) / T)`` minimises the negative
    log-likelihood on a held-out calibration set.

    Args:
        init_temperature: Initial value for T before fitting (default 1.0).
        lr: Gradient-descent learning rate (default 0.01).
        max_iter: Maximum number of gradient-descent steps (default 1000).
        tol: Convergence tolerance on the loss (default 1e-6).

    Attributes:
        temperature_: Fitted temperature parameter (available after
            calling :meth:`fit`).
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive.")
        self._init_temperature = init_temperature
        self._lr = lr
        self._max_iter = max_iter
        self._tol = tol
        self.temperature_: float = init_temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        confidences: list[float] | np.ndarray,
        outcomes: list[int] | np.ndarray,
    ) -> "TemperatureScaling":
        """Fit the temperature on a calibration set.

        Minimises the binary cross-entropy (NLL) between the temperature-scaled
        confidences and the binary outcomes using gradient descent.

        Args:
            confidences: Held-out confidence values in (0, 1).
            outcomes: Binary correctness outcomes (1 = correct, 0 = wrong).

        Returns:
            *self* (for chaining).
        """
        p = np.asarray(confidences, dtype=float)
        o = np.asarray(outcomes, dtype=float)

        if len(p) != len(o) or len(p) == 0:
            raise ValueError("confidences and outcomes must be non-empty and same length.")

        logits = np.array([_logit(float(pi)) for pi in p])
        T = self._init_temperature
        prev_loss = float("inf")

        for _ in range(self._max_iter):
            scaled = logits / T
            p_cal = np.array([_sigmoid(float(s)) for s in scaled])
            p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)

            # NLL loss
            loss = float(-np.mean(o * np.log(p_cal) + (1.0 - o) * np.log(1.0 - p_cal)))

            # Gradient of NLL w.r.t. T
            # d/dT [-y * log(sigma(l/T)) - (1-y) * log(1 - sigma(l/T))]
            # = (sigma(l/T) - y) * (-l / T^2)
            grad = float(np.mean((p_cal - o) * (-logits / (T ** 2))))

            T = T - self._lr * grad
            T = max(T, 1e-4)  # keep T positive

            if abs(prev_loss - loss) < self._tol:
                break
            prev_loss = loss

        self.temperature_ = float(T)
        return self

    def transform(
        self,
        confidences: list[float] | np.ndarray,
    ) -> list[float]:
        """Apply temperature scaling to a list of confidences.

        Args:
            confidences: Raw confidence values in (0, 1).

        Returns:
            List of calibrated confidence values.
        """
        result = []
        for p in confidences:
            logit_p = _logit(float(p))
            result.append(_sigmoid(logit_p / self.temperature_))
        return result

    def fit_transform(
        self,
        confidences: list[float] | np.ndarray,
        outcomes: list[int] | np.ndarray,
    ) -> list[float]:
        """Fit and then transform using the same data.

        Convenience wrapper combining :meth:`fit` and :meth:`transform`.

        Args:
            confidences: Confidence values.
            outcomes: Binary outcomes.

        Returns:
            Calibrated confidence values.
        """
        self.fit(confidences, outcomes)
        return self.transform(confidences)
