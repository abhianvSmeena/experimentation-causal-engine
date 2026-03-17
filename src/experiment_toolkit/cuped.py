"""CUPED variance reduction (Deng, Xu, Kohavi, Walker 2013).

Uses a metric measured before the experiment started (e.g. last month's spend) to remove
predictable noise from the experiment metric. Because the pre-period covariate cannot be
affected by the treatment, the effect estimate stays unbiased while its variance drops by
roughly corr(y, x)^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_toolkit.frequentist import TestResult, welch_test_arrays


@dataclass(frozen=True)
class CupedResult:
    raw: TestResult
    adjusted: TestResult
    theta: float
    variance_reduction: float


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return float(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1))


def cuped_adjust(
    y: np.ndarray, x: np.ndarray, theta: float | None = None, x_mean: float | None = None
) -> np.ndarray:
    """Return y - theta * (x - mean(x)). The mean of y is unchanged."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = cuped_theta(y, x) if theta is None else theta
    x_mean = x.mean() if x_mean is None else x_mean
    return y - theta * (x - x_mean)


def cuped_test(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, alpha: float = 0.05
) -> CupedResult:
    """Compare a raw Welch test with a CUPED-adjusted one.

    theta and the covariate mean are estimated on both arms pooled, so the same
    adjustment is applied to every user regardless of their group.
    """
    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    theta = cuped_theta(y_all, x_all)
    x_mean = float(x_all.mean())
    adj_c = cuped_adjust(y_c, x_c, theta, x_mean)
    adj_t = cuped_adjust(y_t, x_t, theta, x_mean)
    raw = welch_test_arrays(y_c, y_t, alpha)
    adjusted = welch_test_arrays(adj_c, adj_t, alpha)
    return CupedResult(
        raw=raw,
        adjusted=adjusted,
        theta=theta,
        variance_reduction=float(1 - adjusted.se**2 / raw.se**2),
    )
