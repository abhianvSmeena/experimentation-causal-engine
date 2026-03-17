"""Classical (fixed-horizon) tests that work from summary statistics or raw arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_dist


@dataclass(frozen=True)
class TestResult:
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    rel_lift: float
    rel_ci_low: float
    rel_ci_high: float


def welch_test(
    mean_c: float,
    var_c: float,
    n_c: float,
    mean_t: float,
    var_t: float,
    n_t: float,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test from summary statistics (means, sample variances, counts).

    Also returns the relative lift (mean_t / mean_c - 1) with a delta-method CI,
    which is what product teams usually ask for ("+2.1% conversion").
    """
    se2_c, se2_t = var_c / n_c, var_t / n_t
    se = float(np.sqrt(se2_c + se2_t))
    diff = mean_t - mean_c
    dof = (se2_c + se2_t) ** 2 / (se2_c**2 / (n_c - 1) + se2_t**2 / (n_t - 1))
    t_stat = diff / se
    p_value = float(2 * t_dist.sf(abs(t_stat), dof))
    crit = float(t_dist.ppf(1 - alpha / 2, dof))

    rel = mean_t / mean_c - 1
    rel_se = float(np.sqrt(se2_t / mean_c**2 + mean_t**2 * se2_c / mean_c**4))
    z = norm.ppf(1 - alpha / 2)
    return TestResult(
        diff=float(diff),
        se=se,
        ci_low=float(diff - crit * se),
        ci_high=float(diff + crit * se),
        p_value=p_value,
        rel_lift=float(rel),
        rel_ci_low=float(rel - z * rel_se),
        rel_ci_high=float(rel + z * rel_se),
    )


def welch_test_arrays(y_c: np.ndarray, y_t: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Welch's t-test on raw per-user values."""
    y_c, y_t = np.asarray(y_c, dtype=float), np.asarray(y_t, dtype=float)
    return welch_test(
        y_c.mean(), y_c.var(ddof=1), len(y_c), y_t.mean(), y_t.var(ddof=1), len(y_t), alpha
    )


def proportion_test(
    conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05
) -> TestResult:
    """Two-proportion test (unpooled), e.g. for conversion rates."""
    p_c, p_t = conv_c / n_c, conv_t / n_t
    return welch_test(p_c, p_c * (1 - p_c), n_c, p_t, p_t * (1 - p_t), n_t, alpha)


def delta_method_ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """Mean and variance of a ratio metric such as revenue per session.

    When the experiment randomizes users but the metric is per session, sessions from the
    same user are correlated and the naive variance is too small. The delta method gives
    the correct variance of sum(numerator) / sum(denominator) with users as the unit.
    """
    x = np.asarray(numerator, dtype=float)
    y = np.asarray(denominator, dtype=float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    ratio = mx / my
    var_ratio = (vx / my**2 - 2 * mx * cov / my**3 + mx**2 * vy / my**4) / n
    return float(ratio), float(var_ratio)


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries controlling the false discovery rate at ``q``.

    Needed when many metrics or variants are tested at once: at alpha = 0.05,
    one in twenty null comparisons is "significant" by chance.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresholds = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        rejected[order[: k + 1]] = True
    return rejected
