"""Validity checks that must pass before an experiment result can be trusted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chisquare


@dataclass(frozen=True)
class SRMResult:
    chi2: float
    p_value: float
    observed_shares: tuple[float, ...]
    expected_shares: tuple[float, ...]
    mismatch: bool


def srm_test(
    observed_counts: list[float] | np.ndarray,
    expected_shares: list[float] | np.ndarray | None = None,
    threshold: float = 0.001,
) -> SRMResult:
    """Sample ratio mismatch test (chi-square goodness of fit).

    If users were meant to be split 50/50 but the observed split is off by more than
    chance allows, randomization or logging is broken and the result should not be used.
    The conventional threshold is p < 0.001 because SRM checks run on every experiment.
    """
    observed = np.asarray(observed_counts, dtype=float)
    if expected_shares is None:
        expected_shares = np.full(len(observed), 1 / len(observed))
    shares = np.asarray(expected_shares, dtype=float)
    shares = shares / shares.sum()
    chi2, p_value = chisquare(observed, f_exp=shares * observed.sum())
    return SRMResult(
        chi2=float(chi2),
        p_value=float(p_value),
        observed_shares=tuple(observed / observed.sum()),
        expected_shares=tuple(shares),
        mismatch=bool(p_value < threshold),
    )


def covariate_balance(x: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Standardized mean difference per column between treated and control rows.

    In a properly randomized experiment every value should be close to 0
    (|SMD| < 0.1 is the usual rule of thumb).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(treatment).astype(bool)
    m1, m0 = x[t].mean(axis=0), x[~t].mean(axis=0)
    pooled_sd = np.sqrt((x[t].var(axis=0, ddof=1) + x[~t].var(axis=0, ddof=1)) / 2)
    return (m1 - m0) / pooled_sd
