"""Experiment design: sample size, minimum detectable effect (MDE), and power.

All formulas use the normal approximation, which is standard for online experiments
where each arm has thousands of users or more.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _z_values(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    return z_alpha, norm.ppf(power)


def _se(n_control, n_treatment, sd_control, sd_treatment) -> np.ndarray:
    sd_treatment = sd_control if sd_treatment is None else sd_treatment
    return np.sqrt(
        np.asarray(sd_control) ** 2 / n_control + np.asarray(sd_treatment) ** 2 / n_treatment
    )


def sample_size_means(
    sd: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> tuple[int, int]:
    """Users needed per arm to detect an absolute difference ``mde`` in a mean.

    ``ratio`` is n_treatment / n_control. Returns (n_control, n_treatment).
    """
    if mde <= 0 or sd <= 0:
        raise ValueError("sd and mde must be positive")
    z_alpha, z_beta = _z_values(alpha, power, two_sided)
    n_c = (z_alpha + z_beta) ** 2 * sd**2 * (1 + 1 / ratio) / mde**2
    return math.ceil(n_c), math.ceil(n_c * ratio)


def sample_size_proportions(
    p_baseline: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> tuple[int, int]:
    """Users needed per arm to detect a relative lift in a conversion rate.

    Example: ``p_baseline=0.10, relative_mde=0.05`` means detecting 10% -> 10.5%.
    """
    p1 = p_baseline
    p2 = p_baseline * (1 + relative_mde)
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("baseline and treated rates must be in (0, 1)")
    z_alpha, z_beta = _z_values(alpha, power, two_sided)
    variance = p1 * (1 - p1) + p2 * (1 - p2) / ratio
    n_c = (z_alpha + z_beta) ** 2 * variance / (p2 - p1) ** 2
    return math.ceil(n_c), math.ceil(n_c * ratio)


def mde(
    n_control: float,
    n_treatment: float,
    sd_control: float,
    sd_treatment: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> np.ndarray | float:
    """Smallest absolute difference detectable with the given sample sizes. Vectorized."""
    z_alpha, z_beta = _z_values(alpha, power, two_sided)
    return (z_alpha + z_beta) * _se(n_control, n_treatment, sd_control, sd_treatment)


def power(
    n_control: float,
    n_treatment: float,
    sd_control: float,
    effect: float,
    sd_treatment: float | None = None,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> np.ndarray | float:
    """Probability of detecting a true absolute difference ``effect``. Vectorized."""
    z = np.abs(effect) / _se(n_control, n_treatment, sd_control, sd_treatment)
    if two_sided:
        z_alpha = norm.ppf(1 - alpha / 2)
        return norm.cdf(z - z_alpha) + norm.cdf(-z - z_alpha)
    return norm.cdf(z - norm.ppf(1 - alpha))
