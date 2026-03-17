"""Bayesian A/B decisions: probability to beat control and expected loss.

Two models are provided:

- Beta-Binomial for raw conversion counts.
- Normal-Normal for any metric, given an estimated difference and its standard error.
  The prior variance can be learned from many past experiments (empirical Bayes), which
  shrinks noisy early results toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BayesResult:
    prob_treatment_better: float
    expected_loss_ship: float
    expected_loss_keep: float
    posterior_mean: float
    ci_low: float
    ci_high: float


def beta_binomial_ab(
    conv_c: int,
    n_c: int,
    conv_t: int,
    n_t: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    n_samples: int = 200_000,
    seed: int = 0,
) -> BayesResult:
    """Posterior of (rate_t - rate_c) with independent Beta priors, by Monte Carlo.

    expected_loss_ship: average conversion rate lost if we ship and treatment is worse.
    expected_loss_keep: average conversion rate forgone if we keep control and treatment is better.
    """
    rng = np.random.default_rng(seed)
    rate_c = rng.beta(prior_a + conv_c, prior_b + n_c - conv_c, n_samples)
    rate_t = rng.beta(prior_a + conv_t, prior_b + n_t - conv_t, n_samples)
    delta = rate_t - rate_c
    return BayesResult(
        prob_treatment_better=float(np.mean(delta > 0)),
        expected_loss_ship=float(np.mean(np.maximum(-delta, 0))),
        expected_loss_keep=float(np.mean(np.maximum(delta, 0))),
        posterior_mean=float(delta.mean()),
        ci_low=float(np.quantile(delta, 0.025)),
        ci_high=float(np.quantile(delta, 0.975)),
    )


def normal_posterior(
    diff: np.ndarray, se: np.ndarray, prior_var: float = np.inf, prior_mean: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and sd of the true difference. ``prior_var=inf`` means a flat prior."""
    diff, se = np.asarray(diff, dtype=float), np.asarray(se, dtype=float)
    if np.isinf(prior_var):
        return diff, se
    precision = 1 / se**2 + 1 / prior_var
    mean = (diff / se**2 + prior_mean / prior_var) / precision
    return mean, np.sqrt(1 / precision)


def normal_decision(
    diff: float, se: float, prior_var: float = np.inf, prior_mean: float = 0.0
) -> BayesResult:
    """Decision quantities for a normal posterior, all in closed form."""
    m, s = normal_posterior(diff, se, prior_var, prior_mean)
    m, s = float(m), float(s)
    loss_ship = s * norm.pdf(m / s) - m * norm.cdf(-m / s)
    return BayesResult(
        prob_treatment_better=float(norm.cdf(m / s)),
        expected_loss_ship=float(loss_ship),
        expected_loss_keep=float(loss_ship + m),
        posterior_mean=m,
        ci_low=m - 1.959964 * s,
        ci_high=m + 1.959964 * s,
    )


def estimate_prior_variance(effects: np.ndarray, ses: np.ndarray) -> float:
    """Method-of-moments estimate of the spread of true effects across experiments.

    Observed effects vary for two reasons: true effects differ, and each estimate has noise.
    Var(observed) = Var(true) + mean(se^2), so Var(true) = Var(observed) - mean(se^2).
    """
    effects, ses = np.asarray(effects, dtype=float), np.asarray(ses, dtype=float)
    return float(max(effects.var(ddof=1) - np.mean(ses**2), 1e-12))
