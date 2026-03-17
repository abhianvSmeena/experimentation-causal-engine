"""Always-valid inference with the mixture sequential probability ratio test (mSPRT).

Johari, Koomen, Pekelis, Walsh (2017), "Peeking at A/B tests". With a normal mixing
distribution N(theta0, tau2) over the true difference, the likelihood ratio after
observing an estimated difference ``diff`` with variance ``var_diff`` is

    Lambda = sqrt(V / (V + tau2)) * exp(tau2 * (diff - theta0)^2 / (2 V (V + tau2)))

with V = var_diff. The always-valid p-value is the running minimum of 1 / Lambda, so an
experimenter may look at it after every checkpoint and stop whenever p < alpha while the
false-positive rate stays at most alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def msprt_log_lr(
    diff: np.ndarray, var_diff: np.ndarray, tau2: float, theta0: float = 0.0
) -> np.ndarray:
    diff = np.asarray(diff, dtype=float)
    v = np.asarray(var_diff, dtype=float)
    return 0.5 * np.log(v / (v + tau2)) + tau2 * (diff - theta0) ** 2 / (2 * v * (v + tau2))


def msprt_pvalues(
    diffs: np.ndarray, var_diffs: np.ndarray, tau2: float, theta0: float = 0.0
) -> np.ndarray:
    """Always-valid p-values along the last axis (checkpoints in time order)."""
    log_lr = msprt_log_lr(diffs, var_diffs, tau2, theta0)
    p = np.minimum(1.0, np.exp(-log_lr))
    return np.minimum.accumulate(p, axis=-1)


def always_valid_ci(
    diff: np.ndarray, var_diff: np.ndarray, tau2: float, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence sequence: the set of theta0 for which Lambda(theta0) < 1 / alpha.

    The running intersection over checkpoints is also valid; callers who want it can
    apply np.maximum.accumulate / np.minimum.accumulate to the bounds.
    """
    diff = np.asarray(diff, dtype=float)
    v = np.asarray(var_diff, dtype=float)
    half_width = np.sqrt(2 * v * (v + tau2) / tau2 * np.log(np.sqrt((v + tau2) / v) / alpha))
    return diff - half_width, diff + half_width


def peeking_simulation(
    n_experiments: int = 10_000,
    n_per_arm: int = 10_000,
    n_looks: int = 10,
    alpha: float = 0.05,
    tau2_in_sd_units: float = 0.01,
    effect_in_sd_units: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Simulate experiments that are checked ``n_looks`` times at equal intervals.

    Responses are standardized (sd = 1). With ``effect_in_sd_units = 0`` (A/A tests)
    every "significant" result is a false positive. Returns the share of experiments
    declared significant by three stopping rules:

    - fixed: one test at the planned end
    - naive_peeking: stop at the first look where the Welch p-value < alpha
    - msprt: stop at the first look where the always-valid p-value < alpha
    """
    rng = np.random.default_rng(seed)
    batch = n_per_arm // n_looks
    # Each look adds `batch` users per arm. The difference of batch means is normal with
    # variance 2 / batch, so we simulate batch-level differences directly.
    batch_diffs = rng.normal(effect_in_sd_units, np.sqrt(2 / batch), size=(n_experiments, n_looks))
    n = batch * np.arange(1, n_looks + 1)
    cum_diff = np.cumsum(batch_diffs, axis=1) / np.arange(1, n_looks + 1)
    var_diff = 2 / n
    z = cum_diff / np.sqrt(var_diff)
    naive_p = 2 * norm.sf(np.abs(z))
    seq_p = msprt_pvalues(cum_diff, np.broadcast_to(var_diff, cum_diff.shape), tau2_in_sd_units)
    return {
        "fixed": float(np.mean(naive_p[:, -1] < alpha)),
        "naive_peeking": float(np.mean((naive_p < alpha).any(axis=1))),
        "msprt": float(np.mean(seq_p[:, -1] < alpha)),
    }
