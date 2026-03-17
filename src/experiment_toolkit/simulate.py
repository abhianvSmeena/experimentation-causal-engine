"""Simulated experiments with a known true effect, used by the CLI and the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_experiment(
    n_per_arm: int = 20_000,
    baseline_rate: float = 0.10,
    relative_lift: float = 0.05,
    spend_mean: float = 50.0,
    pre_post_corr: float = 0.7,
    treated_share: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per user with a pre-period spend, an in-experiment spend, and a conversion flag.

    The treatment multiplies both the conversion rate and the expected spend by
    (1 + relative_lift). ``pre_post_corr`` controls how useful CUPED will be.
    """
    rng = np.random.default_rng(seed)
    n = int(2 * n_per_arm)
    treated = rng.random(n) < treated_share
    # Correlated standard normals for pre-period and in-experiment spend.
    z_pre = rng.standard_normal(n)
    z_post = pre_post_corr * z_pre + np.sqrt(1 - pre_post_corr**2) * rng.standard_normal(n)
    sd = spend_mean * 0.5
    lift = np.where(treated, 1 + relative_lift, 1.0)
    pre_spend = spend_mean + sd * z_pre
    spend = (spend_mean + sd * z_post) * lift
    converted = rng.random(n) < baseline_rate * lift
    return pd.DataFrame(
        {"treated": treated, "pre_spend": pre_spend, "spend": spend, "converted": converted}
    )
