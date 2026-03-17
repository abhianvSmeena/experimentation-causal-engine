import numpy as np
import pytest

from experiment_toolkit import cuped
from experiment_toolkit.simulate import simulate_experiment


def _split(df):
    c, t = df[~df.treated], df[df.treated]
    return (c.spend.to_numpy(), c.pre_spend.to_numpy(), t.spend.to_numpy(), t.pre_spend.to_numpy())


def test_cuped_variance_reduction_is_about_rho_squared():
    result = cuped.cuped_test(*_split(simulate_experiment(n_per_arm=50_000, pre_post_corr=0.7)))
    assert result.variance_reduction == pytest.approx(0.49, abs=0.03)
    assert result.adjusted.se < result.raw.se


def test_cuped_estimate_is_unbiased():
    true_diff = 50.0 * 0.05
    estimates = [
        cuped.cuped_test(*_split(simulate_experiment(n_per_arm=5_000, seed=s))).adjusted.diff
        for s in range(200)
    ]
    assert np.mean(estimates) == pytest.approx(true_diff, abs=0.1)


def test_cuped_does_nothing_with_useless_covariate():
    result = cuped.cuped_test(*_split(simulate_experiment(n_per_arm=50_000, pre_post_corr=0.0)))
    assert abs(result.variance_reduction) < 0.01
