import numpy as np

from experiment_toolkit import sequential


def test_naive_peeking_inflates_false_positives_but_msprt_does_not():
    rates = sequential.peeking_simulation(n_experiments=20_000, n_looks=10, seed=3)
    assert 0.04 <= rates["fixed"] <= 0.06
    assert rates["naive_peeking"] > 0.15
    assert rates["msprt"] <= 0.055


def test_msprt_detects_a_real_effect():
    rates = sequential.peeking_simulation(
        n_experiments=2_000, n_looks=10, effect_in_sd_units=0.06, seed=4
    )
    assert rates["msprt"] > 0.6


def test_always_valid_pvalues_never_increase():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 0.01, 50)
    var = np.linspace(1e-3, 1e-5, 50)
    p = sequential.msprt_pvalues(diffs, var, tau2=1e-4)
    assert np.all(np.diff(p) <= 0)
    assert np.all((p > 0) & (p <= 1))


def test_always_valid_ci_is_wider_than_fixed_ci_and_contains_estimate():
    lo, hi = sequential.always_valid_ci(0.02, 1e-4, tau2=1e-4)
    fixed_half_width = 1.96 * 1e-2
    assert lo < 0.02 < hi
    assert (hi - lo) / 2 > fixed_half_width
