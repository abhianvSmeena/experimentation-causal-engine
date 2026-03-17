import numpy as np

from experiment_toolkit import checks


def test_srm_passes_for_balanced_split():
    assert not checks.srm_test([500_120, 499_880]).mismatch


def test_srm_flags_48_52_split_on_a_million_users():
    result = checks.srm_test([480_000, 520_000])
    assert result.mismatch
    assert result.p_value < 1e-10


def test_srm_respects_planned_unequal_split():
    assert not checks.srm_test([200_100, 799_900], expected_shares=[0.2, 0.8]).mismatch


def test_covariate_balance_is_near_zero_under_randomization():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50_000, 3))
    t = rng.random(50_000) < 0.5
    assert np.all(np.abs(checks.covariate_balance(x, t)) < 0.05)
