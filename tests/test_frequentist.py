import numpy as np
import pytest
from scipy import stats

from experiment_toolkit import frequentist


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    y_c = rng.normal(10, 3, 800)
    y_t = rng.normal(10.4, 4, 1200)
    ours = frequentist.welch_test_arrays(y_c, y_t)
    ref = stats.ttest_ind(y_t, y_c, equal_var=False)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert ours.ci_low < ours.diff < ours.ci_high


def test_relative_lift_ci_covers_truth_about_95_percent_of_the_time():
    rng = np.random.default_rng(1)
    true_rel = 0.05
    covered = 0
    for _ in range(400):
        y_c = rng.normal(100, 30, 2000)
        y_t = rng.normal(100 * (1 + true_rel), 30, 2000)
        r = frequentist.welch_test_arrays(y_c, y_t)
        covered += r.rel_ci_low <= true_rel <= r.rel_ci_high
    assert 0.92 <= covered / 400 <= 0.98


def test_delta_method_variance_matches_bootstrap():
    rng = np.random.default_rng(2)
    sessions = rng.poisson(3, 5000) + 1
    revenue = sessions * rng.gamma(2, 5, 5000)
    ratio, var = frequentist.delta_method_ratio(revenue, sessions)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, 5000, 5000)
        boots.append(revenue[idx].sum() / sessions[idx].sum())
    assert ratio == pytest.approx(revenue.sum() / sessions.sum())
    assert var == pytest.approx(np.var(boots), rel=0.2)


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rejected = frequentist.benjamini_hochberg(p, q=0.05)
    assert rejected.tolist() == [True, True] + [False] * 8
