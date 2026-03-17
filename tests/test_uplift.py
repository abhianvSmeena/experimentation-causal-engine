import numpy as np
import pytest

from experiment_toolkit import uplift


@pytest.fixture(scope="module")
def synthetic():
    """Randomized data where only users with x0 > 0 respond to treatment (+10 points)."""
    rng = np.random.default_rng(0)
    n = 40_000
    x = rng.normal(size=(n, 4))
    t = (rng.random(n) < 0.7).astype(int)
    true_tau = np.where(x[:, 0] > 0, 0.10, 0.0)
    base = 0.05 + 0.03 * (x[:, 1] > 0)
    y = (rng.random(n) < base + t * true_tau).astype(int)
    return x, t, y, true_tau


@pytest.mark.parametrize(
    "learner", [uplift.SLearner, uplift.TLearner, uplift.XLearner, uplift.DRLearner]
)
def test_learners_recover_who_responds(synthetic, learner):
    x, t, y, true_tau = synthetic
    model = learner().fit(x[:30_000], t[:30_000], y[:30_000])
    pred = model.predict(x[30_000:])
    responders = true_tau[30_000:] > 0
    assert pred[responders].mean() - pred[~responders].mean() > 0.05


def test_qini_endpoint_equals_ate_and_perfect_model_beats_random(synthetic):
    x, t, y, true_tau = synthetic
    perfect = uplift.SortedData.from_scores(y, t, true_tau)
    random = uplift.SortedData.from_scores(y, t, np.random.default_rng(1).random(len(y)))
    ate = y[t == 1].mean() - y[t == 0].mean()
    _, q = uplift.qini_curve(perfect)
    assert q[-1] == pytest.approx(ate)
    assert uplift.qini_coefficient(perfect) > 0.01
    assert abs(uplift.qini_coefficient(random)) < uplift.qini_coefficient(perfect) / 3


def test_uplift_at_k_for_perfect_ranking(synthetic):
    _, t, y, true_tau = synthetic
    perfect = uplift.SortedData.from_scores(y, t, true_tau)
    assert uplift.uplift_at_k(perfect, 0.3) == pytest.approx(0.10, abs=0.02)
    assert uplift.uplift_at_k(perfect, 1.0) == pytest.approx(
        y[t == 1].mean() - y[t == 0].mean(), abs=1e-9
    )


def test_learners_with_estimated_propensity(synthetic):
    x, t, y, true_tau = synthetic
    for learner in (uplift.XLearner, uplift.DRLearner):
        model = learner(propensity_model=uplift.default_classifier()).fit(x, t, y)
        pred = model.predict(x)
        assert pred[true_tau > 0].mean() - pred[true_tau == 0].mean() > 0.05


def test_aipw_ate_is_unbiased_and_tighter_than_difference_in_means(synthetic):
    x, t, y, _ = synthetic
    true_ate = 0.05  # half the users get +0.10
    mu0 = uplift.default_classifier().fit(x[t == 0], y[t == 0]).predict_proba(x)[:, 1]
    mu1 = uplift.default_classifier().fit(x[t == 1], y[t == 1]).predict_proba(x)[:, 1]
    ate, se = uplift.aipw_ate(y, t, mu0, mu1, t.mean())
    naive_se = np.sqrt(y[t == 1].var() / (t == 1).sum() + y[t == 0].var() / (t == 0).sum())
    assert abs(ate - true_ate) < 3 * se
    assert se < naive_se


def test_paired_bootstrap_is_paired(synthetic):
    _, t, y, true_tau = synthetic
    data = uplift.SortedData.from_scores(y, t, true_tau)
    samples = uplift.bootstrap_samples({"a": data, "b": data}, uplift.qini_coefficient, n_boot=20)
    assert np.allclose(samples["a"], samples["b"])
    assert samples["a"].std() > 0


def test_bootstrap_ci_contains_point_estimate(synthetic):
    _, t, y, true_tau = synthetic
    data = uplift.SortedData.from_scores(y, t, true_tau)
    point, lo, hi = uplift.bootstrap(data, uplift.qini_coefficient, n_boot=50)
    assert lo < point < hi
