import numpy as np
import pytest

from experiment_toolkit import bayesian


def test_identical_arms_give_coin_flip():
    r = bayesian.beta_binomial_ab(1000, 10_000, 1000, 10_000)
    assert r.prob_treatment_better == pytest.approx(0.5, abs=0.01)
    assert r.expected_loss_ship == pytest.approx(r.expected_loss_keep, rel=0.05)


def test_clear_winner():
    r = bayesian.beta_binomial_ab(1000, 10_000, 1200, 10_000)
    assert r.prob_treatment_better > 0.999
    assert r.expected_loss_ship < 1e-5


def test_normal_decision_closed_form_matches_monte_carlo():
    diff, se = 0.3, 0.2
    r = bayesian.normal_decision(diff, se)
    draws = np.random.default_rng(0).normal(diff, se, 1_000_000)
    assert r.prob_treatment_better == pytest.approx(np.mean(draws > 0), abs=0.002)
    assert r.expected_loss_ship == pytest.approx(np.mean(np.maximum(-draws, 0)), rel=0.02)
    assert r.expected_loss_keep == pytest.approx(np.mean(np.maximum(draws, 0)), rel=0.02)


def test_prior_shrinks_noisy_estimates_toward_zero():
    mean, sd = bayesian.normal_posterior(0.5, 0.5, prior_var=0.25)
    assert 0 < mean < 0.5
    assert sd < 0.5


def test_prior_variance_is_recovered_from_many_experiments():
    rng = np.random.default_rng(5)
    true_effects = rng.normal(0, 0.1, 5000)
    ses = rng.uniform(0.05, 0.15, 5000)
    observed = true_effects + rng.normal(0, ses)
    assert bayesian.estimate_prior_variance(observed, ses) == pytest.approx(0.01, rel=0.1)
