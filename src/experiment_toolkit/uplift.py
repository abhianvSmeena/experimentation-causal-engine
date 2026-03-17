"""Uplift modeling: estimate who responds *because of* a treatment, and evaluate targeting.

Meta-learners estimate the conditional average treatment effect (CATE)
tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0] using any scikit-learn model as a base learner.
The default base learner is scikit-learn's histogram gradient boosting (LightGBM-style).

All learners assume a randomized experiment or a known/constant propensity score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

GBM_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=500)


def default_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed)


def default_regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**GBM_PARAMS, random_state=seed)


def _proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def aipw_pseudo_outcome(
    y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray | float
) -> np.ndarray:
    """Doubly robust (AIPW) score per row. Its mean is the ATE; it is the DR-learner target."""
    t = np.asarray(t).astype(bool)
    return mu1 - mu0 + t * (y - mu1) / e - (~t) * (y - mu0) / (1 - e)


def aipw_ate(
    y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, e: np.ndarray | float
) -> tuple[float, float]:
    """Average treatment effect and its standard error from AIPW scores.

    In a randomized experiment this is unbiased like the difference in means, but the
    outcome models absorb predictable noise, so the standard error is usually smaller
    (the same idea as CUPED, with many covariates and a flexible model).
    """
    scores = aipw_pseudo_outcome(y, t, mu0, mu1, e)
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(len(scores)))


class SLearner:
    """One model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0)."""

    def __init__(self, model=None):
        self.model = model if model is not None else default_classifier()

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> SLearner:
        self.model_ = clone(self.model).fit(np.column_stack([x, t]), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        ones, zeros = np.ones(len(x)), np.zeros(len(x))
        return _proba(self.model_, np.column_stack([x, ones])) - _proba(
            self.model_, np.column_stack([x, zeros])
        )


class TLearner:
    """Separate outcome models for treated and control; uplift = mu1(x) - mu0(x)."""

    def __init__(self, model=None):
        self.model = model if model is not None else default_classifier()

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> TLearner:
        t = t.astype(bool)
        self.mu0_ = clone(self.model).fit(x[~t], y[~t])
        self.mu1_ = clone(self.model).fit(x[t], y[t])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _proba(self.mu1_, x) - _proba(self.mu0_, x)


class XLearner:
    """Künzel et al. (2019). Works well when one arm is much larger than the other.

    Stage 1: T-learner outcome models. Stage 2: impute individual effects for each arm
    using the other arm's model, regress them on x, and blend the two effect models
    weighted by the propensity score.
    """

    def __init__(self, outcome_model=None, effect_model=None, propensity_model=None):
        self.outcome_model = outcome_model if outcome_model is not None else default_classifier()
        self.effect_model = effect_model if effect_model is not None else default_regressor()
        self.propensity_model = propensity_model

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> XLearner:
        t = t.astype(bool)
        self.propensity_ = float(t.mean())
        if self.propensity_model is not None:
            self.propensity_model_ = clone(self.propensity_model).fit(x, t)
        mu0 = clone(self.outcome_model).fit(x[~t], y[~t])
        mu1 = clone(self.outcome_model).fit(x[t], y[t])
        d1 = y[t] - _proba(mu0, x[t])
        d0 = _proba(mu1, x[~t]) - y[~t]
        self.tau1_ = clone(self.effect_model).fit(x[t], d1)
        self.tau0_ = clone(self.effect_model).fit(x[~t], d0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.propensity_model is not None:
            e = np.clip(_proba(self.propensity_model_, x), 0.01, 0.99)
        else:
            e = self.propensity_
        return e * self.tau0_.predict(x) + (1 - e) * self.tau1_.predict(x)


class DRLearner:
    """Doubly robust learner (Kennedy 2020) with cross-fitting.

    Builds a pseudo-outcome whose expectation equals the true individual effect if either
    the outcome models or the propensity score are correct, then regresses it on x.
    Without a ``propensity_model`` the propensity is the treated share, which is exact in a
    perfectly randomized experiment; with one, it is estimated out-of-fold and clipped.
    """

    def __init__(
        self,
        outcome_model=None,
        effect_model=None,
        propensity_model=None,
        n_folds: int = 2,
        seed: int = 0,
    ):
        self.outcome_model = outcome_model if outcome_model is not None else default_classifier()
        self.effect_model = effect_model if effect_model is not None else default_regressor()
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> DRLearner:
        t = t.astype(bool)
        mu0_hat = np.empty(len(y))
        mu1_hat = np.empty(len(y))
        e = np.full(len(y), t.mean())
        folds = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        for train_idx, pred_idx in folds.split(x):
            tt = t[train_idx]
            mu0 = clone(self.outcome_model).fit(x[train_idx][~tt], y[train_idx][~tt])
            mu1 = clone(self.outcome_model).fit(x[train_idx][tt], y[train_idx][tt])
            mu0_hat[pred_idx] = _proba(mu0, x[pred_idx])
            mu1_hat[pred_idx] = _proba(mu1, x[pred_idx])
            if self.propensity_model is not None:
                prop = clone(self.propensity_model).fit(x[train_idx], tt)
                e[pred_idx] = np.clip(_proba(prop, x[pred_idx]), 0.01, 0.99)
        pseudo = aipw_pseudo_outcome(y, t, mu0_hat, mu1_hat, e)
        self.tau_ = clone(self.effect_model).fit(x, pseudo)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.tau_.predict(x)


# ---------------------------------------------------------------------------
# Evaluation. We never see both outcomes for one person, so uplift models are judged on
# groups: rank people by predicted uplift, then compare treated vs control outcomes among
# the top k%. Weights make bootstrap resampling cheap (Poisson bootstrap on a fixed order).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortedData:
    """Outcomes and treatment flags sorted by descending predicted uplift."""

    y: np.ndarray
    t: np.ndarray
    order: np.ndarray

    @classmethod
    def from_scores(
        cls, y: np.ndarray, t: np.ndarray, score: np.ndarray, seed: int = 0
    ) -> SortedData:
        """Ties in the score (common with tree models) are broken at random, not by row order."""
        score = np.asarray(score)
        perm = np.random.default_rng(seed).permutation(len(score))
        order = perm[np.argsort(-score[perm], kind="stable")]
        return cls(np.asarray(y, dtype=float)[order], np.asarray(t, dtype=float)[order], order)


def _cumulative(data: SortedData, weights: np.ndarray | None):
    w = np.ones(len(data.y)) if weights is None else weights
    n_t = np.cumsum(w * data.t)
    n_c = np.cumsum(w * (1 - data.t))
    y_t = np.cumsum(w * data.t * data.y)
    y_c = np.cumsum(w * (1 - data.t) * data.y)
    return n_t, n_c, y_t, y_c


def qini_curve(
    data: SortedData, weights: np.ndarray | None = None, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcomes from treating only the top fraction of users.

    Q(k) = [Y_t(k) - Y_c(k) * N_t(k) / N_c(k)] / N_t(total). The normalisation makes
    Q(1) equal to the average treatment effect. Returns (fraction targeted, Q).
    """
    n_t, n_c, y_t, y_c = _cumulative(data, weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0) / n_t[-1]
    n = len(q)
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    frac = (idx + 1) / n
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], q[idx]])


def qini_coefficient(data: SortedData, weights: np.ndarray | None = None) -> float:
    """Area between the Qini curve and the random-targeting line (higher is better)."""
    frac, q = qini_curve(data, weights, n_points=1001)
    random_line = frac * q[-1]
    return float(np.trapezoid(q - random_line, frac))


def uplift_at_k(data: SortedData, k: float, weights: np.ndarray | None = None) -> float:
    """Treated minus control outcome rate among the top k fraction ranked by the model."""
    n_t, n_c, y_t, y_c = _cumulative(data, weights)
    total = n_t[-1] + n_c[-1]
    i = int(np.searchsorted(n_t + n_c, k * total))
    i = min(i, len(n_t) - 1)
    return float(y_t[i] / n_t[i] - y_c[i] / n_c[i])


def bootstrap(
    data: SortedData,
    metric: Callable[[SortedData, np.ndarray | None], float],
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI using Poisson(1) weights on the evaluation rows."""
    rng = np.random.default_rng(seed)
    point = metric(data, None)
    stats = [metric(data, rng.poisson(1.0, len(data.y)).astype(float)) for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bootstrap_samples(
    data_by_model: dict[str, SortedData],
    metric: Callable[[SortedData, np.ndarray | None], float],
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Paired bootstrap: every model is scored on the same resampled users in each draw.

    All SortedData must come from the same evaluation rows. Because the weights are drawn
    per user (in original row order) and then permuted into each model's ranking,
    differences between models get valid CIs.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(data_by_model.values())).y)
    out: dict[str, list[float]] = {name: [] for name in data_by_model}
    for _ in range(n_boot):
        w = rng.poisson(1.0, n).astype(float)
        for name, data in data_by_model.items():
            out[name].append(metric(data, w[data.order]))
    return {name: np.asarray(v) for name, v in out.items()}
