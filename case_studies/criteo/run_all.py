"""Case study B: uplift modeling and targeting policy on the Criteo uplift dataset (14M users).

Question: if we can only show the ad to a fraction of users, whom should we pick?

Steps (per outcome: visit, conversion)
1. Train S/T/X/DR meta-learners and a causal forest on the training split.
2. Baselines: random targeting, and a "response model" that ranks by P(outcome | treated),
   which is what teams often do by default.
3. Evaluate on the untouched 30% test split (~4.2M users): Qini coefficient with paired
   bootstrap CIs, uplift at 10/20/30/50% budgets, share of all incremental outcomes captured.
4. Average treatment effect: difference in means vs doubly robust (AIPW).
5. DoWhy refutation tests on a 1M-row sample.

Run:  python case_studies/criteo/prepare.py   (once)
      python case_studies/criteo/run_all.py [--train-rows 1000000] [--outcomes visit]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from experiment_toolkit import uplift
from experiment_toolkit.causal import estimate_and_refute

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "criteo.parquet"
OUT = ROOT / "reports" / "criteo"
FIG = ROOT / "reports" / "figures"
FEATURES = [f"f{i}" for i in range(12)]
BUDGETS = (0.1, 0.2, 0.3, 0.5)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def load(con: duckdb.DuckDBPyConnection, split: str, rows: int | None, seed: int) -> pd.DataFrame:
    cols = ", ".join(FEATURES + ["treatment", "visit", "conversion"])
    query = f"SELECT {cols} FROM '{PARQUET}' WHERE split = '{split}'"
    if rows:
        # DuckDB samples before WHERE, so filter in a subquery first.
        query = f"SELECT * FROM ({query}) USING SAMPLE {rows} ROWS (reservoir, {seed})"
    return con.execute(query).df()


def causal_forest(x: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int):
    from econml.dml import CausalForestDML

    model = CausalForestDML(
        model_y=HistGradientBoostingRegressor(max_iter=100, random_state=seed),
        model_t=HistGradientBoostingClassifier(max_iter=100, random_state=seed),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=200,
        cv=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(y, t, X=x)
    return model


def fit_and_score(
    train: pd.DataFrame, test: pd.DataFrame, outcome: str, cf_rows: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    x_tr, t_tr, y_tr = (
        train[FEATURES].to_numpy(),
        train.treatment.to_numpy(),
        train[outcome].to_numpy(),
    )
    x_te = test[FEATURES].to_numpy()
    prop = uplift.default_classifier(seed)
    learners = {
        "S-learner": uplift.SLearner(uplift.default_classifier(seed)),
        "T-learner": uplift.TLearner(uplift.default_classifier(seed)),
        "X-learner": uplift.XLearner(propensity_model=prop),
        "DR-learner": uplift.DRLearner(propensity_model=prop, seed=seed),
    }
    scores: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    scores["Random"] = np.random.default_rng(seed).random(len(test))
    start = time.perf_counter()
    response = uplift.default_classifier(seed).fit(x_tr[t_tr == 1], y_tr[t_tr == 1])
    scores["Response model"] = response.predict_proba(x_te)[:, 1]
    timings["Response model"] = time.perf_counter() - start

    for name, learner in learners.items():
        start = time.perf_counter()
        learner.fit(x_tr, t_tr, y_tr)
        scores[name] = learner.predict(x_te)
        timings[name] = time.perf_counter() - start
        print(f"  {name:<15} fitted + scored in {timings[name]:6.1f}s")

    start = time.perf_counter()
    idx = np.random.default_rng(seed).choice(
        len(train), size=min(cf_rows, len(train)), replace=False
    )
    cf = causal_forest(x_tr[idx], t_tr[idx], y_tr[idx], seed)
    scores["Causal forest"] = cf.effect(x_te)
    timings["Causal forest"] = time.perf_counter() - start
    print(
        f"  {'Causal forest':<15} fitted + scored in {timings['Causal forest']:6.1f}s "
        f"({len(idx):,} training rows)"
    )

    # ATE on the test split: difference in means vs AIPW with models fitted on train.
    t_te, y_te = test.treatment.to_numpy(), test[outcome].to_numpy()
    tl = learners["T-learner"]
    e = np.clip(prop.fit(x_tr, t_tr).predict_proba(x_te)[:, 1], 0.01, 0.99)
    mu0, mu1 = tl.mu0_.predict_proba(x_te)[:, 1], tl.mu1_.predict_proba(x_te)[:, 1]
    aipw, aipw_se = uplift.aipw_ate(y_te, t_te, mu0, mu1, e)
    y1, y0 = y_te[t_te == 1], y_te[t_te == 0]
    naive = y1.mean() - y0.mean()
    naive_se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    ate = dict(naive=naive, naive_se=naive_se, aipw=aipw, aipw_se=aipw_se)
    pseudo = uplift.aipw_pseudo_outcome(y_te, t_te, mu0, mu1, e)
    return scores, timings, ate, pseudo


def evaluate(
    test: pd.DataFrame,
    outcome: str,
    scores: dict[str, np.ndarray],
    pseudo: np.ndarray,
    n_boot: int,
    seed: int,
):
    y, t = test[outcome].to_numpy(), test.treatment.to_numpy()
    datas = {name: uplift.SortedData.from_scores(y, t, s) for name, s in scores.items()}
    boot = uplift.bootstrap_samples(datas, uplift.qini_coefficient, n_boot=n_boot, seed=seed)
    ate = uplift.uplift_at_k(datas["Random"], 1.0)

    rows = []
    for name, data in datas.items():
        q = uplift.qini_coefficient(data)
        diff = boot[name] - boot["Response model"]
        frac, curve = uplift.qini_curve(data, n_points=101)
        row = dict(
            model=name,
            qini=q,
            qini_ci_low=float(np.quantile(boot[name], 0.025)),
            qini_ci_high=float(np.quantile(boot[name], 0.975)),
            vs_response_ci_low=float(np.quantile(diff, 0.025)),
            vs_response_ci_high=float(np.quantile(diff, 0.975)),
        )
        for b in BUDGETS:
            pct = int(b * 100)
            row[f"uplift_per_1000_top{pct}"] = uplift.uplift_at_k(data, b) * 1000
            row[f"share_captured_top{pct}"] = float(np.interp(b, frac, curve) / curve[-1])
            # Robustness: covariate-adjusted (AIPW) uplift in the same top-k group.
            top = data.order[: int(b * len(pseudo))]
            row[f"dr_uplift_per_1000_top{pct}"] = float(pseudo[top].mean() * 1000)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["lift_vs_random_top20"] = table["uplift_per_1000_top20"] / (ate * 1000)
    return table, datas, ate


def plot_qini(datas_by_outcome: dict[str, dict[str, uplift.SortedData]]) -> None:
    fig, axes = plt.subplots(1, len(datas_by_outcome), figsize=(6 * len(datas_by_outcome), 4.4))
    axes = np.atleast_1d(axes)
    styles = {"Random": dict(color="grey", ls="--"), "Response model": dict(color="black", ls=":")}
    for ax, (outcome, datas) in zip(axes, datas_by_outcome.items(), strict=True):
        for name, data in datas.items():
            frac, q = uplift.qini_curve(data, n_points=201)
            ax.plot(frac * 100, q * 1000, label=name, lw=1.6, **styles.get(name, {}))
        ax.set_xlabel("Users targeted, ranked by model score (%)")
        ax.set_ylabel(f"Incremental {outcome}s per 1,000 users")
        ax.set_title(f"Qini curves on 4.2M held-out users: {outcome}", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "criteo_qini_curves.png")
    plt.close(fig)


def plot_policy(tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4))
    axes = np.atleast_1d(axes)
    for ax, (outcome, table) in zip(axes, tables.items(), strict=True):
        table = table.set_index("model")
        cols = [f"uplift_per_1000_top{int(b * 100)}" for b in BUDGETS]
        table[cols].T.set_axis([f"Top {int(b * 100)}%" for b in BUDGETS]).plot.bar(ax=ax, rot=0)
        ax.set_ylabel(f"Incremental {outcome}s per 1,000 treated")
        ax.set_title(f"Who to treat under a budget: {outcome}", fontsize=10)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "criteo_targeting_policy.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=0, help="0 = full train split")
    parser.add_argument("--cf-rows", type=int, default=300_000, help="causal forest training rows")
    parser.add_argument("--dowhy-rows", type=int, default=1_000_000)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--outcomes", nargs="+", default=["visit", "conversion"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    train = load(con, "train", args.train_rows or None, args.seed)
    test = load(con, "test", None, args.seed)
    print(f"train rows: {len(train):,} | test rows: {len(test):,}")

    results: dict[str, object] = dict(train_rows=len(train), test_rows=len(test), outcomes={})
    tables, datas_by_outcome = {}, {}
    for outcome in args.outcomes:
        print(f"\n=== outcome: {outcome} ===")
        scores, timings, ate, pseudo = fit_and_score(train, test, outcome, args.cf_rows, args.seed)
        table, datas, _ = evaluate(test, outcome, scores, pseudo, args.n_boot, args.seed)
        table["fit_seconds"] = table.model.map(timings)
        table.to_csv(OUT / f"uplift_{outcome}.csv", index=False)
        print(table.round(5).to_string(index=False))
        print(
            f"ATE naive {ate['naive']:.5f} (se {ate['naive_se']:.5f}) | "
            f"AIPW {ate['aipw']:.5f} (se {ate['aipw_se']:.5f})"
        )
        tables[outcome], datas_by_outcome[outcome] = table, datas
        results["outcomes"][outcome] = dict(ate=ate, table=table.to_dict(orient="records"))

    plot_qini(datas_by_outcome)
    plot_policy(tables)

    print("\n=== DoWhy refutations ===")
    sample = train.sample(n=min(args.dowhy_rows, len(train)), random_state=args.seed)
    refutations = {}
    for outcome in args.outcomes:
        refutations[outcome] = estimate_and_refute(
            sample[FEATURES + ["treatment", outcome]].astype(float),
            treatment="treatment",
            outcome=outcome,
            common_causes=FEATURES,
            num_simulations=10,
            seed=args.seed,
        )
        print(outcome, json.dumps(refutations[outcome], indent=2))
    results["refutations"] = refutations
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
