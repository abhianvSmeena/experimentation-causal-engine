"""Case study A: re-analysis of 78 real A/B experiments from ASOS.

Steps
1. Data validity: sample ratio mismatch (SRM) and split-ratio drift over time.
2. Fixed-horizon Welch tests at the end of each experiment, with false discovery control.
3. Sensitivity: the smallest relative change each experiment could reliably detect.
4. Replay every experiment checkpoint by checkpoint under three stopping rules:
   fixed horizon, naive peeking, and mSPRT (always-valid).
5. Empirical-Bayes decisions: probability of improvement and expected loss.

Run:  python case_studies/asos/run_all.py
Outputs go to reports/asos/ (tables, results.json) and reports/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

from experiment_toolkit import bayesian, checks, design, frequentist, sequential

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "asos_digital_experiments_dataset.parquet"
OUT = ROOT / "reports" / "asos"
FIG = ROOT / "reports" / "figures"
KEYS = ["experiment_id", "variant_id", "metric_id"]
ALPHA = 0.05

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    n_raw = len(df)
    df = df.dropna(subset=["variance_c", "variance_t"])
    df = df[(df.count_c > 1) & (df.count_t > 1) & (df.variance_c > 0) & (df.variance_t > 0)]
    df = df.sort_values(KEYS + ["time_since_start"]).reset_index(drop=True)
    print(f"Loaded {n_raw:,} rows, kept {len(df):,} with valid variances")

    se2 = df.variance_c / df.count_c + df.variance_t / df.count_t
    df["diff"] = df.mean_t - df.mean_c
    df["se"] = np.sqrt(se2)
    df["p_naive"] = 2 * norm.sf(np.abs(df["diff"] / df["se"]))
    # Standardized effect (Cohen's d style) so experiments on different scales are comparable.
    pooled_sd = np.sqrt((df.variance_c + df.variance_t) / 2)
    df["d"] = df["diff"] / pooled_sd
    df["d_var"] = se2 / pooled_sd**2
    df["rel"] = df.mean_t / df.mean_c - 1
    df["rel_se"] = np.sqrt(
        df.variance_t / df.count_t / df.mean_c**2
        + df.mean_t**2 * df.variance_c / df.count_c / df.mean_c**4
    )
    df["frac_time"] = df.time_since_start / df.groupby(KEYS).time_since_start.transform("max")
    df["checkpoint"] = df.groupby(KEYS).cumcount()
    return df


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(KEYS).tail(1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Validity
# ---------------------------------------------------------------------------


def srm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment-variant (counts are identical across metrics)."""
    m1 = df[df.metric_id == df.metric_id.min()]
    rows = []
    for (exp, var), g in m1.groupby(["experiment_id", "variant_id"]):
        last = g.iloc[-1]
        share_t = last.count_t / (last.count_c + last.count_t)
        planned_50_50 = 0.45 <= share_t <= 0.55
        srm_p = checks.srm_test([last.count_c, last.count_t]).p_value if planned_50_50 else np.nan
        # Drift: new users per interval should split with a constant probability.
        inc = np.diff(np.column_stack([g.count_c, g.count_t]), axis=0, prepend=0)
        inc = inc[(inc > 0).all(axis=1)]
        drift_p = chi2_contingency(inc.T)[1] if len(inc) >= 2 else np.nan
        rows.append(
            dict(
                experiment_id=exp,
                variant_id=var,
                users=int(last.count_c + last.count_t),
                treatment_share=share_t,
                assumed_design="50/50" if planned_50_50 else "unequal (plan unknown)",
                srm_p=srm_p,
                srm_flag=bool(planned_50_50 and srm_p < 0.001),
                drift_p=drift_p,
                drift_flag=bool(drift_p < 0.001),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2-3. Final-checkpoint analysis and sensitivity
# ---------------------------------------------------------------------------


def final_analysis(final: pd.DataFrame) -> pd.DataFrame:
    tests = [
        frequentist.welch_test(r.mean_c, r.variance_c, r.count_c, r.mean_t, r.variance_t, r.count_t)
        for r in final.itertuples()
    ]
    out = final[KEYS + ["count_c", "count_t", "mean_c", "time_since_start"]].copy()
    out["rel_lift"] = [t.rel_lift for t in tests]
    out["rel_ci_low"] = [t.rel_ci_low for t in tests]
    out["rel_ci_high"] = [t.rel_ci_high for t in tests]
    out["p_value"] = [t.p_value for t in tests]
    out["significant"] = out.p_value < ALPHA
    out["bh_discovery"] = frequentist.benjamini_hochberg(out.p_value.to_numpy(), q=ALPHA)
    mde_abs = design.mde(
        final.count_c, final.count_t, np.sqrt(final.variance_c), np.sqrt(final.variance_t)
    )
    out["mde_rel"] = mde_abs / final.mean_c.abs()
    out["power_1pct"] = design.power(
        final.count_c,
        final.count_t,
        np.sqrt(final.variance_c),
        0.01 * final.mean_c.abs(),
        np.sqrt(final.variance_t),
    )
    return out


# ---------------------------------------------------------------------------
# 4. Replay under three stopping rules
# ---------------------------------------------------------------------------


def learn_prior(final: pd.DataFrame) -> dict[int, float]:
    """Spread of true standardized effects per metric, learned across experiments."""
    return {
        int(m): bayesian.estimate_prior_variance(g.d, np.sqrt(g.d_var))
        for m, g in final.groupby("metric_id")
    }


def replay(df: pd.DataFrame, tau2: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per series with each rule's verdict, checkpoints with mSPRT p-values)."""
    df = df.copy()
    df["p_msprt"] = np.nan
    for key, g in df.groupby(KEYS):
        df.loc[g.index, "p_msprt"] = sequential.msprt_pvalues(
            g.d.to_numpy(), g.d_var.to_numpy(), tau2[key[2]]
        )

    rows = []
    for key, g in df.groupby(KEYS):
        last = g.iloc[-1]
        naive_hits = g[g.p_naive < ALPHA]
        seq_hits = g[g.p_msprt < ALPHA]
        first_naive = naive_hits.iloc[0] if len(naive_hits) else None
        first_seq = seq_hits.iloc[0] if len(seq_hits) else None
        rows.append(
            dict(
                zip(KEYS, key, strict=True),
                n_checkpoints=len(g),
                fixed_sig=bool(last.p_naive < ALPHA),
                final_sign=np.sign(last["diff"]),
                naive_sig=first_naive is not None,
                naive_stop_frac=np.nan if first_naive is None else first_naive.frac_time,
                naive_sign=np.nan if first_naive is None else np.sign(first_naive["diff"]),
                msprt_sig=first_seq is not None,
                msprt_stop_frac=np.nan if first_seq is None else first_seq.frac_time,
                msprt_sign=np.nan if first_seq is None else np.sign(first_seq["diff"]),
            )
        )
    return pd.DataFrame(rows), df


# ---------------------------------------------------------------------------
# 5. Empirical-Bayes decisions on relative lift
# ---------------------------------------------------------------------------


def bayes_decisions(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in final.groupby("metric_id"):
        prior_var = bayesian.estimate_prior_variance(g.rel, g.rel_se)
        for r in g.itertuples():
            b = bayesian.normal_decision(r.rel, r.rel_se, prior_var=prior_var)
            if b.prob_treatment_better >= 0.95:
                decision = "ship"
            elif b.prob_treatment_better <= 0.05:
                decision = "roll back"
            else:
                decision = "inconclusive"
            rows.append(
                dict(
                    experiment_id=r.experiment_id,
                    variant_id=r.variant_id,
                    metric_id=m,
                    raw_rel_lift=r.rel,
                    posterior_rel_lift=b.posterior_mean,
                    prob_better=b.prob_treatment_better,
                    expected_loss_ship=b.expected_loss_ship,
                    prior_sd=np.sqrt(prior_var),
                    decision=decision,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_peeking_simulation() -> pd.DataFrame:
    looks = [1, 2, 3, 5, 10, 20, 50, 100]
    rows = []
    for k in looks:
        r = sequential.peeking_simulation(n_experiments=20_000, n_per_arm=100_000, n_looks=k)
        rows.append(dict(looks=k, **r))
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(res.looks, res.naive_peeking * 100, "o-", label="Naive peeking (t-test at every look)")
    ax.plot(res.looks, res.msprt * 100, "s-", label="mSPRT (always-valid p-value)")
    ax.axhline(5, color="grey", ls="--", lw=1, label="Promised 5% false-positive rate")
    ax.set_xscale("log")
    ax.set_xticks(looks, [str(x) for x in looks])
    ax.set_xlabel("Number of times results are checked during the test")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title("A/A simulation, 20,000 experiments per point")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "asos_peeking_simulation.png")
    plt.close(fig)
    return res


def fig_pvalue_hist(final_res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, (m, g) in zip(axes, final_res.groupby("metric_id"), strict=True):
        ax.hist(g.p_value, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
        ax.set_title(f"Metric {m}  ({g.significant.sum()}/{len(g)} with p<0.05)", fontsize=9)
        ax.set_xlabel("p-value")
    axes[0].set_ylabel("Experiment-variants")
    fig.suptitle("End-of-experiment Welch t-test p-values", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "asos_pvalue_hist.png")
    plt.close(fig)


def fig_mde(final_res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    data = [g.mde_rel.to_numpy() * 100 for _, g in final_res.groupby("metric_id")]
    ax.boxplot(data, tick_labels=[f"Metric {m}" for m in sorted(final_res.metric_id.unique())])
    ax.axhline(1, color="crimson", ls="--", lw=1, label="1% relative change")
    ax.set_yscale("log")
    ticks = [0.2, 0.5, 1, 2, 5]
    ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("Minimum detectable effect (log scale)")
    ax.set_title(
        "Smallest relative change each experiment could detect\n(80% power, alpha 0.05)",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "asos_mde.png")
    plt.close(fig)


def fig_example(series: pd.DataFrame, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = series.time_since_start
    ax1.plot(t, series.rel * 100, color="#4C72B0")
    ax1.fill_between(
        t,
        (series.rel - 1.96 * series.rel_se) * 100,
        (series.rel + 1.96 * series.rel_se) * 100,
        alpha=0.2,
        color="#4C72B0",
    )
    ax1.axhline(0, color="grey", lw=1)
    ax1.set_ylabel("Relative lift (%)\nwith 95% CI")
    ax1.set_title(title, fontsize=10)
    ax2.plot(t, series.p_naive, label="t-test p-value (what a peeker sees)")
    ax2.plot(t, series.p_msprt, label="mSPRT always-valid p-value")
    ax2.axhline(ALPHA, color="crimson", ls="--", lw=1, label="0.05")
    ax2.set_yscale("log")
    ax2.set_xlabel("Days since start")
    ax2.set_ylabel("p-value (log)")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "asos_example_experiment.png")
    plt.close(fig)


def fig_shrinkage(bayes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m, g in bayes.groupby("metric_id"):
        ax.scatter(g.raw_rel_lift * 100, g.posterior_rel_lift * 100, s=14, label=f"Metric {m}")
    lim = np.nanpercentile(np.abs(bayes.raw_rel_lift * 100), 98)
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=1, ls="--", label="No shrinkage")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Observed relative lift (%)")
    ax.set_ylabel("Empirical-Bayes estimate (%)")
    ax.set_title("Noisy results are pulled toward zero", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "asos_shrinkage.png")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    final = final_rows(df)

    srm = srm_analysis(df)
    final_res = final_analysis(final)
    tau2 = learn_prior(final)
    rep, df = replay(df, tau2)
    bayes = bayes_decisions(final)
    sim = fig_peeking_simulation()

    fig_pvalue_hist(final_res)
    fig_mde(final_res)
    fig_shrinkage(bayes)
    peek_only = rep[rep.naive_sig & ~rep.fixed_sig]
    first = peek_only.sort_values("naive_stop_frac").iloc[0]
    example_key = [first.experiment_id, int(first.variant_id), int(first.metric_id)]
    example = df[(df[KEYS] == example_key).all(axis=1)]
    fig_example(
        example,
        f"Experiment {example_key[0]}, variant {example_key[1]}, metric {example_key[2]}: "
        "looks like a win early, is not one at the end",
    )

    srm.to_csv(OUT / "srm_checks.csv", index=False)
    final_res.to_csv(OUT / "final_results.csv", index=False)
    rep.to_csv(OUT / "stopping_rules.csv", index=False)
    bayes.to_csv(OUT / "bayes_decisions.csv", index=False)
    sim.to_csv(OUT / "peeking_simulation.csv", index=False)

    by_metric = []
    for m in sorted(final_res.metric_id.unique()):
        fr, rp, bd = (x[x.metric_id == m] for x in (final_res, rep, bayes))
        by_metric.append(
            dict(
                metric=int(m),
                series=len(fr),
                fixed_significant=int(fr.significant.sum()),
                bh_discoveries=int(fr.bh_discovery.sum()),
                naive_peeking_significant=int(rp.naive_sig.sum()),
                msprt_significant=int(rp.msprt_sig.sum()),
                median_mde_rel_pct=float(fr.mde_rel.median() * 100),
                share_powered_for_1pct=float((fr.power_1pct >= 0.8).mean()),
                bayes_ship=int((bd.decision == "ship").sum()),
                bayes_rollback=int((bd.decision == "roll back").sum()),
                bayes_inconclusive=int((bd.decision == "inconclusive").sum()),
                tau2=tau2[int(m)],
            )
        )
    both = rep[rep.fixed_sig & rep.msprt_sig]
    results = dict(
        n_experiments=int(df.experiment_id.nunique()),
        n_experiment_variants=int(df.groupby(["experiment_id", "variant_id"]).ngroups),
        n_series=len(rep),
        srm=dict(
            tested_50_50=int((srm.assumed_design == "50/50").sum()),
            unequal_designs=int((srm.assumed_design != "50/50").sum()),
            srm_flags=int(srm.srm_flag.sum()),
            drift_flags=int(srm.drift_flag.sum()),
        ),
        fixed_significant=int(rep.fixed_sig.sum()),
        bh_discoveries=int(final_res.bh_discovery.sum()),
        naive_peeking_significant=int(rep.naive_sig.sum()),
        peeking_only_wins=len(peek_only),
        peeking_only_sign_flips=int((peek_only.naive_sign != peek_only.final_sign).sum()),
        msprt_significant=int(rep.msprt_sig.sum()),
        msprt_and_fixed=len(both),
        msprt_not_fixed=int((rep.msprt_sig & ~rep.fixed_sig).sum()),
        fixed_not_msprt=int((rep.fixed_sig & ~rep.msprt_sig).sum()),
        msprt_median_stop_frac_when_both=float(both.msprt_stop_frac.median()),
        simulation=sim.to_dict(orient="records"),
        example_series=example_key,
        by_metric=by_metric,
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(
        json.dumps({k: v for k, v in results.items() if k != "simulation"}, indent=2, default=float)
    )


if __name__ == "__main__":
    main()
