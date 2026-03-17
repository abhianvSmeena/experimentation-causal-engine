"""``abtest``: run the full analysis on a simulated experiment with a known true lift.

Example:
    abtest --n 20000 --lift 0.05
"""

from __future__ import annotations

import argparse

import numpy as np

from experiment_toolkit import bayesian, checks, cuped, design, frequentist, sequential
from experiment_toolkit.simulate import simulate_experiment


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def run(args: argparse.Namespace) -> dict[str, float]:
    df = simulate_experiment(
        n_per_arm=args.n,
        baseline_rate=args.baseline,
        relative_lift=args.lift,
        pre_post_corr=args.rho,
        seed=args.seed,
    )
    c, t = df[~df.treated], df[df.treated]
    print(
        f"Simulated {len(df):,} users | true relative lift = {args.lift:+.1%} | "
        f"baseline conversion = {args.baseline:.1%}"
    )

    _section("1. Design")
    n_c, n_t = design.sample_size_proportions(args.baseline, args.lift)
    print(f"Users per arm needed to detect {args.lift:+.1%} on conversion (80% power): {n_c:,}")
    print(f"Users per arm in this experiment: {len(c):,} / {len(t):,}")
    sd = np.sqrt(args.baseline * (1 - args.baseline))
    achieved = design.power(len(c), len(t), sd, args.baseline * args.lift)
    warning = (
        "  <- UNDERPOWERED: a non-significant result is weak evidence" if achieved < 0.8 else ""
    )
    print(f"Achieved power for the true lift: {achieved:.0%}{warning}")

    _section("2. Sample ratio mismatch check")
    srm = checks.srm_test([len(c), len(t)])
    verdict = "MISMATCH, do not trust results" if srm.mismatch else "OK"
    print(
        f"Split {srm.observed_shares[0]:.2%} / {srm.observed_shares[1]:.2%}, "
        f"p = {srm.p_value:.3f} -> {verdict}"
    )

    _section("3. Fixed-horizon tests")
    conv = frequentist.proportion_test(
        int(c.converted.sum()), len(c), int(t.converted.sum()), len(t)
    )
    print(
        f"Conversion: lift {conv.rel_lift:+.2%} "
        f"(95% CI {conv.rel_ci_low:+.2%} to {conv.rel_ci_high:+.2%}), p = {conv.p_value:.4f}"
    )

    _section("4. CUPED on spend (pre-period spend as covariate)")
    cr = cuped.cuped_test(
        c.spend.to_numpy(), c.pre_spend.to_numpy(), t.spend.to_numpy(), t.pre_spend.to_numpy()
    )
    print(
        f"Raw:   lift {cr.raw.rel_lift:+.2%} "
        f"(CI {cr.raw.rel_ci_low:+.2%} to {cr.raw.rel_ci_high:+.2%}), p = {cr.raw.p_value:.2e}"
    )
    print(
        f"CUPED: lift {cr.adjusted.rel_lift:+.2%} (CI {cr.adjusted.rel_ci_low:+.2%} to "
        f"{cr.adjusted.rel_ci_high:+.2%}), p = {cr.adjusted.p_value:.2e}"
    )
    print(f"Variance reduction: {cr.variance_reduction:.1%}")

    _section(f"5. Sequential test (mSPRT), {args.looks} looks")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(df))
    y, treated = df.converted.to_numpy()[order], df.treated.to_numpy()[order]
    looks = np.linspace(len(df) / args.looks, len(df), args.looks).astype(int)
    diffs, vars_ = [], []
    for m in looks:
        yc, yt = y[:m][~treated[:m]], y[:m][treated[:m]]
        diffs.append(yt.mean() - yc.mean())
        vars_.append(yc.var(ddof=1) / len(yc) + yt.var(ddof=1) / len(yt))
    tau2 = (args.baseline * args.lift) ** 2
    p_seq = sequential.msprt_pvalues(np.array(diffs), np.array(vars_), tau2)
    stop = int(np.argmax(p_seq < 0.05)) if (p_seq < 0.05).any() else None
    print("Always-valid p-value by look: " + ", ".join(f"{p:.3f}" for p in p_seq))
    if stop is None:
        print("Never crossed p < 0.05: keep running or accept that the effect is too small to see")
    else:
        print(f"Can stop at look {stop + 1} ({looks[stop]:,} users) with error control intact")

    _section("6. Bayesian decision (Beta-Binomial)")
    b = bayesian.beta_binomial_ab(
        int(c.converted.sum()), len(c), int(t.converted.sum()), len(t), seed=args.seed
    )
    print(f"P(treatment better) = {b.prob_treatment_better:.1%}")
    print(f"Expected loss if we ship = {b.expected_loss_ship:.2e} (conversion-rate units)")
    decision = "SHIP" if b.prob_treatment_better > 0.95 else "DON'T SHIP YET"
    print(f"Decision rule P > 95%: {decision}")

    return {
        "conversion_p": conv.p_value,
        "cuped_variance_reduction": cr.variance_reduction,
        "prob_better": b.prob_treatment_better,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--n", type=int, default=20_000, help="users per arm")
    parser.add_argument("--baseline", type=float, default=0.10, help="control conversion rate")
    parser.add_argument("--lift", type=float, default=0.05, help="true relative lift")
    parser.add_argument("--rho", type=float, default=0.7, help="pre/post spend correlation")
    parser.add_argument("--looks", type=int, default=10, help="number of interim looks")
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
