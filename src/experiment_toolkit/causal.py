"""Average treatment effect estimation with DoWhy, plus refutation tests.

A refutation test changes the problem in a way that has a known answer and checks the
estimator gets it: a placebo treatment should give an effect near 0, adding a random
confounder or using a random subset of the data should leave the estimate unchanged.
DoWhy is an optional dependency (install with the ``case-studies`` extra).
"""

from __future__ import annotations

import pandas as pd


def estimate_and_refute(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: list[str],
    method: str = "backdoor.linear_regression",
    num_simulations: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    from dowhy import CausalModel

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name=method)

    refuters = {
        "placebo_treatment": dict(method_name="placebo_treatment_refuter", placebo_type="permute"),
        "random_common_cause": dict(method_name="random_common_cause"),
        "data_subset": dict(method_name="data_subset_refuter", subset_fraction=0.5),
    }
    results: dict[str, object] = {"ate": float(estimate.value), "refutations": {}}
    for name, kwargs in refuters.items():
        ref = model.refute_estimate(
            estimand, estimate, num_simulations=num_simulations, random_seed=seed, **kwargs
        )
        p_value = ref.refutation_result.get("p_value") if ref.refutation_result else None
        results["refutations"][name] = {
            "new_effect": float(ref.new_effect),
            "p_value": None if p_value is None else float(p_value),
        }
    return results
