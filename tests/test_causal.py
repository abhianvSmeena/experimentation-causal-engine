import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy")

from experiment_toolkit.causal import estimate_and_refute  # noqa: E402


def test_estimate_and_refute_on_known_effect():
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["t"] = (rng.random(n) < 0.5).astype(float)
    df["y"] = 0.3 * df.t + df.x1 + rng.normal(size=n)
    result = estimate_and_refute(df, "t", "y", ["x1", "x2"], num_simulations=5)
    assert result["ate"] == pytest.approx(0.3, abs=0.05)
    assert abs(result["refutations"]["placebo_treatment"]["new_effect"]) < 0.05
    assert result["refutations"]["random_common_cause"]["new_effect"] == pytest.approx(
        result["ate"], abs=0.01
    )
