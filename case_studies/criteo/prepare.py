"""Convert the Criteo CSV to parquet, add a deterministic train/test split, and run EDA SQL.

Run:  python case_studies/criteo/prepare.py
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from experiment_toolkit import checks

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PARQUET = ROOT / "data" / "processed" / "criteo.parquet"
SQL = Path(__file__).parent / "sql" / "eda.sql"
OUT = ROOT / "reports" / "criteo"
FEATURES = [f"f{i}" for i in range(12)]
TEST_SHARE_PCT = 30


def build_parquet(con: duckdb.DuckDBPyConnection) -> None:
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features = ", ".join(f"CAST({f} AS FLOAT) AS {f}" for f in FEATURES)
    con.execute(
        f"""
        COPY (
            SELECT {features},
                   CAST(treatment AS TINYINT) AS treatment,
                   CAST(conversion AS TINYINT) AS conversion,
                   CAST(visit AS TINYINT) AS visit,
                   CAST(exposure AS TINYINT) AS exposure,
                   CASE WHEN hash(row_id) % 100 < {TEST_SHARE_PCT} THEN 'test' ELSE 'train' END
                       AS split
            FROM (SELECT *, row_number() OVER () AS row_id FROM read_csv_auto('{RAW}'))
        ) TO '{PARQUET}' (FORMAT PARQUET)
        """
    )


def run_eda(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    con.execute(f"CREATE OR REPLACE VIEW criteo AS SELECT * FROM '{PARQUET}'")
    blocks = re.split(r"^-- name: (\w+)\s*$", SQL.read_text(), flags=re.M)[1:]
    results = {}
    for name, query in zip(blocks[::2], blocks[1::2], strict=True):
        results[name] = con.execute(query).df()
        print(f"\n[{name}]\n{results[name].to_string(index=False)}")
    return results


def balance_check(con: duckdb.DuckDBPyConnection, n: int = 1_000_000) -> pd.DataFrame:
    """Randomization check: feature distributions should not differ between arms."""
    sample = con.execute(
        f"SELECT {', '.join(FEATURES)}, treatment FROM criteo USING SAMPLE {n} ROWS (reservoir, 7)"
    ).df()
    smd = checks.covariate_balance(sample[FEATURES].to_numpy(), sample.treatment.to_numpy())
    return pd.DataFrame({"feature": FEATURES, "standardized_mean_diff": smd})


def propensity_auc(con: duckdb.DuckDBPyConnection) -> float:
    """How well can features predict who was treated? 0.5 means perfectly random assignment."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def sample(split: str, n: int) -> pd.DataFrame:
        cols = ", ".join(FEATURES + ["treatment"])
        return con.execute(
            f"SELECT * FROM (SELECT {cols} FROM criteo WHERE split = '{split}') "
            f"USING SAMPLE {n} ROWS (reservoir, 1)"
        ).df()

    train, test = sample("train", 2_000_000), sample("test", 1_000_000)
    model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    model.fit(train[FEATURES], train.treatment)
    return float(roc_auc_score(test.treatment, model.predict_proba(test[FEATURES])[:, 1]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    if not PARQUET.exists():
        print("Converting CSV to parquet (one-off, ~1 minute) ...")
        build_parquet(con)
    results = run_eda(con)
    results["balance"] = balance_check(con)
    print(f"\n[balance]\n{results['balance'].to_string(index=False)}")
    arms = results["arm_summary"].set_index("treatment").users
    srm = checks.srm_test([arms[0], arms[1]], expected_shares=[0.15, 0.85])
    print(
        f"\nSRM vs planned 15/85: treated share {srm.observed_shares[1]:.7f}, p = {srm.p_value:.3g}"
    )
    auc = propensity_auc(con)
    print(f"Propensity model AUC (treatment from features, held-out): {auc:.4f}")
    results["randomization"] = pd.DataFrame(
        [
            {
                "observed_treated_share": srm.observed_shares[1],
                "srm_p_value_vs_85pct": srm.p_value,
                "propensity_auc": auc,
                "max_abs_smd": float(np.abs(results["balance"].standardized_mean_diff).max()),
            }
        ]
    )
    for name, frame in results.items():
        frame.to_csv(OUT / f"eda_{name}.csv", index=False)


if __name__ == "__main__":
    main()
