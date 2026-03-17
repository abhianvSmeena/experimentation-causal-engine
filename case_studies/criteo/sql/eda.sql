-- Exploratory checks on the Criteo uplift dataset, run with DuckDB over the parquet file.
-- Each query is preceded by a "-- name:" line; run_all.py executes them one by one.

-- name: arm_summary
-- Group sizes and outcome rates per arm. Treatment = eligible to see the ad (intent-to-treat).
SELECT
    treatment,
    COUNT(*)                         AS users,
    COUNT(*) / SUM(COUNT(*)) OVER () AS share,
    AVG(visit)                       AS visit_rate,
    AVG(conversion)                  AS conversion_rate,
    AVG(exposure)                    AS exposure_rate
FROM criteo
GROUP BY treatment
ORDER BY treatment;

-- name: naive_ate
-- Difference in means with a normal-approximation 95% CI, for both outcomes.
WITH s AS (
    SELECT
        AVG(visit) FILTER (WHERE treatment = 1)           AS v1,
        AVG(visit) FILTER (WHERE treatment = 0)           AS v0,
        AVG(conversion) FILTER (WHERE treatment = 1)      AS c1,
        AVG(conversion) FILTER (WHERE treatment = 0)      AS c0,
        COUNT(*) FILTER (WHERE treatment = 1)             AS n1,
        COUNT(*) FILTER (WHERE treatment = 0)             AS n0
    FROM criteo
)
SELECT 'visit' AS outcome, v0 AS control_rate, v1 AS treated_rate, v1 - v0 AS ate,
       1.96 * SQRT(v1 * (1 - v1) / n1 + v0 * (1 - v0) / n0) AS ci_half_width,
       v1 / v0 - 1 AS relative_lift
FROM s
UNION ALL
SELECT 'conversion', c0, c1, c1 - c0,
       1.96 * SQRT(c1 * (1 - c1) / n1 + c0 * (1 - c0) / n0),
       c1 / c0 - 1
FROM s;

-- name: funnel
-- Conversions almost always happen after a visit; share of conversions without a visit.
SELECT
    SUM(conversion)                                       AS conversions,
    SUM(conversion * (1 - visit))                         AS conversions_without_visit,
    AVG(conversion) FILTER (WHERE visit = 1)              AS conversion_rate_given_visit
FROM criteo;

-- name: exposure_by_arm
-- Control users should never be exposed; treated users are exposed only sometimes.
SELECT treatment, AVG(exposure) AS exposure_rate,
       AVG(visit) FILTER (WHERE exposure = 1) AS visit_rate_if_exposed,
       AVG(visit) FILTER (WHERE exposure = 0) AS visit_rate_if_not_exposed
FROM criteo
GROUP BY treatment
ORDER BY treatment;

-- name: split_sizes
SELECT split, COUNT(*) AS users, AVG(treatment) AS treated_share, AVG(visit) AS visit_rate
FROM criteo
GROUP BY split
ORDER BY split;
