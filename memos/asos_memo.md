# Memo: how we should read our A/B tests

**To:** Product and growth leads
**From:** Data Science
**Based on:** re-analysis of 78 real experiments (381 experiment-metric results) from the public
ASOS experiments dataset. Full report: [`reports/asos_report.md`](../reports/asos_report.md).

## The problem

If you check an experiment's dashboard every day and ship as soon as it shows p < 0.05, you will
ship many changes that do nothing, and some that hurt.

## What we found

1. **Daily peeking doubles the "wins".** Checking every day would have declared 224 results
   significant. Testing once at the planned end gives 105.
2. **Some of those wins were losses.** 119 results looked significant at some point but were not
   at the end, and **36 of them had reversed direction by the end of the test**.
3. **Early numbers are the least reliable.** The median false win appeared 13% of the way into
   the test.
4. **Data checks catch broken tests.** 9 experiments had a user split that did not match a 50/50
   design, and 5 had a split that changed during the test. Their results should not be used until
   explained.

## What we recommend

| Situation | Do this |
|---|---|
| You want to stop early when results look clear | Use the **sequential test (mSPRT)** p-value, not the normal p-value. It is safe to check daily. Clear effects were caught after a median of **24% of the planned duration**. |
| You will only look at the end | Fix the sample size up front with the power calculator and look once. |
| You track many metrics or variants | Treat a single p < 0.05 among many as weak evidence; we use false discovery rate control. |
| Every experiment | Run the automatic **sample ratio check** before reading results. |
| Talking to stakeholders | Report "probability it helps" and "expected cost if we are wrong" (Bayesian view) alongside the p-value. |

## The cost of doing nothing

At 10 daily looks, about **1 in 5** experiments with no real effect will show a "significant"
result by chance (simulation of 20,000 tests). With the sequential test, fewer than 1 in 100 do.
