# Memo: whom to show the ad to

**To:** Marketing and growth leads
**From:** Data Science
**Based on:** a randomized ad experiment with 14 million users (public Criteo dataset).
Full report: [`reports/criteo_report.md`](../reports/criteo_report.md).

## The question

We can't afford to show the ad to everyone. Who should get it?

## What we found

1. **The ad works, but less than the raw numbers say.** Visit rate goes from 3.82% to about
   4.55%: **+0.73 points (+19%)**, not the +1.03 points (+27%) a simple comparison shows. The
   treated group was slightly more active to begin with, and we corrected for that.
2. **Targeting matters a lot.** If we pick users at random, every 1,000 ads bring about 6 extra
   visits. If we pick the top 10% by our uplift model, every 1,000 ads bring about **60 extra
   visits**, ten times more.
3. **"Most likely to visit" is not the same as "visits because of the ad".** The usual approach of
   targeting the most active users gives 52 extra visits per 1,000. The uplift model gives 60:
   **16% more from the same budget**.
4. **A small budget goes far.** The top 20% of users picked by the model account for **78% of all
   the extra visits** the ad could produce.
5. **For purchases, keep it simple.** Purchases are rare (0.3%), and the simple "most likely to
   buy" model was the best way to target them. The more complex models did not beat it.

## What we recommend

| Goal | Targeting method |
|---|---|
| Drive visits, budget for 10 to 20% of users | Uplift model (X-learner) |
| Drive visits, budget for 30%+ of users | Any good model; they perform the same |
| Drive purchases | "Most likely to buy" response model |
| Report campaign impact | Use the covariate-adjusted effect, not the raw difference |

## What it is worth (illustrative)

If an impression costs ₹0.50 and a visit is worth ₹10, then per 1,000 ads:
- random targeting **loses ₹440**
- response-model targeting **makes ₹19**
- uplift-model targeting **makes ₹101**

Replace these with real costs before deciding.

## Before rollout

Validate with a fresh A/B test: model-targeted vs response-targeted, same budget. This report
uses historical data; a live test confirms the gain under current conditions.
