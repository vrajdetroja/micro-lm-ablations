# 001 — Seed variance of the baseline (the noise floor)

*Committed before any run of this experiment.*

**Configs:** `configs/baseline.yaml`, `train.seed` ∈ {0, 1, 2, 3, 4}, at the baseline's best LR
from the baseline LR sweep. Optionally a second set with `train.data_seed` ∈ {1, 2} at seed 0 to
separate data-order noise from initialization noise.
**Metric:** final val_iid loss (nats/token), full ~10M-token split.
**Matching:** everything identical except the seed.

## Hypothesis
Seed-to-seed variation in final val loss is small but not negligible next to typical
architecture effects at this scale.

## Prediction
σ (init seed) between 0.003 and 0.010 nats. Data-order σ of a similar or larger size.
With 10M eval tokens the eval sampling error (~0.0005 nats) is well below σ, so σ
measures training noise rather than measurement noise.

## Meaningful effect size
Not applicable: this experiment *defines* the threshold (2σ) used by every other one.

## What would falsify it
σ > 0.02 nats would mean most planned ablations cannot be resolved with 3 seeds. The
response would be more seeds or longer runs, not a lower bar.

## Outcome *(filled in after)*
