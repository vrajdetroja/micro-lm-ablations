# 001 — Seed variance of the baseline (the noise floor)

*Committed before any run of this experiment.*

**Configs:** `configs/baseline.yaml` (10.62M non-embedding params, 212M tokens).
**Protocol:** `experiments/protocol/01_noise_floor.sh`.
**Protocol:** LR sweep {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2} at seed 0, then `train.seed` ∈ {1, 2, 3, 4}
at the best LR, giving 5 seeds at the optimum. Optionally, `train.data_seed`
∈ {1, 2} at seed 0 to separate data-order noise from initialization noise.
**Metric:** final val_iid loss (nats/token) over the full ~10M-token split.
**Matching:** everything identical except the seed.

## Hypothesis
Seed-to-seed variation in final val loss is small but not negligible next to typical
architecture effects at this scale.

## Prediction
- Best LR: 4e-3 or 8e-3 (small models tolerate high AdamW LRs).
- σ (init seed): between 0.004 and 0.015 nats. That is wider than a 25M-param model would
  give, because smaller models are noisier.
- Eval sampling error (~0.0005 nats at 10M tokens) is well below σ, so σ measures training
  noise rather than measurement noise.

## Meaningful effect size
Not applicable: this experiment *defines* the threshold (2σ) used by every other one.

## What would falsify it
σ > 0.02 nats would mean most planned ablations cannot be resolved with 3 seeds. The
response would be more seeds on fewer ablations, not a lower bar.

## Outcome *(filled in after)*
