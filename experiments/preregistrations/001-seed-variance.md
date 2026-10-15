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

## Amendment — 2026-10-16, before any noise-floor run

**Why.** Validation runs on an RTX 4090 (`experiments/protocol/00_validation.sh`) found that training is
not deterministic. Two fresh runs with the same seed and config were bitwise equal for 2
steps, differed by ~5e-6 at step 3 (non-deterministic kernel reductions), and by 0.0065
nats on average over steps 81–100 as warmup amplified the difference. So "seed 0" names a
distribution of runs, not a single run.

**Added measurement.** Two more runs of the best LR at seed 0 (`baseline_repeat`, same
config), giving σ_kernel from 3 same-seed runs next to σ_seed from seeds 0–4. Same metric.
n = 3 is a coarse estimate; it is reported with that caveat, not as a precise number.

**Prediction.** σ_kernel is 50–100% of σ_seed. Small optimization trajectories are chaotic
enough that kernel-level differences decorrelate runs almost as much as a new seed.
Summers & Dinneen (2021) found that a single-bit change to the initialization produces
variability comparable to a full seed change, though in image classification, not language
modelling.

**Falsifier.** σ_kernel < 25% of σ_seed: then initialization and data order dominate, and
kernel noise is a footnote.

**Unchanged.** The decision rule. σ is still the standard deviation across seeds, which
already includes kernel noise, and every comparison is against 2σ.

**Resume check.** The noise-floor experiment also checks resumption on the GPU. With kernel noise, "resumed equals uninterrupted" is the wrong test. The
criterion is that after the resume point, a resumed run stays as close to an uninterrupted
run as a fresh same-seed repeat does: mean |Δ train loss| ≤ 3 × the repeat's + 0.005
(`analysis/check_resume.py`). On a second accelerator a correct resume passed (0.0008 against
0.0006) and a deliberately broken one, with optimizer state wiped, failed (0.050).

## Outcome *(filled in after)*
