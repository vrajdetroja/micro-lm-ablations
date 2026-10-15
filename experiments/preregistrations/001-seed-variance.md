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

## Outcome *(filled in 2026-10-16)*

**LR sweep, seed 0 (val_iid nats):** 1e-3 3.8000 · 2e-3 3.6326 · **4e-3 3.5909** · 8e-3 3.6005 ·
1.6e-2 5.7084 (unstable, did not recover). The optimum is interior, so no extension was needed.
Prediction "4e-3 or 8e-3": **held** (4e-3).

**Seeds 0–4 at 4e-3:** 3.5909, 3.5883, 3.5964, 3.5915, 3.5870. Mean 3.5908, range 0.0094.
**σ_seed = 0.0036 nats** (0.0013 bits per byte); 95% CI ≈ [0.0022, 0.0104] (χ², 4 df).
Prediction 0.004–0.015: **missed, low.** The noise floor is tighter than predicted, so smaller
effects are resolvable. The CI is wide, though: five seeds bound σ only within about 3×.

**Same-seed repeats:** 3.5909 (the seed-0 run), 3.5880, 3.5853. **σ_kernel = 0.0028 nats,
78% of σ_seed**; 95% CI ≈ [0.0015, 0.018] (χ², 2 df, coarse). Prediction 50–100%: **held.**
GPU kernel non-determinism alone produces most of the run-to-run spread, in line with
Summers & Dinneen (2021). Falsifier (< 25%): not triggered.

**Resume check on the GPU:** PASS (mean |Δ train loss| after resuming 0.0103 against the
same-seed repeat's 0.0063).

**LR-selection bias:** the mean of the four fresh seeds (3.5908) equals the five-seed mean
(3.5908). Choosing the LR on seed 0 introduced no visible bias at this σ.

**Threshold for every later ablation: 2σ = 0.0072 nats.** σ > 0.02 falsifier: not triggered.

**Predictions: 2 of 3 held** (best LR, σ_kernel share). σ itself came in below the range.

**Forward note** (to be preregistered with Tier A, not applied here): every Tier A variant gets
3 seeds at its best LR. Pooling the within-variant variance across 16 variants gives σ with
32 degrees of freedom instead of 4, a much tighter estimate than these five seeds alone.
