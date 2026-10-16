# 005 — Learning-rate schedule

*Committed 2026-10-16, before any run of Tier A. Shared rules: see 002.*

**Configs** (warmup 200 steps in all; the baseline is cosine to 10% of the peak):
- `configs/ablations/sched_cosine0.yaml`: cosine to 0.
- `sched_linear.yaml`: linear decay to 0.
- `sched_wsd.yaml`: warmup–stable–decay, flat until the last 20% of steps, then linear to 0.

Architecture, data, and number of training tokens are identical to the baseline.
**Seeds × LRs:** shared protocol, 5 runs per variant. The best LR is part of the result:
schedules that end at zero may tolerate a higher peak.
**Metric:** val_iid loss (shared rule 2).

## Hypothesis
Decaying all the way to zero beats stopping at 10%. Linear decay to zero is at least as good
as cosine to zero. WSD with a 20% cooldown matches cosine.

## Prediction
Δ against cosine-to-10% (3.5908):
- Cosine to 0: −0.003 to −0.012. It may not clear 2σ.
- Linear to 0: −0.005 to −0.015, and ≤ cosine to 0 (the gap may be within noise).
- WSD: within 2σ of cosine to 0.
- Best peak LRs of the decay-to-zero schedules ≥ the baseline's 4e-3.

Bergsma et al. (2025) argue linear decay to zero works best for compute-optimal LLM
training, and Hägele et al. (2024) find WSD with a short cooldown matches cosine. Both work
at much larger scale than 10M.

## Meaningful effect size
2σ (0.0072), under shared rule 3. Effects of the predicted size may land as inconclusive,
and that is a legitimate outcome, not a failure of the experiment.

## What would falsify it
- The baseline (cosine to 10%) beats both decay-to-zero schedules by ≥ 2σ.
- WSD differs from cosine to 0 by ≥ 2σ in either direction.

## Outcome *(filled in 2026-10-16)*

All best at LR 4e-3. Verdicts under the rule as written / the post-hoc pool (see 002's Outcome).

| Schedule | Mean | Δ against cosine to 10% | Verdict | Predicted Δ |
|---|---|---|---|---|
| Cosine to 0 | 3.6022 | **+0.011** | threshold-sensitive / threshold-sensitive | −0.003 to −0.012 |
| Linear to 0 | 3.5870 | −0.004 | inconclusive / inconclusive | −0.005 to −0.015 |
| **WSD** (20% linear cooldown) | 3.5552 | **−0.036** | threshold-sensitive / **better** | within 2σ of cosine to 0 |

- **Cosine to 0: missed, wrong direction.** It is slightly *worse* than stopping at 10%.
- **Linear to 0:** **missed** the range, inconclusive. "≤ cosine to 0" **held**.
- **WSD within 2σ of cosine to 0: falsified.** WSD is the best schedule tested, 0.047 better
  than cosine to 0. Its flat peak followed by a short, steep cooldown suits this short
  training run better than any smooth decay.
- **Best LRs ≥ 4e-3: held** (all at 4e-3).
- **Falsifiers:** "the baseline beats both decay-to-zero schedules by ≥ 2σ" was not triggered
  (it beats cosine to 0 but not linear). "WSD differs from cosine to 0 by ≥ 2σ" **was
  triggered.**

