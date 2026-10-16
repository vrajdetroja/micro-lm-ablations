# 006 — Depth vs width

*Committed 2026-10-16, before any run of Tier A. Shared rules: see 002.*

**Configs** (d_head fixed at 64; the baseline is 6 × 384, 10.62M non-embedding, 1.00× FLOPs):

| Config | Layers × width | Aspect d/L | Non-embedding params | FLOPs vs baseline | Measured tokens/s |
|---|---|---|---|---|---|
| `dw_L2_d640` | 2 × 640 | 320 | 9.83M (−7%) | 1.02× | 1,046k (62% MFU) |
| baseline | 6 × 384 | 64 | 10.62M | 1.00× | 935k (55%) |
| `dw_L12_d256` | 12 × 256 | 21 | 9.44M (−11%) | 0.91× | not measured |
| `dw_L24_d192` | 24 × 192 | 8 | 10.63M (+0%) | 1.05× | 704k (43%) |

Widths must be multiples of 64 for whole heads, so parameters match only to within about
11%. Every comparison reports N alongside, and the 12-layer model is under-parameterized
relative to the rest.
**Seeds × LRs:** shared protocol, 5 runs per variant.
**Metric:** val_iid loss (shared rule 2), reported three ways:
- **Equal steps** (equal tokens): the final value. This is the verdict metric.
- **Equal FLOPs:** each run's periodic val curve (1M-token evals every 200 steps),
  interpolated log-linearly in cumulative FLOPs at the baseline's total. Interpolation only;
  no extrapolation past a run's last eval.
- **Equal wallclock:** the same interpolation in measured training seconds.

The periodic evals use 1M tokens, so they are noisier than the final 10M-token eval; equal-
FLOPs and equal-wallclock values are reported with that caveat. The induction probe (copy
accuracy) is reported for each shape.

## Hypothesis
The optimum is broad (Kaplan et al., 2020). The 2-layer model is clearly too shallow, while
12 and 24 layers sit near the baseline per token. But the deep-narrow model gets 22% lower
MFU on this GPU (43% against 55%; 25% fewer tokens per second, with 5% more FLOPs per token),
so the ranking changes when wallclock is matched.

## Prediction
Equal steps, Δ against the baseline (3.5908):
- 2 × 640: Δ ≥ +0.04. Two layers are the minimum for induction heads (Olsson et al., 2022),
  so its copy accuracy will also be lower.
- 12 × 256: |Δ| < 0.01, despite 11% fewer parameters.
- 24 × 192: Δ between −0.01 and +0.02.

Equal wallclock: 24 × 192 is ≥ 0.02 worse than the baseline at the baseline's training time.
Equal FLOPs: 12 × 256 ranks at least as well as it does at equal steps, since it spends 9%
fewer FLOPs.

## Meaningful effect size
2σ (0.0072) at equal steps, under shared rule 3. The equal-FLOPs and equal-wallclock rankings
are descriptive; a reversal relative to equal steps is the finding to look for.

## What would falsify it
- 2 × 640 within 2σ of the baseline: depth barely matters, even at two layers.
- 24 × 192 better than the baseline by ≥ 2σ at equal steps: deeper is better at this scale.
- No reversal between the equal-steps and equal-wallclock rankings for 24 × 192.

## Amendment — 2026-10-16, after Tier A started, before any 006 run

Tier A runs on a different RTX 4090 from the noise-floor runs, a 48 GB card. The same throughput
benchmark (baseline, micro-batch 64, default compile) gives 970k tokens/s on it against
926k on the 24 GB card used for the noise floor: 4.7% faster. All 006 variants train on the Tier A card, but
the baseline's five runs trained on the 24 GB card. For the **equal-wallclock** comparison,
the baseline's measured training seconds are therefore scaled by 926/970 = 0.955 to this
card. Equal-steps and equal-FLOPs comparisons are unaffected: the kernels are the same, and
their run-to-run noise is already part of σ. If Tier A runs on more than one card, each
card's own benchmark sets its scale factor the same way.

## Outcome *(filled in 2026-10-16)*

Δ against the 6 × 384 baseline. Equal-FLOPs and equal-wallclock deltas are interpolated from
the periodic val curves (first 1M val_iid tokens), with both runs read at the same FLOPs or wallclock
(`analysis/tier_a_secondary.py`).

| Shape | Best LR | Mean | Δ steps | Δ FLOPs | Δ wallclock | Verdict (written / post hoc) | Predicted Δ steps |
|---|---|---|---|---|---|---|---|
| 2 × 640 | 2e-3 | 3.6494 | +0.059 | +0.062 | +0.041 | threshold-sensitive / **worse** | ≥ +0.04: **held** |
| **6 × 384** | 4e-3 | 3.5908 | — | — | — | — | — |
| 12 × 256 | 8e-3 | 3.6999 | +0.109 | +0.088 | +0.124 | threshold-sensitive / **worse** | \|Δ\| < 0.01: **missed** |
| 24 × 192 | 8e-3 | 3.7563 | +0.165 | +0.178 | +0.278 | worse / **worse** | −0.01 to +0.02: **missed** |

- **The optimum is not broad here.** 6 layers beats 2, 12, and 24 at equal steps, FLOPs, and
  wallclock, with no rank reversals. Kaplan et al.'s broad optimum does not appear at 10M,
  with this fixed head size.
- **Caveat:** with d_head fixed at 64, depth is confounded with head count (2 × 640 has 10
  heads; 6 × 384 has 6; 12 × 256 has 4; 24 × 192 has 3). The 12-layer model also has 11%
  fewer parameters.
- **Equal wallclock, 24 × 192 ≥ 0.02 worse: held** (+0.278).
- **Equal FLOPs, 12 × 256 ranks at least as well: held in degree.** Its gap narrows from
  +0.109 to +0.088, but its rank is unchanged.
- **Falsifier "no reversal for 24 × 192": triggered.** The predicted reversal needed it to be
  close to the baseline per step, which it was not.
- **Induction probe (not a verdict metric).** Copy accuracy on repeated random tokens:
  2 × 640 0.049 (lowest: **held**), 6 × 384 0.097, 12 × 256 0.136, 24 × 192 0.123.
  In-context gain: 4.08, 4.37, 5.35, 5.32. **Deeper models copy better while modelling
  language worse.** At fixed parameters, depth buys in-context copying at the cost of
  next-token loss.

