# 004 — Normalization

*Committed 2026-10-16, before any run of Tier A. Shared rules: see 002.*

**Configs:** `configs/ablations/norm_rmsnorm.yaml` (RMSNorm instead of LayerNorm),
`norm_post.yaml` (post-LN: normalize after each residual add), and `norm_no_final.yaml`
(pre-LN without the final norm before the head), against pre-LN LayerNorm with a final norm.
All have 10.62M non-embedding params.
**Seeds × LRs:** shared protocol, 5 runs per variant.
**Metric:** val_iid loss (shared rule 2); throughput reported for RMSNorm.

**Divergence demonstration** (`runs/warmup_demo/`, kept out of the variant table):
- Post-LN at warmup ∈ {0, 50, 800} × LR ∈ {4e-3, 8e-3}, seed 0.
- Pre-LN control at warmup 0 × the same LRs.

That is 8 runs. The warmup-200 reference points come from the protocol's own seed-0 runs:
post-LN from `norm_post`, pre-LN from the baseline (3.5909 at 4e-3, 3.6005 at 8e-3).

## Hypothesis
(a) RMSNorm and LayerNorm are interchangeable. (b) In pre-LN, the final norm before the head
matters. (c) At 6 layers, post-LN trains with standard warmup and comes close to pre-LN at
its own best LR, but it is fragile: with little warmup and a high LR it fails where pre-LN
does not.

## Prediction
- RMSNorm: |Δ| < 2σ (inconclusive); throughput 0–3% higher.
- No final norm: Δ ≥ +0.010 (worse, clears 2σ).
- Post-LN at its best LR (warmup 200): Δ within ±0.015. Its best LR is no higher than
  pre-LN's (2e-3 or 4e-3).
- Divergence demo:
  - Post-LN at warmup 0 fails at LR 8e-3: non-finite loss, or a final val loss ≥ 0.5 nats
    worse than post-LN at warmup 200 and the same LR. It probably fails at 4e-3 too.
  - Post-LN becomes trainable (within 0.05 of its warmup-200 run) at warmup ≥ 50 for 4e-3,
    and at warmup ≥ 200 for 8e-3.
  - Pre-LN at warmup 0 trains at both LRs, within 0.03 of its warmup-200 run.

Xiong et al. (2020) show post-LN needs warmup because of large gradients near the output,
and Liu et al. (2020) argue post-LN can match or beat pre-LN once stabilized. Both studied
deeper models than 6 layers, where the instability is stronger. So a null in the
demonstration is a plausible outcome here, and it would be reported as such.

## Meaningful effect size
2σ (0.0072) for the variants (shared rule 3). For the demonstration: a failure is a
non-finite loss or a ≥ 0.5-nat penalty; "trainable" is within 0.05 of the warmup-200 run.

## What would falsify it
- Post-LN at warmup 0 and LR 8e-3 trains to within 0.05 of warmup 200: no post-LN
  instability at this depth.
- No-final-norm within 2σ of the baseline: the final norm is unnecessary here.
- RMSNorm differs from LayerNorm by ≥ 2σ in either direction.

## Outcome *(filled in after)*
