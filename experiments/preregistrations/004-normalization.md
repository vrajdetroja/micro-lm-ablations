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

## Amendment — 2026-10-16, after the first post-LN runs, before any run of the new design

**What happened.** `norm_post` (post-LN, GPT-2 initialization: N(0, 0.02), residual
projections scaled by 1/√(2L)) ended at 7.3380 nats in every run: seed 0 at LR 1e-3, 2e-3,
4e-3 and 8e-3, and seeds 1–2 at 2e-3. The loss fell to ~7.35 by step 50 and stayed there.
Attention entropy stayed at ~5.25 nats in every layer, which is uniform attention over the
context, and the model learned token frequencies only. The prediction "post-LN within
±0.015 of pre-LN" is **falsified** for this configuration. Post-LN under GPT-2
initialization collapses at every LR tried.

**Diagnosis** (6 × 128 model on TinyStories, 200 steps):
- This repository's post-LN block gives **exactly** the same losses as PyTorch's
  `nn.TransformerEncoderLayer(norm_first=False)` under the same initialization (6.137 at
  LR 1e-3, 5.913 at 3e-4). The implementation is not at fault.
- Under PyTorch's default initialization (Xavier-uniform QKV, default Linear init
  elsewhere), the reference layer trains: 4.575 at LR 1e-3 with 100 warmup steps.
- This repository's post-LN with that initialization (`init_scheme: torch`) also trains:
  4.475 at LR 1e-3 with warmup 100, against pre-LN's 4.406.
- It still collapses with GPT-2 init at the same settings (6.083), and nearly collapses
  with standard init when warmup is short (20 steps: 5.94 at 1e-3, 6.08 at 4e-3).
- So post-LN trains here only with **both** standard initialization **and** a gentle LR or
  warmup.
- Standard init is not simply better: pre-LN under it is worse than under GPT-2 init on the
  tiny model (5.19 against 4.41 at 1e-3).

**New design: a 2 × 2 of norm placement {pre, post} × initialization {gpt2, torch}.**
Two cells already exist: the baseline (pre, gpt2) and `norm_post` (post, gpt2: collapsed).
- `init_torch` (pre, torch): the shared protocol.
- `norm_post_torchinit` (post, torch): seed 0 at LR {5e-4, 1e-3, 2e-3}, with the usual
  edge extension, then 2 seeds at its best. That is centered at 1e-3 instead of the shared
  4e-3, which is justified before running by post-LN's known LR sensitivity (Xiong et al.,
  2020) and by the diagnostics above.

**Predictions:**
- `init_torch` is worse than the baseline: Δ in +0.01 to +0.10 (clears 2σ).
- `norm_post_torchinit` trains (val < 4.0, no collapse), with best LR ≤ 1e-3. Δ against
  `init_torch` is within ±0.03. Δ against the baseline clears 2σ upward.
- **Interaction:** switching init changes post-LN by ≥ 3 nats (7.338 → < 4.0) but pre-LN by
  ≤ 0.10. Post-LN's viability depends on initialization; pre-LN's does not.

**Divergence demonstration, revised** to post-LN's trainable range (`runs/warmup_demo/`,
8 runs). Post-LN with torch init at warmup ∈ {0, 50, 800} × LR ∈ {1e-3, 2e-3}, seed 0.
Pre-LN with torch init at warmup 0 × the same LRs as the control. Warmup-200 references
come from the protocol runs where they exist.
- Prediction: post-LN at warmup 0 fails (collapse, or ≥ 0.5 nats worse than its warmup-200
  run) at both LRs.
- Post-LN is trainable at warmup ≥ 50 for 1e-3 and only at ≥ 200 for 2e-3.
- Pre-LN at warmup 0 trains, within 0.05 of its warmup-200 run where that exists.

The original demo (GPT-2-init post-LN at 4e-3 and 8e-3) was never run.

## Outcome *(filled in 2026-10-16)*

Verdicts under the rule as written / the post-hoc pool (see 002's Outcome).

| Variant | Best LR | Mean | Δ | Verdict | Prediction |
|---|---|---|---|---|---|
| RMSNorm | 4e-3 | 3.5888 | −0.002 | inconclusive / inconclusive | \|Δ\| < 2σ: **held** |
| No final norm | 2e-3 | 3.6284 | +0.038 | threshold-sensitive / **worse** | ≥ +0.010: **held** |
| Post-LN, GPT-2 init | 2e-3 | 7.3380 | +3.747 | worse / worse (unstable) | ±0.015: **falsified** (collapsed) |
| Pre-LN, standard init (amendment) | 8e-3 | 3.6444 | +0.054 | threshold-sensitive / **worse** | +0.01 to +0.10: **held** |
| Post-LN, standard init (amendment) | 1e-3 | 3.8563 | +0.265 | worse / **worse** | trains, best LR ≤ 1e-3: **held** |

- **Post-LN vs pre-LN at matched (standard) init:** +0.212. "Within ±0.03": **missed** by a
  wide margin. Even when post-LN trains, it is far behind.
- **Interaction:** switching to standard init moves post-LN by −3.48 nats (7.338 → 3.856) and
  pre-LN by +0.054. "≥ 3 nats vs ≤ 0.10": **held.** Post-LN's viability depends on
  initialization; pre-LN's does not.
- **RMSNorm throughput:** 943k tokens/s, about 2% *slower* than LayerNorm on the same card.
  The predicted 0–3% speedup: **missed.** This repository's RMSNorm is a plain PyTorch
  function computed in fp32, while LayerNorm uses a fused kernel.

**Divergence demonstration** (amended design; post-LN and pre-LN with standard init; seed 0;
val_iid nats; "collapsed" = stuck at the unigram loss, 7.338):

| | warmup 0 | 50 | 200 (protocol run) | 800 |
|---|---|---|---|---|
| Post-LN, LR 1e-3 | collapsed | collapsed | 3.857 | 3.915 |
| Post-LN, LR 2e-3 | collapsed | collapsed | collapsed | **3.711** |
| Pre-LN, LR 1e-3 | 4.304 | | | |
| Pre-LN, LR 2e-3 | 4.199 | | 3.771 | |

- **Post-LN fails at warmup 0 at both LRs: held.**
- **Trainable at warmup ≥ 50 for 1e-3: missed** (it needs 200).
- **Trainable at ≥ 200 for 2e-3: missed** (200 collapses; 800 trains).
- **Pre-LN at warmup 0 within 0.05 of warmup 200: missed.** It trains at every setting but
  is 0.43 nats worse without warmup. Pre-LN never collapses, but it still benefits from warmup.

With 800 warmup steps, post-LN at LR 2e-3 (3.711) beats its own best protocol run (3.856).
The shared protocol's fixed 200-step warmup understates post-LN.

