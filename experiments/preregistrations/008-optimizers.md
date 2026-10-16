# 008 — Optimizer: AdamW vs Muon vs Lion

*Committed 2026-10-16, before any Tier B run. Shared rules: 002, with 007's change to the
pooled σ.*

**Configs** (same model, data, schedule, and number of training tokens as the baseline):
- `configs/ablations/opt_muon.yaml`: Muon on the hidden 2-D matrices (Nesterov momentum 0.95,
  5 Newton–Schulz steps, each fused matrix orthogonalized block by block), with AdamW at the
  baseline's 4e-3 for embeddings and vectors.
- `opt_lion.yaml`: Lion (β 0.9/0.99) everywhere, weight decay 1.0.

**Seeds × LRs** (each optimizer is swept in its own units, centered on its own default):
- Muon: `muon_lr` ∈ {0.01, 0.02, 0.04}.
- Lion: LR ∈ {2e-4, 4e-4, 8e-4}.

Both get the usual edge extension, then 2 seeds at the best: 5 runs each.
**Metric:** val_iid loss in nats (shared rule 2).
- **Equal steps** is the verdict.
- **Equal wallclock** is reported from the periodic val curves: Muon measures 833k tokens/s
  against AdamW's ~935k on the 24 GB card, about 11% slower per step.

## Hypothesis
Muon is more sample-efficient than AdamW at this scale, and most of that survives once its
higher per-step cost is paid. Lion is no better than a well-tuned AdamW at 10M.

## Prediction
- Muon: Δ −0.015 to −0.045 at equal steps (better). At equal wallclock it still wins, by
  at least half of its equal-steps gain.
- Lion: Δ within ±0.02 of AdamW (inconclusive or small), with a sharper LR optimum (the
  worse of its two neighbouring LRs ≥ 0.05 worse than its best).
- Ranking: Muon < AdamW ≈ Lion.

Jordan et al. (2024) report Muon's gains in the modded-nanoGPT speedrun at 124M. Chen et al.
(2023) report Lion's gains mostly at larger scale.

## Meaningful effect size
2σ under both noise estimates (shared rule 3 as amended in 007).

## What would falsify it
- Muon within 2σ of AdamW, or worse, at equal steps.
- Muon's gain vanishing (|Δ| < 2σ) at equal wallclock.
- Lion beating Muon.

## Outcome *(filled in after)*
