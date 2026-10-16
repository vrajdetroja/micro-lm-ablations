# 007 — Tokenizer vocabulary size

*Committed 2026-10-16, after Tier A and before any Tier B run. Shared rules: 002, with one
change, adopted here before these runs: σ_pooled excludes rule-5-unstable variants. In Tier
A that exclusion was applied post hoc, because one unstable variant dominated the pool.*

**Configs** (all trained to the baseline's training FLOPs, 2.05e16, within 1%; total params
matched by depth within 4%):

| Config | Vocab | Layers | N_non_embedding | N_total | Train tokens | Train bytes |
|---|---|---|---|---|---|---|
| `vocab_2048` | 2,048 | 7 | 12.39M | 13.38M | 215M | 644 MB |
| `vocab_4096` | 4,096 | 7 | 12.39M | 14.16M | 205M | 711 MB |
| baseline | 8,192 | 6 | 10.62M | 13.96M | 212M | 835 MB |
| `vocab_16384` | 16,384 | 4 | 7.08M | 13.57M | 229M | 993 MB |

**The trade at fixed total parameters and FLOPs.** A bigger vocabulary spends parameters on
embeddings, so the body gets shallower, but each token covers more bytes, so the model sees
more text. A smaller vocabulary buys depth but sees less text.
**Seeds × LRs:** shared protocol (centered on 4e-3), 5 runs per variant.
**Metric: val_iid bits per byte.** The validation text is byte-identical for every tokenizer
(40,003,039 bytes). Per-token loss is not comparable across vocabularies, but it is reported
alongside to show why. Baseline: **1.3218 BPB** (5 seeds); σ_seed = 0.0013 (2σ = 0.0027);
σ_pooled over Tier A's stable variants = 0.0046 (2σ = 0.0091).

## Hypothesis
At this scale the best vocabulary is small, nowhere near the 32k–100k common in practice.
Large vocabularies cost too much body depth for the extra text they let the model see.
Very small ones see too little text.

## Prediction
Δ BPB against the 8k baseline:
- 2,048: +0.010 to +0.030 (worse: 23% fewer bytes outweighs one extra layer).
- 4,096: within ±0.006.
- 16,384: +0.005 to +0.025 (worse: 4 layers, after Tier A showed depth below 6 costs).

The optimum is 4k or 8k. In nats per token the ordering differs from BPB (bigger vocab, higher
per-token loss), and it would mislead.

## Meaningful effect size
2σ in BPB under both noise estimates (shared rule 3 as amended above).

## What would falsify it
- 16,384 better than 8,192 by ≥ 2σ: more text beats depth even at 4 layers.
- 2,048 within 2σ of 8,192: compression barely matters at this scale.
- The ranking by BPB matches the ranking by nats per token: the measurement subtlety does
  not matter here.

## Outcome *(filled in 2026-10-17)*

All best at LR 4e-3 (interior). Noise floor in BPB: 2σ_seed = 0.0027, and 2σ_pooled = 0.0083
(48 df, unstable variants excluded as preregistered). Every Δ clears both.

| Vocab | Layers | BPB | Δ BPB | Verdict | Predicted Δ | Nats/token |
|---|---|---|---|---|---|---|
| 2,048 | 7 | 1.3703 | **+0.0485** | worse | +0.010 to +0.030 (**missed**, larger) | 2.838 |
| 4,096 | 7 | 1.3460 | **+0.0242** | worse | ±0.006 (**missed**) | 3.230 |
| 8,192 (baseline) | 6 | 1.3218 | — | — | — | 3.591 |
| **16,384** | 4 | **1.3053** | **−0.0165** | **better** | +0.005 to +0.025 (**falsified**) | 3.905 |

- **Hypothesis falsified.** At equal training FLOPs and matched total parameters, BPB
  improves steadily with vocabulary size up to the largest tested (16k), even though the 16k
  model has only 4 layers. Seeing more text (993 MB against 835 MB) beat losing two layers.
  The optimum is at 16k or above; 32k was not tested.
- **Falsifiers:** "16k better than 8k by ≥ 2σ" was **triggered**. "2k within 2σ of 8k" was
  not triggered.
- **Measured per token, the ranking is exactly reversed** (2k looks best, 16k worst). The
  prediction that the two orderings differ **held**; the falsifier "BPB ranking matches the
  nats ranking" was not triggered. Comparing tokenizers per token would have produced the
  opposite conclusion.
- **Predictions held: 1 of 5.**

