# 003 — FFN activation

*Committed 2026-10-16, before any run of Tier A. Shared rules (protocol, metric, verdict,
matching, instability): see 002.*

**Configs:** `configs/ablations/ffn_{swiglu,geglu,relu2}.yaml` against the baseline's GELU
FFN. Every FFN has 2 · 4 · d² = 1.18M weights per layer. The gated variants get hidden size
1024 = (2/3) · 4 · 384 with three matrices; GELU and ReLU² get 1536 with two. Non-embedding
params 10.62M in all four (within 0.1%, asserted in `tests/test_model.py`); FLOPs equal.
**Seeds × LRs:** shared protocol, 5 runs per variant.
**Metric:** val_iid loss (shared rule 2). Measured tokens/s is reported, since three smaller
matmuls do not run at the speed of two larger ones.

## Hypothesis
Gated FFNs beat GELU at matched parameters even at 10M. Squared ReLU is no better than GELU
at this scale.

## Prediction
Δ against GELU (3.5908):
- SwiGLU: −0.008 to −0.025 (clears 2σ).
- GEGLU: within 2σ of SwiGLU.
- ReLU²: within ±0.008 (inconclusive).

Throughput of the gated variants within 5% of GELU.

Shazeer (2020) reports GLU-variant gains at T5 scale, and So et al. (2021, Primer) found
squared ReLU helpful at larger scale. This tests whether either survives at 10M parameters with
exact parameter matching.

## Meaningful effect size
2σ (0.0072), under shared rule 3.

## What would falsify it
- SwiGLU within 2σ of GELU, or worse: the gating advantage does not appear at 10M. That is
  a null result worth reporting, given how routinely SwiGLU is adopted.
- ReLU² better than GELU by ≥ 2σ: squared ReLU helps even at this scale.

## Outcome *(filled in 2026-10-16)*

All best at LR 4e-3. Verdicts under the rule as written / the post-hoc pool (see 002's Outcome).

| Variant | Mean | Δ | Verdict (written / post hoc) | Predicted Δ |
|---|---|---|---|---|
| SwiGLU | 3.5768 | −0.014 | threshold-sensitive / threshold-sensitive | −0.008 to −0.025 |
| GEGLU | 3.5689 | −0.022 | threshold-sensitive / threshold-sensitive | within 2σ of SwiGLU |
| ReLU² | 3.5774 | −0.013 | threshold-sensitive / threshold-sensitive | ±0.008 |

- **SwiGLU's range: held.** Its gain clears 2σ_seed (0.0072) but not 2σ_pooled (0.025), so
  it is threshold-sensitive under both pools.
- **GEGLU vs SwiGLU:** they differ by 0.008, just over 2σ_seed and well inside 2σ_pooled.
  **Held under the pooled σ, borderline under σ_seed.**
- **ReLU² within ±0.008: missed.** It is as good as the gated variants.
- **Throughput within 5%: held, barely.** SwiGLU 922k, GEGLU 921k, ReLU² 938k tokens/s,
  against the baseline's ~965k on the same card (921k measured on the 24 GB card, ×1.047).
- **Falsifier "SwiGLU within 2σ of GELU":** not triggered under σ_seed, triggered under the
  pooled σ. The honest summary: at 10M, every alternative FFN beats GELU by 0.013–0.022 nats,
  an effect real against seed noise but small against variant-to-variant noise.

