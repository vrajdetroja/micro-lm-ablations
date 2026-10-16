# 009 — Scaling ladder (C1) and the cross-scale check (C2)

*Committed 2026-10-16, before any ladder run. Shared rules: 002, with 007's change to the
pooled σ.*

## Setup

A width ladder at fixed depth (6 layers, d_head 64). Every point trains for 1,620 steps at
~20 tokens per non-embedding parameter, so tokens per step grow with N (`configs/generate.py`):

| Width | N_non_embedding | Train tokens | Tokens/step | FLOPs |
|---|---|---|---|---|
| 128 | 1.18M | 27M | 16,384 | 4.8e14 |
| 192 | 2.66M | 53M | 32,768 | 1.7e15 |
| 256 | 4.72M | 93M | 57,344 | 4.7e15 |
| 384 (= baseline) | 10.62M | 212M | 131,072 | 2.05e16 |
| 512 | 18.88M | 378M | 233,472 | 5.95e16 |

That spans 16× in N and 124× in compute. Two families run at every width, otherwise identical:
- **Learned positions** (`ladder_d*`; the baseline at 384).
- **RoPE** (`ladder_d*_rope`; `pos_rope` at 384).

**Seeds × LRs:** the shared protocol centered on 4e-3, but with up to **two** edge extensions.
The best LR is expected to move with width. 5–7 runs per point and family. The d=384 points
reuse the noise-floor and Tier A runs.
**Metric:** val_iid loss in nats (same tokenizer everywhere).

## C1: the scaling law

**Fit:** L(N) = A · N^(−α) + E over the five learned-position points, using each width's
3-seed mean at its best LR and N = N_non_embedding.
- **Method:** profile least squares over E on a grid below the smallest observed loss. For
  each E, α and A come from linear regression of log(L − E) on log N.
- **Uncertainty:** 95% CIs for (α, E) from 2,000 bootstrap resamples, each drawing one seed
  per width.
- The same fit is applied to the RoPE family.

**Predictions:**
- α between 0.25 and 0.50. N and D grow together here, so the fitted α blends Chinchilla's
  N and D exponents (0.34, 0.28; Hoffmann et al., 2022). It is not comparable to Kaplan et
  al.'s data-unlimited α_N ≈ 0.076.
- The CI on α is wide (half-width ≥ 0.10). Five points over 16× cannot pin α and E apart,
  and saying so is part of the result.
- The best LR falls with width: ≥ 8e-3 at d = 128, ≤ 4e-3 at d = 512.

**Falsifier:** the five points do not fit a power law plus constant (any residual > 3× the
seed spread), or α falls outside [0.15, 0.7].

## C2: does RoPE's advantage survive scale?

Tier A's clearest effect: RoPE −0.084 nats against learned positions at d = 384, about 12×
2σ_seed. Here gap(N) = RoPE mean − learned mean at each width, both at their own best LR, 3 seeds.

**Predictions:**
- RoPE is better at all five widths, each by ≥ 2σ under both noise estimates.
- The gap shrinks with scale: |gap| at d = 512 between 0.03 and 0.08, and |gap| at d = 128
  between 0.06 and 0.15.

The reasoning: a learned position table costs a small model more relative capacity, and the
relative-position prior matters most when data is scarcest. The study design frames this
experiment directly: an inverting or closing gap would be the headline result.

**Falsifiers:**
- The sign flips at any width: learned beats RoPE by ≥ 2σ somewhere.
- The gap grows monotonically with N.
- The gap is constant within 2σ across the ladder: no scale dependence.

## Outcome *(filled in after)*
