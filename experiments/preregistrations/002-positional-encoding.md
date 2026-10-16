# 002 — Positional encoding

*Committed 2026-10-16, before any run of Tier A. Do not edit after the first result exists;
add a dated "Amendments" section instead.*

## Shared rules for Tier A (002–006)

These apply to every Tier A preregistration; 003–006 refer back here.

1. **Protocol** (`experiments/launch/protocol.py --center auto`, run by `experiments/protocol/02_tier_a.sh`).
   Seed 0 at LR {2e-3, 4e-3, 8e-3}, centered on the baseline's best LR (4e-3, factor 2). If
   the best of those is at an edge, one more point beyond it. Then seeds 1 and 2 at the
   variant's own best LR. The baseline reference is 001's: 5 seeds at 4e-3, mean **3.5908**.
2. **Primary metric.** Final val_iid loss in nats/token over the full 10.2M-token split. All
   of Tier A uses the 8k tokenizer, so nats are comparable; BPB is reported alongside.
3. **Verdict.** Δ = (variant mean over its 3 seeds) − 3.5908. An effect is claimed only if
   |Δ| ≥ 2σ under **both** noise estimates:
   - σ_seed = 0.0036 from 001 (2σ = **0.0072**), and
   - σ_pooled: the within-variant standard deviation pooled over the baseline and every
     Tier A variant at its best LR, Σ(nᵢ−1)sᵢ² / Σ(nᵢ−1), about 36 degrees of freedom
     against 001's 4.

   Clearing exactly one of the two thresholds is reported as **threshold-sensitive**.
   Clearing neither is **inconclusive**. `analysis/compare.py` applies this mechanically.
4. **Matching.** Every variant trains on the identical 212M-token stream in the identical
   order, with the same batch (131,072 tokens/step) and step count (1,620). Non-embedding
   parameters are matched exactly, except depth vs width (006), which says how close it gets.
5. **Instability.** A run whose loss goes non-finite is "diverged" and ranks last in its
   LR sweep. A finished run more than 0.5 nats worse than the baseline is reported as
   unstable, never silently dropped.

## This ablation

**Configs:** `configs/ablations/pos_{sinusoidal,rope,alibi,nope}.yaml` against the baseline's
learned absolute positions. 10.62M non-embedding params in all five. The learned table's
0.2M parameters are embedding parameters, so N_total is 13.77M against 13.96M.
**Seeds × LRs:** shared protocol, 5 runs per variant (+1 if an LR edge extension is needed).
**Metrics:**
- In-distribution: val_iid loss at the 512 training length (shared rule 2).
- Length extrapolation: on val_long (200 documents, the first L tokens of each), mean loss
  at L ∈ {512, 1024, 2048, 4096}, plus loss by position bucket (`summary.json` →
  `extrapolation`). Learned positions are undefined beyond 512 and are reported as n/a.
  The baseline's val_long loss at 512 is 3.5426.

## Hypothesis
In distribution, the positional method matters little: RoPE and ALiBi are slightly better
than learned positions, and NoPE slightly worse. Beyond the training length the methods
separate sharply: ALiBi extrapolates, while RoPE and sinusoidal break down.

## Prediction
In-distribution, Δ against learned (3.5908):
- RoPE: −0.010 to −0.025 (clears 2σ).
- ALiBi: −0.005 to −0.020.
- Sinusoidal: within ±0.008 (inconclusive).
- NoPE: +0.005 to +0.030 (worse).

Ranking: RoPE ≤ ALiBi < learned ≈ sinusoidal < NoPE.

Extrapolation, as each method's loss at L minus its own loss at 512:
- ALiBi: ≤ +0.02 at 4096 (graceful; its distance penalty makes it effectively local).
- RoPE: ≥ +0.3 at 2048.
- Sinusoidal: ≥ +0.3 at 2048.
- NoPE: smaller rise than RoPE at 1024, but ≥ +0.3 at 4096.

Ranking at 4096: ALiBi < NoPE < {RoPE, sinusoidal}. In the position-bucket curves, ALiBi's
loss stays flat or keeps falling past position 512, and RoPE's rises within a few hundred
positions after it.

This is the natural-language-perplexity analogue of Kazemnejad et al. (2023), who found
NoPE extrapolates surprisingly well on reasoning and algorithmic tasks with small decoders.
Whether that carries over to LM loss is the open question here. Press et al. (2021) report
ALiBi extrapolating on perplexity.

## Meaningful effect size
In-distribution: 2σ (0.0072), under shared rule 3. Extrapolation: differences of ≥ 0.1 nats
at 2048 or 4096. That is more than ten times the in-distribution seed spread; the spread
across each method's 3 seeds is reported at every L.

## What would falsify it
- ALiBi's loss at 4096 exceeds its loss at 512 by more than 0.1: ALiBi does not extrapolate here.
- NoPE beats ALiBi at 4096: falsifies the extrapolation ranking and sides with Kazemnejad et al.
- RoPE fails to beat learned positions by 2σ in distribution: no in-distribution advantage at 10M.

## Amendment — 2026-10-16, after the first sinusoidal runs, before any run of the new variant

**What happened.** `pos_sinusoidal` as configured adds the raw sin/cos table, RMS 1/√2 ≈ 0.71,
to token embeddings initialized at std 0.02. Its seed-0 sweep reached 6.00 (2e-3), 5.46
(4e-3), 5.40 (8e-3), and 6.23 at the edge extension (1.6e-2). Seeds 1–2 at 8e-3 gave 5.62
and 5.06. The table is ~35× larger than the token embeddings and drowns out token identity.
That is a real result for this configuration, and it stays in the report. But it does not
test fixed sinusoidal positions in the sense this preregistration intended, and the
in-distribution prediction above (within ±0.008 of learned) does not apply to it.

**Diagnosis** (6 × 128 model on TinyStories, 200 steps). Raw table: 6.08 nats at LR
1e-3 and 4e-3, near unigram level. Table rescaled to RMS 0.02: 4.40 at 1e-3 and 4.24 at
4e-3, against 4.41 and 4.13 for learned positions.

**New variant:** `pos_sinusoidal_scaled`. Same fixed table, rescaled to RMS 0.02, the
learned table's starting magnitude, so the only difference from the baseline is "fixed
pattern vs learned table". The scale was chosen by that principle, not by the tiny-model
numbers. Shared protocol, 5 runs.

**Predictions for `pos_sinusoidal_scaled`.** In distribution, Δ against learned within
±0.015, most likely inconclusive. Extrapolation: loss at 2048 ≥ its own 512 loss + 0.3,
as predicted above for sinusoidal positions.

## Outcome *(filled in after)*
