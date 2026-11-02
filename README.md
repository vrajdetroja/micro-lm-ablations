# micro-lm-ablations

**What actually matters at 10M parameters?** A preregistered, controlled ablation study of
small transformer language models. It compares 24 architecture, initialization, schedule,
tokenizer, and optimizer variants against a vanilla GPT-2-style baseline. Every comparison is
matched on parameters and tokens, each variant gets its own learning-rate sweep, and every
result is judged against a measured noise floor.

The full write-up is **[REPORT.md](REPORT.md)**.

## Findings

10.6M non-embedding parameters, 212M FineWeb-Edu tokens, 3 seeds per variant, each at its own
best LR. Δ is in nats/token against the baseline's 3.5908.

| | Result |
|---|---|
| Noise floor | σ_seed = 0.0036 over 5 seeds; **78% of it is GPU kernel non-determinism** (same-seed repeats) |
| Better | Muon −0.115 · RoPE −0.084 · ALiBi −0.059 · WSD schedule −0.036 |
| Worse | NoPE +0.129 · fixed sinusoid +0.047 · no final norm +0.038 · standard init +0.054 · any depth other than 6 layers (+0.06 to +0.17) · Lion +0.089 |
| Small, threshold-sensitive | SwiGLU / GEGLU / ReLU² −0.013 to −0.022 |
| Null | RMSNorm vs. LayerNorm · linear vs. cosine decay |
| Post-LN | collapses to the unigram solution under GPT-2 initialization at every LR; trains only with standard init **and** long warmup, and is still +0.21 behind pre-LN |
| Vocabulary (bits per byte, equal FLOPs) | 2k 1.370 · 4k 1.346 · 8k 1.322 · **16k 1.305**; per-token loss ranks them exactly backwards |
| Length extrapolation | only ALiBi (its loss at 4,096 is no higher than at the 512 training length); NoPE degrades most |
| Out of distribution (code) | RoPE and ALiBi are *worse* than learned positions, despite winning in distribution |
| Depth | deeper models copy in context better (induction probe) but model language worse |
| Scaling law | L(N) = 151 · N^−0.299 + 2.41, α 95% CI [0.26, 0.35], over 1.2M–18.9M parameters |
| **RoPE across scale** | better at every width, but the gap shrinks steadily: −0.337 at 1.2M → −0.074 at 18.9M |
| Induction heads | form between steps 400 and 800 at every depth; depth changes their strength, not their timing |
| Predictions | 32 of 59 preregistered numeric predictions held; effect sizes at 10M were consistently underestimated |

## Method

- **Noise floor first.** Five baseline seeds give σ_seed, and same-seed repeats isolate GPU
  kernel noise. An effect is claimed only if it clears **2σ under two noise estimates**:
  - σ_seed (2σ = 0.0072);
  - σ_pooled, the within-variant spread pooled over all variants (2σ = 0.0236).

  Clearing one but not the other is *threshold-sensitive*; clearing neither is *inconclusive*.
- **Optimum to optimum.** Every variant is swept at three LRs centered on the baseline's best,
  extended past any edge, then run at two more seeds at its own best LR.
- **Preregistration.** Each hypothesis, prediction, meaningful effect size, and falsifier was
  committed before its runs. Amendments are dated and were committed before the runs they
  affect, and every deviation is reported as one.
- **Identical data.** Every run reads the same frozen token stream in the same order, with
  SHA-256 verified at start. A run's ID hashes every config field that can change the
  trained model, so "one change at a time" is mechanically checkable
  (`python -m microlm.train.config A.yaml B.yaml`).
- **Three matchings.** Results are reported at equal steps (equal tokens), equal FLOPs, and
  equal wallclock. Vocabulary sizes are compared at equal FLOPs, in bits per byte.

## Conventions

| Quantity | Definition |
|---|---|
| `N_non_embedding` | All weights except token/position embeddings and the LM head (Kaplan et al.'s N) |
| `N_total` | Everything, excluding vocab padding rows |
| FLOPs/token | `6·N_non_emb + 6·d·V` (unembedding) `+ 12·L·T·d` (attention), PaLM appendix-B convention. The textbook 6N undercounts by 34% at d = 384 |
| Bits per byte | Total nats over evaluated targets ÷ total UTF-8 bytes of those targets ÷ ln 2, exact per token |
| Parameter-matched FFN | Gated FFNs get hidden size (2/3)·4d, so every FFN has 8d² weights |
| Precision | bf16 autocast, fp32 master weights |

## Repository

```
src/microlm/
  tokenizer/   byte-level BPE: incremental trainer (exactly matches a naive reference), encoder/decoder
  data/        content-hash splits, uint16 memmaps, SHA-256 manifests, deterministic window order
  model/       config-switched transformer: positions (learned/sinusoidal/RoPE/ALiBi/NoPE),
               attention (MHA/GQA/MQA, QK-norm), FFN (GELU/SwiGLU/GEGLU/ReLU²), norms
               (LayerNorm/RMSNorm, pre/post), init schemes, parameter and FLOP accounting
  train/       training loop, AdamW/Lion/Muon, schedules (cosine/linear/WSD), exact-resume
               checkpoints, run configs and hashing
  eval/        loss and bits per byte, loss by position, length extrapolation, induction probe
configs/       baseline.yaml; generate.py writes ablations/, ladder/, reserve/ (designed, not run)
experiments/   preregistrations/, protocol/ (one script per stage), launch/
               (LR-sweep protocol, sweep runner, throughput benchmark), prepare_data.sh
analysis/      collect.py → results/all_runs.csv; compare.py (verdicts); tier_a_secondary.py;
               scaling.py; induction_timing.py; check_resume.py; figures.py
results/       all_runs.csv, scaling.json, figures/
tests/         113 tests
```

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -e ".[data,dev]" && .venv/bin/pytest
bash experiments/prepare_data.sh                         # download, tokenizers, frozen data
bash experiments/protocol/00_validation.sh               # throughput, resume determinism
bash experiments/protocol/01_noise_floor.sh              # preregistration 001
bash experiments/protocol/02_tier_a.sh                   # 002–006
bash experiments/protocol/03_tier_b_and_scaling.sh       # 007–009
python analysis/collect.py && python analysis/compare.py --exclude-unstable
python analysis/tier_a_secondary.py && python analysis/scaling.py
python analysis/induction_timing.py && python analysis/figures.py
```

The runs reported here used one NVIDIA RTX 4090 (PyTorch 2.10, `torch.compile`), training the
unit model at 55% MFU. Every training command is idempotent: finished runs are skipped, and
interrupted runs resume from their last checkpoint. Data hashes for every split are listed in
REPORT.md §9.

## What the tests pin down

- `decode(encode(x)) == x` for random text, including emoji, CJK, and lone surrogates, and
  for arbitrary invalid-UTF-8 bytes.
- The incremental BPE trainer produces exactly the naive trainer's merges.
- Analytic parameter counts equal the instantiated module's for every variant, and gated
  FFNs are within 0.5% of the GELU baseline.
- The fused (SDPA) attention path equals a naive softmax reference for every positional
  variant, and every variant is causal.
- A resumed run is bitwise identical to an uninterrupted one on CPU, for AdamW, Lion, and Muon.
- The IDs of finished runs survive code changes. Flipping one bit of a frozen split makes
  every run refuse to start.

## References

The methodological template is Narang et al. (2021), *Do Transformer Modifications Transfer
Across Implementations and Applications?* Full references are in REPORT.md.

## Citation

```bibtex
@misc{detroja2026microlm,
  author       = {Detroja, Vraj},
  title        = {What Actually Matters at 10M Parameters? A Preregistered Ablation
                  Study of Small Transformer Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/vrajdetroja/micro-lm-ablations}}
}
```

## License

MIT. See [LICENSE](LICENSE).
