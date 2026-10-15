# micro-lm-ablations

**What actually matters at 25M parameters?** A controlled study of architecture and
optimization choices in small transformer language models, built from scratch, with a
measured seed-variance noise floor and every comparison matched on parameters and tokens.

The subject is a language model; the deliverable is the experimental method. Every result
is reported against a noise floor σ (the standard deviation over baseline seeds), and any
effect smaller than 2σ is reported as **inconclusive**, not as a finding.

> **Status:** infrastructure complete and tested (96 tests). Tokenizer, data pipeline,
> model, training loop, and evaluation all run end-to-end locally. No results yet: the
> next step is freezing the FineWeb-Edu data and measuring σ on a GPU.

---

## Methodology (the rules every result in this repo follows)

1. **Noise floor first.** The baseline is trained with 5 seeds before any ablation. σ is
   the std of final `val_iid` loss. Effects < 2σ are *inconclusive*.
2. **≥ 3 seeds** for every headline comparison; figures show mean ± range, never a single line.
3. **Optimum-to-optimum.** The peak LR is swept independently for every variant, including
   the baseline. Comparisons are best LR vs. best LR (`analysis/compare.py`).
4. **One change at a time, checked mechanically.** Every ablation is a YAML file that
   inherits `configs/baseline.yaml`. A run's ID is a hash of every semantic config field
   plus the data hash, and `python -m microlm.train.config A.yaml B.yaml` prints exactly
   which fields differ.
5. **Identical data.** Every run reads the same frozen token stream in the same order. The
   order depends only on `data_seed`, never on the model seed. The SHA-256 of every split is
   asserted at the start of each run.
6. **Preregistration.** Before each ablation a hypothesis, metric, meaningful effect size,
   and falsification criterion are committed to `experiments/preregistrations/`. The report
   lists predictions vs. outcomes.
7. **Three matchings.** Results are reported at equal steps (equal data), equal FLOPs, and
   equal wallclock.

## Conventions

| Quantity | Definition |
|---|---|
| `N_non_embedding` | All weights except token/position embeddings and the LM head (Kaplan et al.'s N) |
| `N_total` | Everything, excluding vocab padding rows. Every table reports both |
| FLOPs/token | `6·N_non_emb + 6·d·V` (unembedding) `+ 12·L·T·d` (attention), PaLM appendix-B convention. MFU uses this |
| BPB | Total nats over evaluated targets ÷ total UTF-8 bytes of those targets ÷ ln 2, computed exactly per token. Used whenever the tokenizer varies; per-token loss is not comparable across vocabularies |
| Param-matched FFN | Gated FFNs (SwiGLU/GEGLU) get hidden size `(2/3)·4d` so they have the same `8d²` weights as a 4d GELU FFN. Asserted in `tests/test_model.py` |
| Tokens per step | 524,288, constant across every run; micro-batch changes are absorbed by gradient accumulation |
| Precision | bf16 autocast, fp32 master weights, no `GradScaler` (it exists for fp16's exponent range) |

## Compute arithmetic

The unit run is 8 layers × d=512, 25.17M non-embedding params, 1,144 steps × 0.5M tokens = 600M tokens.

```
6·N·D                      = 6 × 25.17M × 600M            = 9.1e16 FLOPs   (textbook)
+ unembedding  6·d·V·D     = 6 × 512 × 8192 × 600M        = 1.5e16
+ attention    12·L·T·d·D  = 12 × 8 × 1024 × 512 × 600M   = 3.0e16
= total                                                     1.36e17 FLOPs per run
```

At this scale the textbook `6N` estimate undercounts by **50%**. The unembedding and
attention terms are not small when `d` is only 512. Expected wallclock per run:

| GPU | Peak bf16 | Assumed MFU | Minutes / run |
|---|---|---|---|
| RTX 4090 | 165 TFLOP/s | 28% | ~49 |
| A100 40GB | 312 TFLOP/s | 40% | ~18 |
| H100 SXM | 989 TFLOP/s | 35% | ~7 |

MFU is measured and logged every step. The table's MFU values are assumptions until measured.

## Repository

```
src/microlm/
  tokenizer/   byte-level BPE: incremental trainer + naive reference, encoder/decoder
  data/        download, content-hash split, tokenize to uint16 memmap, manifest, loader
  model/       config-switched components: attention (MHA/GQA/MQA, QK-norm), FFN
               (GELU/SwiGLU/GEGLU/ReLU²), norms (LN/RMS, pre/post), positions
               (learned/sinusoidal/RoPE/ALiBi/NoPE), param + FLOP accounting
  train/       loop, AdamW/Lion/Muon, schedules (cosine/linear/WSD/constant),
               exact-resume checkpoints, run config + hashing
  eval/        val loss & BPB, loss by position, length extrapolation, induction probe
configs/       baseline.yaml, smoke.yaml, ablations/*.yaml (one change each)
experiments/   preregistrations/, launch/sweep.py
analysis/      collect.py (runs → results/all_runs.csv), compare.py (2σ verdicts)
tests/         96 tests: tokenizer round-trip, fast==naive BPE, param counts,
               fast==naive attention, causality, bitwise-exact resume, schedules, data integrity
```

### Things the tests pin down

- `decode(encode(x)) == x` for random text with emoji, CJK, zero-width characters, and lone
  surrogates, and for arbitrary (invalid UTF-8) bytes.
- The incremental BPE trainer produces exactly the naive trainer's merges.
- Analytic parameter counts equal the instantiated module's for every variant, and gated
  FFNs are within 0.5% of the GELU baseline. An unmatched SwiGLU would fail the test.
- The SDPA (Flash) attention path equals a naive softmax reference for every positional variant.
- A run interrupted at step 6 and resumed is **bitwise identical** to an uninterrupted run,
  for AdamW, Lion, and Muon.
- Flipping one bit of a frozen `.bin` makes every run refuse to start.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[data,dev]"
.venv/bin/pytest
```

End-to-end smoke test on CPU (about 1 minute):

```bash
python -m microlm.data.prepare download --source tinystories_mini
python -m microlm.data.prepare download --source codeparrot_valid
python -m microlm.data.prepare tok-sample --source tinystories_mini --bytes 5e7
python -m microlm.tokenizer.train --input data/tinystories_mini/tok_train_sample.jsonl --vocab-sizes 257 1024 2048 4096
python -m microlm.data.prepare tokenize --source tinystories_mini --tokenizer tokenizers/bpe_2048.json --ood-source codeparrot_valid --val-ood-bytes 2e6
python -m microlm.train.trainer --config configs/smoke.yaml
```

Real data (FineWeb-Edu 10BT sample, first 4 of 14 shards ≈ 8.6 GB download):

```bash
python -m microlm.data.prepare download --source fineweb_edu
python -m microlm.data.prepare tok-sample --source fineweb_edu --bytes 5e8
python -m microlm.tokenizer.train --input data/fineweb_edu/tok_train_sample.jsonl
python -m microlm.data.prepare tokenize --source fineweb_edu --tokenizer tokenizers/bpe_8192.json --ood-source codeparrot_valid --max-train-tokens 2.2e9
```

Seed-variance study and baseline LR sweep (GPU):

```bash
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.seed=0,1,2,3,4
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.lr=1e-3,2e-3,6e-3,1e-2 --grid train.seed=0,1,2
python analysis/collect.py && python analysis/compare.py
```

Every command is idempotent. Finished runs are skipped and interrupted runs resume from
their last checkpoint, so after an interruption you rerun the same command.

## Data splits

Documents are assigned to splits by a hash of their **content** (`blake2b(text) mod 1000`).
This does not depend on file order or shard selection, and exact duplicates can never
straddle train and validation.

| Split | Rule | Size |
|---|---|---|
| `val_iid` | hash < 10, first 40 MB of text | ~10M tokens; defined in bytes so every tokenizer sees identical text |
| `val_long` | 10 ≤ hash < 110 and ≥ 48 KB | up to 500 docs, each ≥ 8192 tokens; for length extrapolation |
| `val_ood` | Python source (codeparrot-clean-valid) | 8 MB of text |
| `train` | everything else | ~2.2B tokens at 8k vocab (enough for the 100M-param scaling run) |

## Tokenizer

Byte-level BPE with the GPT-4 pre-tokenization regex, written from scratch. BPE's merge
sequence does not depend on the target vocab size, so one training run to 32,768 yields
every smaller vocabulary as a prefix (`Tokenizer.truncated`).

The trainer keeps incremental pair counts and a max-heap with lazy invalidation. It
reproduces the naive trainer's merges exactly, 40× faster on a 300-merge slice of 2 MB.
On 100 MB of code (463k unique chunks) it learns 8,000 merges in 67 s using 0.9 GB of RAM
(`python -m microlm.tokenizer.bench`).

## References

Narang et al. 2021 (*Do Transformer Modifications Transfer…*) is the methodological
template.
