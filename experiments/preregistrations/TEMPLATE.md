# NNN — <ablation name>

*Committed before any run of this ablation. Do not edit after the first result exists;
add an "Amendments" section with a date instead.*

**Configs:** `configs/ablations/<...>.yaml` (vs `configs/baseline.yaml`)
**Seeds × LRs:** <e.g. 3 seeds × 5 LRs per variant>
**Metric:** <primary metric, e.g. final val_iid loss (nats/token); BPB if the tokenizer varies>
**Matching:** <what is held equal: tokens, non-embedding params, total params, FLOPs>

## Hypothesis
<one or two sentences, directional>

## Prediction
<the number you expect, with sign: e.g. "RoPE beats learned by 0.01–0.03 nats at 1024">

## Meaningful effect size
<the smallest effect you would act on; must be stated relative to 2σ>

## What would falsify it
<the outcome that would make you say you were wrong>

## Outcome *(filled in after)*
<result with mean ± range, verdict vs 2σ, and whether the prediction held>
