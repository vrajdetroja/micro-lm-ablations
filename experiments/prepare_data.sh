#!/usr/bin/env bash
# Freeze every dataset the study uses. Runs on CPU. Idempotent at the
# level of each output directory: frozen directories are never overwritten.
#
#   bash experiments/prepare_data.sh
set -euo pipefail
P="python -m microlm.data.prepare"

$P download --source fineweb_edu          # first 2 of 14 sample-10BT shards, ~4.3 GB
$P download --source codeparrot_valid     # out-of-distribution eval text (Python)
$P tok-sample --source fineweb_edu --bytes 5e8
python -m microlm.tokenizer.train --input data/fineweb_edu/tok_train_sample.jsonl \
    --vocab-sizes 257 2048 4096 8192 16384 32768 --out tokenizers

# Train-token limits: bpe_8192 serves the unit runs and the ladder (largest needs 378M);
# the other vocabs serve only the equal-FLOPs vocab ablation (largest needs 229M).
freeze() {
    [ -d "data/fineweb_edu/$1" ] && { echo "data/fineweb_edu/$1 exists, skipping"; return; }
    $P tokenize --source fineweb_edu --tokenizer "tokenizers/$1.json" --ood-source codeparrot_valid \
        --max-train-tokens "$2"
}
freeze bpe_8192 4.5e8
freeze bpe_2048 2.6e8
freeze bpe_4096 2.6e8
freeze bpe_16384 2.6e8
for d in data/fineweb_edu/bpe_*; do $P verify --data-dir "$d"; done
