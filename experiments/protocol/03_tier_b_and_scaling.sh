# Tier B and the scaling ladder.
# Preregistrations 007 (vocab), 008 (optimizers), 009 (ladder + cross-scale), committed
# before this runs. No `set -e`: one failed run must not cancel the rest.
# Distinctive comparisons first, d=512 last.

python analysis/collect.py > /dev/null
BEST=$(python analysis/compare.py --best-lr baseline) || { echo "ABORT: no baseline results; run 01_noise_floor.sh first"; exit 1; }
[ "$BEST" = "0.004" ] || { echo "ABORT: baseline best LR is $BEST, expected 0.004 (preregistration 001)"; exit 1; }
for v in 2048 4096 16384; do
    [ -f data/fineweb_edu/bpe_$v/MANIFEST.json ] || { echo "ABORT: data for vocab $v missing"; exit 1; }
done
echo "baseline best lr: $BEST; all vocab datasets present"

P="python experiments/launch/protocol.py"
A=configs/ablations
L=configs/ladder

# 007 vocabulary size (verdict in bits per byte)
$P --center auto --config $A/vocab_2048.yaml $A/vocab_4096.yaml $A/vocab_16384.yaml
# 009 ladder, the small widths: learned and RoPE (C1 + C2); best LR may move, so 2 extensions
for d in 128 192 256; do
    $P --center auto --max-extend 2 --config $L/ladder_d$d.yaml $L/ladder_d${d}_rope.yaml
done
# 008 optimizers, each swept in its own LR units around its own default
$P --center own --lr-key train.muon_lr --config $A/opt_muon.yaml
$P --center own --config $A/opt_lion.yaml
# 009 ladder, d=512 (about 11 min per run)
$P --center auto --max-extend 2 --config $L/ladder_d512.yaml $L/ladder_d512_rope.yaml

python analysis/collect.py && python analysis/compare.py --exclude-unstable \
    && python analysis/compare.py --exclude-unstable --metric val_iid_bpb
