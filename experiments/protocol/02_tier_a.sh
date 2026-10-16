# Tier A.
# Preregistrations 002-006 (shared rules in 002) are committed before this runs.
#
# No `set -e`: one failed run must not cancel the rest.
# protocol.py logs failed runs and ranks them last; explicit runs below log failures too.
# Groups run in order of importance.

# Every LR sweep is centered on the baseline's best LR, which comes from the noise-floor
# runs. Without them, stop now rather than run blind.
python analysis/collect.py > /dev/null
BEST=$(python analysis/compare.py --best-lr baseline) || { echo "ABORT: no baseline results; run 01_noise_floor.sh first"; exit 1; }
[ "$BEST" = "0.004" ] || { echo "ABORT: baseline best LR is $BEST, expected 0.004 (preregistration 001)"; exit 1; }
echo "baseline best lr: $BEST"

P="python experiments/launch/protocol.py --center auto"
A=configs/ablations

# 002 positional encoding
$P --config $A/pos_rope.yaml $A/pos_alibi.yaml $A/pos_nope.yaml $A/pos_sinusoidal.yaml
# 003 FFN activation
$P --config $A/ffn_swiglu.yaml $A/ffn_geglu.yaml $A/ffn_relu2.yaml
# 004 normalization
$P --config $A/norm_post.yaml $A/norm_no_final.yaml $A/norm_rmsnorm.yaml
# 004 divergence demonstration: DEFERRED (2026-10-16). Post-LN collapsed to the unigram
# loss (7.338) at every LR the protocol tried (1e-3 to 8e-3), so the preregistered demo grid
# (4e-3, 8e-3) would add nothing. A corrected design goes in a dated amendment to 004 and a
# follow-up run.
# 005 learning-rate schedule
$P --config $A/sched_linear.yaml $A/sched_cosine0.yaml $A/sched_wsd.yaml
# 006 depth vs width; checkpoints kept for the induction-head timing analysis (C3)
$P --config $A/dw_L2_d640.yaml $A/dw_L12_d256.yaml $A/dw_L24_d192.yaml \
    --set "train.keep_ckpts=[25,50,100,200,400,800,1620]"

python analysis/collect.py && python analysis/compare.py
