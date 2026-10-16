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
# 002 amendment: the fixed table rescaled to the learned table's starting magnitude
$P --config $A/pos_sinusoidal_scaled.yaml
# 003 FFN activation
$P --config $A/ffn_swiglu.yaml $A/ffn_geglu.yaml $A/ffn_relu2.yaml
# 004 normalization
$P --config $A/norm_post.yaml $A/norm_no_final.yaml $A/norm_rmsnorm.yaml
# 004 amendment: norm placement x initialization. GPT-2-init post-LN collapsed at every LR;
# post-LN is swept lower (center 1e-3) under PyTorch's standard init, with its pre-LN control.
$P --config $A/init_torch.yaml
python experiments/launch/protocol.py --center 0.001 --config $A/norm_post_torchinit.yaml
# 004 divergence demonstration, revised to post-LN's trainable range (runs/warmup_demo).
# The warmup-200 references are the protocol's own seed-0 runs.
for lr in 1e-3 2e-3; do
    for w in 0 50 800; do
        python -m microlm.train.trainer --config $A/norm_post_torchinit.yaml \
            --set train.lr=$lr train.warmup_steps=$w train.seed=0 train.out_dir=runs/warmup_demo \
            || echo "RUN FAILED: post-LN torch-init lr=$lr warmup=$w"
    done
    python -m microlm.train.trainer --config $A/init_torch.yaml \
        --set train.lr=$lr train.warmup_steps=0 train.seed=0 train.out_dir=runs/warmup_demo \
        || echo "RUN FAILED: pre-LN torch-init lr=$lr warmup=0"
done
# 005 learning-rate schedule
$P --config $A/sched_linear.yaml $A/sched_cosine0.yaml $A/sched_wsd.yaml
# 006 depth vs width; checkpoints kept for the induction-head timing analysis (C3)
$P --config $A/dw_L2_d640.yaml $A/dw_L12_d256.yaml $A/dw_L24_d192.yaml \
    --set "train.keep_ckpts=[25,50,100,200,400,800,1620]"

python analysis/collect.py && python analysis/compare.py
