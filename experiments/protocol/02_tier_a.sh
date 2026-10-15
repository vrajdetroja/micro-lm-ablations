# Tier A. Every line is resumable.
# Run only after 01_noise_floor.sh has fixed sigma. Preregistrations 002-006 must be committed first.
set -e
P="python experiments/launch/protocol.py --center auto"
$P --config configs/ablations/pos_sinusoidal.yaml configs/ablations/pos_rope.yaml \
            configs/ablations/pos_alibi.yaml configs/ablations/pos_nope.yaml
$P --config configs/ablations/ffn_swiglu.yaml configs/ablations/ffn_geglu.yaml configs/ablations/ffn_relu2.yaml
$P --config configs/ablations/norm_rmsnorm.yaml configs/ablations/norm_post.yaml configs/ablations/norm_no_final.yaml
$P --config configs/ablations/sched_cosine0.yaml configs/ablations/sched_linear.yaml configs/ablations/sched_wsd.yaml
$P --config configs/ablations/dw_L2_d640.yaml configs/ablations/dw_L12_d256.yaml configs/ablations/dw_L24_d192.yaml
python analysis/collect.py && python analysis/compare.py
