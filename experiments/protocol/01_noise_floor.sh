# Noise floor. Preregistration 001, including its 2026-10-16 amendment.
set -e

# 1. Resume check on the GPU, judged against kernel noise. The noise-floor runs never
#    resume, so a failure here is flagged loudly but does not invalidate them.
SHORT="train.total_tokens=26214400 train.warmup_steps=50 train.extrapolation_lengths=[512]"
T="python -m microlm.train.trainer --config configs/baseline.yaml --set $SHORT"
$T train.out_dir=runs/resume_check/a
$T train.out_dir=runs/resume_check/b train.stop_at_step=100
$T train.out_dir=runs/resume_check/b
$T train.out_dir=runs/resume_check/c
python analysis/check_resume.py runs/resume_check/a runs/resume_check/b runs/resume_check/c \
    --resume-step 100 || echo "RESUME CHECK FAILED: fix before Tier A (noise-floor runs are unaffected)"

# 2. Baseline LR sweep at seed 0, keeping checkpoints for induction-head timing (C3).
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.lr=1e-3,2e-3,4e-3,8e-3,1.6e-2 \
    --set train.keep_ckpts=[25,50,100,200,400,800,1620]
python analysis/collect.py
BEST=$(python analysis/compare.py --best-lr baseline)
echo "best baseline lr: $BEST"

# 3. Seeds 1-4 at the best LR: sigma_seed (init seed + kernel noise).
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.seed=1,2,3,4 --set train.lr=$BEST

# 4. Two repeats of the best run, same seed: sigma_kernel (kernel noise alone). An identical
#    config has an identical run ID, so the repeats get their own name and directory.
for r in 1 2; do
    python -m microlm.train.trainer --config configs/baseline.yaml \
        --set name=baseline_repeat train.lr=$BEST train.seed=0 train.out_dir=runs/repeats/r$r
done

python analysis/collect.py && python analysis/compare.py
