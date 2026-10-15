# Validation: the CUDA path, real throughput, and resume determinism on the GPU.
set -e
python -m pytest -q
# Throughput and MFU for the unit model, the slowest planned shapes, and micro-batch choices.
python experiments/launch/bench_throughput.py configs/baseline.yaml --micro-batch 32 64 128 \
    --compile-mode default max-autotune-no-cudagraphs
python experiments/launch/bench_throughput.py configs/ablations/dw_L24_d192.yaml configs/ablations/dw_L2_d640.yaml \
    configs/ablations/vocab_16384.yaml configs/ladder/ladder_d512.yaml configs/ladder/ladder_d128.yaml \
    configs/ablations/opt_muon.yaml --micro-batch 32 64
# Resume determinism on the GPU: 200 steps straight vs 100 + resume 100.
SHORT="train.total_tokens=26214400 train.warmup_steps=50 train.extrapolation_lengths=[512]"
python -m microlm.train.trainer --config configs/baseline.yaml --set $SHORT train.out_dir=runs/validate/a
python -m microlm.train.trainer --config configs/baseline.yaml --set $SHORT train.out_dir=runs/validate/b train.stop_at_step=100
python -m microlm.train.trainer --config configs/baseline.yaml --set $SHORT train.out_dir=runs/validate/b
python -m microlm.train.trainer --config configs/baseline.yaml --set $SHORT train.out_dir=runs/validate/c
python analysis/check_resume.py runs/validate/a runs/validate/b runs/validate/c --resume-step 100
# Every planned job, listed with its training FLOPs.
python experiments/launch/sweep.py --config configs/baseline.yaml configs/ablations/*.yaml configs/ladder/*.yaml --dry-run | head -1
