# Noise floor: baseline LR sweep, then seeds at the best LR.
# Preregistration: experiments/preregistrations/001-seed-variance.md
set -e
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.lr=1e-3,2e-3,4e-3,8e-3,1.6e-2 \
    --set train.keep_ckpts=[25,50,100,200,400,800,1620]
python analysis/collect.py
BEST=$(python analysis/compare.py --best-lr baseline)
echo "best baseline lr: $BEST"
python experiments/launch/sweep.py --config configs/baseline.yaml --grid train.seed=1,2,3,4 --set train.lr=$BEST
python analysis/collect.py && python analysis/compare.py
