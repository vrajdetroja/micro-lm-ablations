"""Apply the study's decision rules to all_runs.csv.

1. Noise floor: sigma = std of the metric over baseline seeds (best baseline LR).
2. Every variant is compared optimum-to-optimum: for each variant, the LR with the best
   seed-mean metric, against the baseline at *its* best LR.
3. A difference smaller than 2 * sigma is reported as INCONCLUSIVE, not as a result.

    python analysis/compare.py [--csv results/all_runs.csv] [--metric val_iid_loss]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def best_lr(rows: list[dict], metric: str) -> tuple[str, list[float]]:
    """Group a variant's runs by LR; return the LR with the lowest seed-mean, and its values."""
    by_lr: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(metric) not in (None, "") and r.get("diverged") != "True":
            lr_key = r["train.lr"] if r["train.optimizer"] != "muon" else f"{r['train.lr']}/{r['train.muon_lr']}"
            by_lr[lr_key].append(float(r[metric]))
    if not by_lr:
        return "-", []
    lr = min(by_lr, key=lambda k: statistics.mean(by_lr[k]))
    return lr, by_lr[lr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_runs.csv")
    ap.add_argument("--metric", default="val_iid_loss")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args()

    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in load(args.csv):
        by_name[r["name"]].append(r)
    if args.baseline not in by_name:
        raise SystemExit(f"no runs named {args.baseline!r}")

    base_lr, base = best_lr(by_name[args.baseline], args.metric)
    if len(base) < 2:
        raise SystemExit("need >= 2 baseline seeds at the best LR to estimate the noise floor")
    sigma = statistics.stdev(base)
    base_mean = statistics.mean(base)
    print(f"baseline @ lr {base_lr}: {base_mean:.4f} (n={len(base)}), sigma = {sigma:.4f}, 2 sigma = {2 * sigma:.4f}\n")
    print(f"{'variant':<22}{'best lr':>14}{'n':>4}{'mean':>10}{'delta':>10}{'range':>18}  verdict")
    for name in sorted(by_name):
        if name == args.baseline:
            continue
        lr, vals = best_lr(by_name[name], args.metric)
        if not vals:
            print(f"{name:<22}{'-':>14}{0:>4}  (no finished, non-diverged runs)")
            continue
        m = statistics.mean(vals)
        delta = m - base_mean
        if len(vals) < 3:
            verdict = "needs >= 3 seeds"
        elif abs(delta) < 2 * sigma:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "better" if delta < 0 else "worse"
        rng = f"[{min(vals):.4f}, {max(vals):.4f}]"
        print(f"{name:<22}{lr:>14}{len(vals):>4}{m:>10.4f}{delta:>+10.4f}{rng:>18}  {verdict}")


if __name__ == "__main__":
    main()
