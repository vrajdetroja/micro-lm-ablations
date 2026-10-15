"""Apply the study's decision rules to all_runs.csv.

1. Noise floor: sigma = std of the metric over baseline seeds (best baseline LR).
2. Every variant is compared optimum-to-optimum: for each variant, the LR with the best
   seed-mean metric, against the baseline at *its* best LR.
3. A difference smaller than 2 * sigma is reported as INCONCLUSIVE, not as a result.
4. Once variants have seeds of their own, sigma_pooled (within-variant variance pooled over
   every variant at its best LR) is a second noise estimate. An effect must clear 2 sigma
   under both; clearing only one is THRESHOLD-SENSITIVE (preregistration 002, shared rule 3).

    python analysis/compare.py [--csv results/all_runs.csv] [--metric val_iid_loss]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


# Fields that may differ between runs of one variant: what the protocol sweeps, plus
# operational settings that do not change the trained model.
VARYING = {"run_dir", "train.lr", "train.seed", "train.muon_lr", "train.micro_batch_size", "train.compile",
           "train.device", "train.eval_interval", "train.log_interval", "train.diag_interval",
           "train.ckpt_interval", "train.keep_ckpts", "train.keep_ckpts_full", "train.peak_flops",
           "train.wandb_project", "train.out_dir", "train.verify_data", "train.stop_at_step",
           "train.extrapolation_lengths", "train.extrapolation_docs"}


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def comparable(rows: list[dict]) -> list[dict]:
    """Runs that enter comparisons: those in the top-level runs/ directory.

    Special-purpose runs (resume checks, validation, same-seed repeats) are written to
    subdirectories and never mix in. As a second guard, every comparable run of one name must
    share its config except for the fields in VARYING."""
    def top_level(r: dict) -> bool:  # the run's actual location decides; out_dir is the fallback
        return "/" not in r["run_dir"] if r.get("run_dir") else r.get("train.out_dir", "runs") == "runs"

    top = [r for r in rows if top_level(r)]
    by_name: dict[str, set] = defaultdict(set)
    for r in top:
        key = tuple(sorted((k, v) for k, v in r.items() if k.startswith(("model.", "train.")) and k not in VARYING))
        by_name[r["name"]].add(key)
    for name, keys in by_name.items():
        if len(keys) > 1:
            a, b = (dict(k) for k in list(keys)[:2])
            diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
            raise SystemExit(f"runs named {name!r} have different configs ({', '.join(diff)}); "
                             "give special-purpose runs their own name or out_dir")
    return top


def best_lr(rows: list[dict], metric: str, fresh: bool = False) -> tuple[str, list[float]]:
    """Group a variant's runs by LR; return the LR with the lowest seed-mean, and its values.

    With fresh=True, return only the values of seeds other than 0: seed 0 chose the LR, so its
    value is biased low, and the fresh seeds give an unbiased estimate at that LR.
    """
    by_lr: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r.get(metric) not in (None, "") and r.get("diverged") != "True":
            lr_key = r["train.lr"] if r["train.optimizer"] != "muon" else f"{r['train.lr']}/{r['train.muon_lr']}"
            by_lr[lr_key].append((int(r["train.seed"]), float(r[metric])))
    if not by_lr:
        return "-", []
    # Choose the LR exactly as protocol.py does, by the seed-0 sweep. Choosing by the mean over
    # whatever seeds exist would let an LR with one lucky run beat one with three honest runs.
    seed0 = {k: v for k in by_lr for seed, v in by_lr[k] if seed == 0}
    if seed0:
        lr = min(seed0, key=seed0.get)
    else:
        lr = min(by_lr, key=lambda k: statistics.mean(v for _, v in by_lr[k]))
    return lr, [v for seed, v in by_lr[lr] if not (fresh and seed == 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_runs.csv")
    ap.add_argument("--metric", default="val_iid_loss")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--best-lr", metavar="NAME", help="print only NAME's best LR (for the protocol scripts) and exit")
    args = ap.parse_args()

    rows = load(args.csv)
    if args.best_lr:
        lr, vals = best_lr([r for r in comparable(rows) if r["name"] == args.best_lr], args.metric)
        if not vals:
            raise SystemExit(f"no finished runs named {args.best_lr!r}")
        print(lr)
        return

    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in comparable(rows):
        by_name[r["name"]].append(r)
    repeats = [r for r in rows if r["name"] == f"{args.baseline}_repeat"]
    if args.baseline not in by_name:
        raise SystemExit(f"no runs named {args.baseline!r}")

    base_lr, base = best_lr(by_name[args.baseline], args.metric)
    if len(base) < 2:
        raise SystemExit("need >= 2 baseline seeds at the best LR to estimate the noise floor")
    sigma = statistics.stdev(base)
    base_mean = statistics.mean(base)
    print(f"baseline @ lr {base_lr}: {base_mean:.4f} (n={len(base)}), sigma = {sigma:.4f}, 2 sigma = {2 * sigma:.4f}")
    # sigma_kernel: same seed, same config, repeated. On a GPU with non-deterministic kernels these
    # still differ; sigma (across seeds) includes this noise, sigma_kernel isolates it.
    repeat = f"{args.baseline}_repeat"
    same_seed = [float(r[args.metric]) for r in by_name[args.baseline] + repeats
                 if r["train.lr"] == base_lr and r["train.seed"] == "0"
                 and r.get(args.metric) not in (None, "") and r.get("diverged") != "True"]
    if repeats and len(same_seed) >= 2:
        sigma_k = statistics.stdev(same_seed)
        print(f"sigma_kernel (seed 0 repeated, n={len(same_seed)}) = {sigma_k:.4f} ({sigma_k / sigma:.0%} of sigma)")
    # Pooled within-variant sigma: each variant's seeds at its best LR, pooled by degrees of freedom.
    num = df = 0
    for rs in by_name.values():
        _, vals = best_lr(rs, args.metric)
        if len(vals) >= 2:
            num += (len(vals) - 1) * statistics.variance(vals)
            df += len(vals) - 1
    sigma_p = (num / df) ** 0.5 if df > len(base) - 1 else None
    if sigma_p is not None:
        print(f"sigma_pooled = {sigma_p:.4f} ({df} df), 2 sigma_pooled = {2 * sigma_p:.4f}")
    thresholds = [2 * sigma] + ([2 * sigma_p] if sigma_p is not None else [])
    print()
    print(f"{'variant':<22}{'best lr':>14}{'n':>4}{'mean':>10}{'delta':>10}{'fresh':>10}{'range':>18}  verdict")
    for name in sorted(by_name):
        if name in (args.baseline, f"{args.baseline}_repeat"):
            continue
        lr, vals = best_lr(by_name[name], args.metric)
        if not vals:
            print(f"{name:<22}{'-':>14}{0:>4}  (no finished, non-diverged runs)")
            continue
        m = statistics.mean(vals)
        delta = m - base_mean
        if len(vals) < 3:
            verdict = "needs >= 3 seeds"
        else:
            clears = [abs(delta) >= t for t in thresholds]
            if all(clears):
                verdict = "better" if delta < 0 else "worse"
            elif any(clears):
                verdict = "THRESHOLD-SENSITIVE"
            else:
                verdict = "INCONCLUSIVE"
        rng = f"[{min(vals):.4f}, {max(vals):.4f}]"
        _, fresh = best_lr(by_name[name], args.metric, fresh=True)
        fresh_s = f"{statistics.mean(fresh):.4f}" if fresh else "-"
        print(f"{name:<22}{lr:>14}{len(vals):>4}{m:>10.4f}{delta:>+10.4f}{fresh_s:>10}{rng:>18}  {verdict}")


if __name__ == "__main__":
    main()
