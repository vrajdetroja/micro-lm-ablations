"""Launch a grid of runs, one subprocess each, skipping runs that already finished.

    python experiments/launch/sweep.py --config configs/baseline.yaml \
        --grid train.lr=1e-3,2e-3,3e-3,6e-3,1e-2 --grid train.seed=0,1,2

    # split the same grid over 4 GPUs / machines
    CUDA_VISIBLE_DEVICES=0 python experiments/launch/sweep.py ... --shard 0/4

Because the trainer resumes from checkpoints and skips finished runs, rerunning the same
command after an interruption picks up exactly where the sweep left off.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time


def parse_grid(items: list[str]) -> list[list[str]]:
    axes = []
    for item in items:
        key, _, values = item.partition("=")
        axes.append([f"{key}={v}" for v in values.split(",")])
    return axes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="one or more config files")
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    ap.add_argument("--set", nargs="*", default=[], help="fixed overrides for every run")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    combos = list(itertools.product(args.config, *parse_grid(args.grid)))
    jobs = [c for k, c in enumerate(combos) if k % n == i]
    print(f"{len(jobs)} of {len(combos)} jobs in shard {args.shard}")
    failures = []
    for k, (config, *overrides) in enumerate(jobs):
        cmd = [sys.executable, "-m", "microlm.train.trainer", "--config", config, "--set", *args.set, *overrides]
        print(f"[{k + 1}/{len(jobs)}] {' '.join(cmd[3:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        print(f"  exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            failures.append(" ".join(cmd[3:]))
    if failures:
        print(f"{len(failures)} failed:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
