"""The per-variant protocol, unattended: LR sweep, edge extension, then seeds at the optimum.

    python experiments/launch/protocol.py --config configs/ablations/pos_*.yaml --center auto

For each config:
  1. seed 0 at center/f, center, center*f
  2. if the best LR is at an edge of what has been tried, one more point beyond it
     (at most --max-extend times)
  3. seeds 1..k at the best LR

--center is a number, `auto` (the baseline's best LR from finished runs), or `own` (the
config's own value of --lr-key, e.g. Lion's lower LR or Muon's muon_lr). Diverged runs rank
last. Runs that already finished are reused, so the command is safe to rerun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from microlm.train.config import load_run_config  # noqa: E402


def finished(name: str, out_dir: str) -> list[dict]:
    rows = []
    for p in (REPO / out_dir).glob(f"{name}__*/summary.json"):
        s = json.loads(p.read_text())
        if s["name"] == name:
            rows.append(s)
    return rows


def get(cfg: dict, key: str):
    section, field = key.split(".")
    return cfg[section][field]


def score(s: dict) -> float:
    return float("inf") if s.get("diverged") or s.get("val_iid_loss") is None else s["val_iid_loss"]


def run(config: str, overrides: list[str], dry: bool) -> None:
    cmd = [sys.executable, "-m", "microlm.train.trainer", "--config", config, "--set", *overrides]
    print("  " + " ".join(cmd[3:]), flush=True)
    if not dry and subprocess.call(cmd, cwd=REPO) != 0:
        print("  (run failed; it ranks as diverged)", flush=True)


def baseline_best(key: str, out_dir: str) -> float:
    rows = [s for s in finished("baseline", out_dir) if s["config"]["train"]["seed"] == 0]
    if not rows:
        raise SystemExit("--center auto needs finished baseline runs (protocol 01)")
    return get(min(rows, key=score)["config"], key)


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--center", default="auto")
    ap.add_argument("--lr-key", default="train.lr")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--max-extend", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], help="fixed overrides for every run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for config in args.config:
        rc = load_run_config(config, args.set)
        out_dir = rc.train.out_dir
        if args.center == "auto":
            center = baseline_best(args.lr_key, out_dir)
        elif args.center == "own":
            center = get(rc.to_dict(), args.lr_key)
        else:
            center = float(args.center)
        print(f"{rc.name}: {args.lr_key} centered at {fmt(center)}", flush=True)

        lrs = [center / args.factor, center, center * args.factor]
        for lr in lrs:
            run(config, [*args.set, f"{args.lr_key}={fmt(lr)}", "train.seed=0"], args.dry_run)
        for _ in range(args.max_extend):
            if args.dry_run:
                break
            seed0 = {fmt(get(s["config"], args.lr_key)): s for s in finished(rc.name, out_dir)
                     if s["config"]["train"]["seed"] == 0}
            tried = sorted(float(k) for k in seed0)
            best = min(tried, key=lambda lr: score(seed0[fmt(lr)]))
            if best == tried[0]:
                nxt = best / args.factor
            elif best == tried[-1]:
                nxt = best * args.factor
            else:
                break
            print(f"  best {fmt(best)} is at an edge; extending to {fmt(nxt)}", flush=True)
            run(config, [*args.set, f"{args.lr_key}={fmt(nxt)}", "train.seed=0"], args.dry_run)

        if args.dry_run:
            best = center
        else:
            seed0 = [s for s in finished(rc.name, out_dir) if s["config"]["train"]["seed"] == 0]
            best = get(min(seed0, key=score)["config"], args.lr_key)
        print(f"  seeds {args.seeds} at {fmt(best)}", flush=True)
        for seed in args.seeds:
            run(config, [*args.set, f"{args.lr_key}={fmt(best)}", f"train.seed={seed}"], args.dry_run)


if __name__ == "__main__":
    main()
