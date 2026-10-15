"""Launch a grid of runs, one subprocess each, skipping runs that already finished.

    python experiments/launch/sweep.py --config configs/baseline.yaml \
        --grid train.lr=1e-3,2e-3,4e-3,8e-3,1.6e-2

    # list every job with its training FLOPs, and run each for one step on CPU
    python experiments/launch/sweep.py --config configs/ablations/pos_*.yaml \
        --grid train.lr=2e-3,4e-3,8e-3 --dry-run --preflight

    # split the same grid over several GPUs
    CUDA_VISIBLE_DEVICES=0 python experiments/launch/sweep.py ... --shard 0/2

The trainer resumes from checkpoints and skips finished runs, so rerunning the same
command after an interruption picks up exactly where the sweep left off.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from microlm.model import ModelConfig, Transformer, flops_per_token  # noqa: E402
from microlm.train.config import RunConfig, load_run_config  # noqa: E402


def parse_grid(items: list[str]) -> list[list[str]]:
    axes = []
    for item in items:
        key, _, values = item.partition("=")
        axes.append([f"{key}={v}" for v in values.split(",")])
    return axes


def vocab_for(run: RunConfig) -> int:
    """The vocab the run will get from its data: the manifest if present, else the dir name."""
    manifest = REPO / run.train.data_dir / "MANIFEST.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["tokenizer"]["vocab_size"]
    m = re.search(r"bpe_(\d+)", run.train.data_dir)
    if not m:
        raise ValueError(f"cannot infer vocab size for {run.train.data_dir}")
    return int(m.group(1))


def train_flops(run: RunConfig) -> float:
    cfg = ModelConfig.from_dict({**run.model.to_dict(), "vocab_size": vocab_for(run)})
    return flops_per_token(cfg, run.train.seq_len)["total"] * run.train.total_steps * run.train.tokens_per_step


def preflight(run: RunConfig) -> None:
    """One optimizer step of the full-size model on CPU with a tiny batch: catches config,
    shape, and optimizer errors before a full run starts."""
    import torch

    from microlm.train.optim import build_optimizer

    cfg = ModelConfig.from_dict({**run.model.to_dict(), "vocab_size": vocab_for(run)})
    model = Transformer(cfg)
    model.check_param_counts()
    opt = build_optimizer(model, run.train)
    T = min(run.train.seq_len, 64)
    idx = torch.randint(0, cfg.vocab_size, (2, T + 1))
    _, loss = model(idx[:, :-1], idx[:, 1:])
    loss.backward()
    opt.step()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss at init")
    data = REPO / run.train.data_dir / "MANIFEST.json"
    if data.exists():
        have = json.loads(data.read_text())["splits"]["train"]["n_tokens"]
        need = run.train.total_steps * run.train.tokens_per_step + 1
        if have < need:
            raise RuntimeError(f"train split has {have:,} tokens; the run needs {need:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True, help="one or more config files")
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (repeatable)")
    ap.add_argument("--set", nargs="*", default=[], help="fixed overrides for every run")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--dry-run", action="store_true", help="list jobs and their training FLOPs; run nothing")
    ap.add_argument("--preflight", action="store_true", help="one CPU step per job before launching")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    combos = list(itertools.product(args.config, *parse_grid(args.grid)))
    jobs = [c for k, c in enumerate(combos) if k % n == i]

    runs, seen, total_flops = [], {}, 0.0
    for config, *overrides in jobs:
        run = load_run_config(config, [*args.set, *overrides])
        rid = run.run_id()
        if rid in seen:
            raise SystemExit(f"duplicate run: {config} {overrides} == {seen[rid]}")
        seen[rid] = f"{config} {overrides}"
        flops = train_flops(run)
        runs.append((config, overrides, run, flops))
        total_flops += flops

    print(f"{len(jobs)} of {len(combos)} jobs in shard {args.shard}; "
          f"{total_flops:.3g} training FLOPs in total")
    if args.preflight:
        bad = 0
        for k, (config, overrides, run, _) in enumerate(runs):
            try:
                preflight(run)
            except Exception as e:  # report every failure, not just the first
                print(f"  PREFLIGHT FAIL [{k + 1}] {config} {overrides}: {type(e).__name__}: {e}")
                bad += 1
        if bad:
            raise SystemExit(f"{bad} of {len(runs)} jobs failed preflight")
        print(f"preflight ok: {len(runs)} jobs ran one CPU step")

    failures = []
    for k, (config, overrides, run, flops) in enumerate(runs):
        cmd = [sys.executable, "-m", "microlm.train.trainer", "--config", config, "--set", *args.set, *overrides]
        print(f"[{k + 1}/{len(runs)}] {flops:.3g} FLOPs  {' '.join(cmd[3:])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=REPO)
        print(f"  exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            failures.append(" ".join(cmd[3:]))
    if failures:
        print(f"{len(failures)} failed:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
