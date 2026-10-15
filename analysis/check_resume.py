"""Is resuming on this GPU as good as never stopping?

    python analysis/check_resume.py runs/resume_check/a runs/resume_check/b \
        runs/resume_check/c --resume-step 100

  a  uninterrupted run
  b  the same config, stopped at --resume-step and resumed
  c  the same config, uninterrupted again (same seed)

With non-deterministic GPU kernels, a and c already drift apart, so "b equals a" is the wrong
test. Resuming is correct if, after the resume point, b stays as close to a as c does:

    mean |a - b|  <=  3 x mean |a - c| + 0.005      (train loss, steps after the resume)

A broken resume (optimizer state lost, wrong data position, wrong LR) is a jump far outside
that envelope. Exit status 0 if the check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def trajectory(root: str) -> tuple[list[float], dict]:
    (metrics,) = Path(root).glob("*/metrics.jsonl")
    losses = [json.loads(line)["train_loss"] for line in metrics.read_text().splitlines() if line]
    summary_path = metrics.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return losses, summary


def identical_prefix(x: list[float], y: list[float]) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def mean_abs_diff(x: list[float], y: list[float], start: int) -> float:
    d = [abs(p - q) for p, q in zip(x[start:], y[start:])]
    return sum(d) / len(d)


def verdict(ab: float, ac: float, factor: float = 3.0, floor: float = 0.005) -> bool:
    return ab <= factor * ac + floor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c")
    ap.add_argument("--resume-step", type=int, required=True)
    args = ap.parse_args()

    (a, sa), (b, sb), (c, sc) = trajectory(args.a), trajectory(args.b), trajectory(args.c)
    if not (len(a) == len(b) == len(c)):
        print(f"FAIL: runs have {len(a)}, {len(b)}, {len(c)} steps; b probably did not resume")
        sys.exit(1)
    k = args.resume_step
    ab, ac = mean_abs_diff(a, b, k), mean_abs_diff(a, c, k)
    first_ab = abs(a[k] - b[k])  # the first step after resuming
    ok = verdict(ab, ac)
    print(f"steps: {len(a)}; resumed after step {k}")
    print(f"identical prefix: a-b {identical_prefix(a, b)} steps, a-c {identical_prefix(a, c)} steps "
          "(GPU kernel noise breaks both after a few steps)")
    print(f"mean |train loss diff| after step {k}: a-b (resumed) {ab:.4f}, a-c (fresh repeat) {ac:.4f}")
    print(f"|a-b| at step {k + 1}, right after resuming: {first_ab:.4f}")
    if sa and sb and sc:
        print(f"final val_iid: a {sa['val_iid_loss']:.4f}, b {sb['val_iid_loss']:.4f}, c {sc['val_iid_loss']:.4f}")
    print("PASS: resuming adds no deviation beyond kernel noise" if ok
          else "FAIL: the resumed run deviates beyond kernel noise")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
