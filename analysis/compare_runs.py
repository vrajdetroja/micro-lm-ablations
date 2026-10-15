"""Compare the loss trajectories of two runs of the same config (resume-determinism check).

    python analysis/compare_runs.py runs/validate/a runs/validate/b
"""

import json
import sys
from pathlib import Path


def losses(root: str) -> list[float]:
    (path,) = Path(root).glob("*/metrics.jsonl")
    return [json.loads(l)["train_loss"] for l in path.read_text().splitlines() if l]


a, b = losses(sys.argv[1]), losses(sys.argv[2])
assert len(a) == len(b), f"{len(a)} vs {len(b)} steps"
diffs = [abs(x - y) for x, y in zip(a, b)]
print(f"{len(a)} steps; bitwise equal: {a == b}; max |delta loss| = {max(diffs):.3e}; "
      f"final {a[-1]:.6f} vs {b[-1]:.6f}")
sys.exit(0 if max(diffs) < 1e-3 else 1)
