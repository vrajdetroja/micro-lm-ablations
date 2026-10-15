"""Gather every runs/**/summary.json into results/all_runs.csv: one row per run, every
config field as a column. Every figure in the report is regenerated from this file.

    python analysis/collect.py [--runs runs] [--out results/all_runs.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SKIP = {"val_iid_position_loss", "induction_prefix_matching", "config", "hardware", "params", "flops_per_token"}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def row(summary: dict) -> dict:
    r = {k: v for k, v in summary.items() if k not in SKIP}
    r.update(flatten(summary["config"]["model"], "model."))
    r.update(flatten(summary["config"]["train"], "train."))
    r.update(flatten(summary["params"], "params."))
    r.update(flatten(summary["flops_per_token"], "flops_per_token."))
    r.update(flatten(summary["hardware"], "hw."))
    return r


def collect(runs_dir: str | Path) -> list[dict]:
    rows = []
    for p in sorted(Path(runs_dir).rglob("summary.json")):
        r = row(json.loads(p.read_text()))
        r["run_dir"] = str(p.parent.relative_to(runs_dir))  # where the run actually lives
        rows.append(r)
    return sorted(rows, key=lambda r: (r["run_id"], r["run_dir"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/all_runs.csv")
    args = ap.parse_args()
    rows = collect(args.runs)
    cols = sorted({k for r in rows for k in r}, key=lambda c: (c.count("."), c))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
