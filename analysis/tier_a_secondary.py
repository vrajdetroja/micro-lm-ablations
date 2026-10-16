"""Secondary Tier A analyses promised in preregistrations 002-006.

    python analysis/tier_a_secondary.py

002: length extrapolation on val_long (mean loss at L, and rise over each method's own L=512).
003/004: training throughput per variant.
004: the post-LN warmup demonstration grid.
006: equal-FLOPs and equal-wallclock comparisons, interpolated log-linearly from each run's
     periodic val curve (1M-token evals every 200 steps), with the baseline's wallclock scaled
     by 926/970 to the Tier A card (006 amendment).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import best_lr, comparable, load  # noqa: E402

RUNS = Path("runs")
CARD_SCALE = 926 / 970  # baseline card -> Tier A card, by benchmark throughput (006 amendment)


def summaries(name: str, lr: str, root: Path = RUNS) -> list[dict]:
    out = []
    for p in root.glob(f"{name}__*/summary.json"):
        s = json.loads(p.read_text())
        if s["name"] == name and repr(s["config"]["train"]["lr"]) == repr(float(lr)):
            out.append(s)
    return out


def best(rows, name):
    lr, _ = best_lr([r for r in rows if r["name"] == name], "val_iid_loss")
    return lr, summaries(name, lr)


def curve(s: dict, x: str) -> list[tuple[float, float]]:
    """(x, val_loss) points from a run's metrics; x = 'flops' or 'seconds'."""
    pts = []
    run_dir = next(RUNS.glob(f"{s['run_id']}/metrics.jsonl"))
    for line in run_dir.read_text().splitlines():
        r = json.loads(line)
        if "val_loss" in r and r["step"] > 1:
            xv = s["flops_per_token"]["total"] * r["tokens"] if x == "flops" else r["train_seconds"]
            pts.append((xv, r["val_loss"]))
    return pts


def interp(pts, x0):
    """Log-linear interpolation of val loss at x0; None outside the curve."""
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x0 <= xb:
            t = (math.log(x0) - math.log(xa)) / (math.log(xb) - math.log(xa))
            return ya + t * (yb - ya)
    return None


def main() -> None:
    rows = comparable(load("results/all_runs.csv"))

    print("== 002 length extrapolation (val_long, 200 docs; mean over seeds at each variant's best LR)")
    print(f"{'variant':<24}" + "".join(f"{'L=' + str(L):>10}" for L in (512, 1024, 2048, 4096)) + "   rise@2048  rise@4096")
    for name in ("baseline", "pos_rope", "pos_alibi", "pos_nope", "pos_sinusoidal_scaled", "pos_sinusoidal"):
        lr, ss = best(rows, name)
        cells, means = [], {}
        for L in ("512", "1024", "2048", "4096"):
            vals = [s["extrapolation"]["lengths"][L]["loss"] for s in ss if "loss" in s["extrapolation"]["lengths"].get(L, {})]
            means[L] = statistics.mean(vals) if vals else None
            cells.append(f"{means[L]:10.4f}" if vals else f"{'n/a':>10}")
        rise = lambda L: f"{means[L] - means['512']:+10.3f}" if means[L] is not None else f"{'n/a':>10}"
        print(f"{name:<24}" + "".join(cells) + f" {rise('2048')} {rise('4096')}")

    print("\n== throughput (training tokens/s, mean over best-LR seeds; the baseline ran on a different card)")
    for name in ("baseline", "pos_rope", "pos_alibi", "ffn_swiglu", "ffn_geglu", "ffn_relu2", "norm_rmsnorm",
                 "dw_L2_d640", "dw_L12_d256", "dw_L24_d192"):
        _, ss = best(rows, name)
        tps = statistics.mean(s["tokens"] / s["train_seconds"] for s in ss)
        print(f"  {name:<22} {tps / 1e3:7.1f}k tok/s")

    print("\n== 004 post-LN warmup demonstration (runs/warmup_demo; seed 0; val_iid nats; >7.3 = collapsed)")
    demo = [json.loads(p.read_text()) for p in (RUNS / "warmup_demo").glob("*/summary.json")]
    for s in sorted(demo, key=lambda s: (s["name"], s["config"]["train"]["lr"], s["config"]["train"]["warmup_steps"])):
        c = s["config"]
        v = s.get("val_iid_loss")
        tag = "COLLAPSED" if v is None or v > 7.0 else ""
        print(f"  {c['model']['norm_position']}-LN {c['model'].get('init_scheme', 'gpt2'):<5} lr {c['train']['lr']:<6} "
              f"warmup {c['train']['warmup_steps']:<4} {v if v is None else round(v, 4)}  {tag}")
    for name, lr in (("norm_post_torchinit", "0.001"), ("norm_post_torchinit", "0.002"), ("init_torch", "0.002")):
        ss = [s for s in summaries(name, lr) if s["config"]["train"]["seed"] == 0]
        if ss:
            print(f"  ref (protocol, warmup 200) {name} lr {lr}: {ss[0]['val_iid_loss']:.4f}")

    print("\n== 006 equal steps / equal FLOPs / equal wallclock")
    print("  Periodic evals use the first 1M val_iid tokens, so their absolute values differ from the")
    print("  final 10M-token eval; compare deltas. Both runs are read at the same amount: the smaller of")
    print("  the two totals (a variant cheaper than the baseline is compared at its own endpoint).")
    _, base = best(rows, "baseline")
    bF = statistics.mean(s["train_flops"] for s in base)
    bW = statistics.mean(s["train_seconds"] for s in base) * CARD_SCALE
    bcurves = {x: [[(px * (CARD_SCALE if x == "seconds" else 1.0), py) for px, py in curve(s, x)] for s in base]
               for x in ("flops", "seconds")}
    base_steps = statistics.mean(s["val_iid_loss"] for s in base)
    print(f"  baseline: {bF:.3g} FLOPs, {bW / 60:.2f} min training (scaled to the Tier A card)")
    print(f"  {'variant':<14}{'d steps':>10}{'d FLOPs':>10}{'d wallclock':>13}   (variant minus baseline, nats)")
    saved = {}
    for name in ("dw_L2_d640", "dw_L12_d256", "dw_L24_d192"):
        _, ss = best(rows, name)
        out = {"steps": statistics.mean(s["val_iid_loss"] for s in ss) - base_steps}
        for x, btotal in (("flops", bF), ("seconds", bW)):
            deltas = []
            for s in ss:
                vc = curve(s, x)
                target = min(btotal, vc[-1][0])
                v = interp(vc, target)
                bvals = [interp(bc, min(target, bc[-1][0])) for bc in bcurves[x]]
                bvals = [b for b in bvals if b is not None]
                if v is not None and bvals:
                    deltas.append(v - statistics.mean(bvals))
            out[x] = statistics.mean(deltas) if deltas else float("nan")
        print(f"  {name:<14}{out['steps']:>+10.4f}{out['flops']:>+10.4f}{out['seconds']:>+13.4f}")
        saved[name] = out
    Path("results/depth_matched.json").write_text(json.dumps(saved, indent=2))

if __name__ == "__main__":
    main()
