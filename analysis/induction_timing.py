"""C3 (exploratory, not preregistered): when do induction heads form, and does depth move it?

    python analysis/induction_timing.py      # needs the kept checkpoints (train.keep_ckpts)

For every kept checkpoint (steps 25, 50, 100, 200, 400, 800, 1620) of the baseline and the
depth-vs-width runs, this runs the induction probe (Olsson et al., 2022) on repeated random
tokens: the prefix-matching score of every head (attention from the second copy of a token to
the token that followed its first copy), copy accuracy, and in-context gain. Results go to
results/induction_timing.json and results/figures/fig12_induction_timing.png.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from microlm.eval.induction import induction_probe  # noqa: E402
from microlm.model import ModelConfig, Transformer  # noqa: E402
from microlm.train.checkpoint import load_checkpoint  # noqa: E402
from microlm.utils import pick_device  # noqa: E402

RUNS = ROOT / "runs"
SHAPES = {"dw_L2_d640": 2, "baseline": 6, "dw_L12_d256": 12, "dw_L24_d192": 24}
BEST_LR = {"dw_L2_d640": 0.002, "baseline": 0.004, "dw_L12_d256": 0.008, "dw_L24_d192": 0.008}
STEPS = (25, 50, 100, 200, 400, 800, 1620)


def runs_with_checkpoints():
    for p in sorted(RUNS.glob("*/summary.json")):
        s = json.loads(p.read_text())
        if s["name"] in SHAPES and s["config"]["train"]["lr"] == BEST_LR[s["name"]] and any(p.parent.glob("ckpt_step*.pt")):
            yield s, p.parent


@torch.no_grad()
def probe(ckpt: Path, device: str, seeds=(0, 1, 2)) -> dict:
    st = load_checkpoint(ckpt)
    cfg = ModelConfig.from_dict(st["run_config"]["model"])
    model = Transformer(cfg)
    model.load_state_dict(st["model"])
    model = model.to(device).eval()
    seq = st["run_config"]["train"]["seq_len"]
    res = [induction_probe(model, cfg.vocab_size, cfg.vocab_size - 1, device, n=seq // 2, seed=s) for s in seeds]
    pm = np.mean([r["prefix_matching"] for r in res], axis=0)  # (layers, heads)
    layer, head = np.unravel_index(int(pm.argmax()), pm.shape)
    return {"max_prefix_matching": float(pm.max()), "best_layer": int(layer), "best_head": int(head),
            "layer_max": pm.max(axis=1).tolist(),
            "copy_accuracy": statistics.mean(r["copy_accuracy"] for r in res),
            "in_context_gain": statistics.mean(r["in_context_gain"] for r in res)}


def main() -> None:
    device = pick_device("auto")
    out = {}
    for s, d in runs_with_checkpoints():
        key = f"{s['name']}/seed{s['config']['train']['seed']}"
        traj = {}
        for step in STEPS:
            ck = d / f"ckpt_step{step}.pt"
            if ck.exists():
                traj[step] = probe(ck, device)
        metrics = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l]
        out[key] = {"name": s["name"], "layers": SHAPES[s["name"]], "seed": s["config"]["train"]["seed"],
                    "probe": traj, "train_loss": [(m["step"], m["train_loss"]) for m in metrics]}
        pm = [round(traj[t]["max_prefix_matching"], 2) for t in sorted(traj)]
        print(f"  {key:<24} prefix-matching by step {sorted(traj)}: {pm}", flush=True)
    (ROOT / "results" / "induction_timing.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figures as F  # shared style and palette

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 3.9))
    for k, (name, L) in enumerate(SHAPES.items()):
        runs = [v for v in out.values() if v["name"] == name]
        if not runs:
            continue
        label = f"{L} × {({2: 640, 6: 384, 12: 256, 24: 192})[L]}"
        steps = sorted({int(t) for r in runs for t in r["probe"]})
        for ax, key in ((a, "max_prefix_matching"), (b, "copy_accuracy")):
            M = np.array([[r["probe"][str(t) if str(t) in r["probe"] else t][key] for t in steps] for r in runs])
            if len(runs) > 1:
                ax.fill_between(steps, M.min(0), M.max(0), color=F.S[k], alpha=0.10, lw=0)
            ax.plot(steps, M.mean(0), color=F.S[k], **F.MK, label=label)
        # training loss, lightly smoothed, mean over seeds
        T = np.array([[l for _, l in r["train_loss"]] for r in runs])
        x = np.array([s for s, _ in runs[0]["train_loss"]])
        y = np.convolve(T.mean(0), np.ones(9) / 9, mode="same")
        c.plot(x[4:-4], y[4:-4], color=F.S[k], label=label)
    for ax, ylab, title in ((a, "max prefix-matching score (any head)", "Induction heads form at steps 400-800, at every depth"),
                            (b, "copy accuracy on repeated random tokens", "In-context copying keeps improving"),
                            (c, "training loss (9-step moving mean)", "Training loss over the same steps")):
        ax.set_xscale("log")
        ax.set_xlabel("optimizer step (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8.5)
    c.set_ylim(3.3, 7.5)
    fig.text(0.01, -0.04, "Exploratory (not preregistered). Checkpoints at steps 25, 50, 100, 200, 400, 800, 1620; "
             "3 seeds for the depth variants (range shaded), 1 for 6 × 384.", color=F.INK2, fontsize=8.5)
    F.save(fig, "fig12_induction_timing.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        plot({k: v for k, v in json.loads((ROOT / "results" / "induction_timing.json").read_text()).items()})
    else:
        main()
