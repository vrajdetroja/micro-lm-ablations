"""Every figure in REPORT.md, regenerated from results/all_runs.csv and the run files.

    python analysis/figures.py            # writes results/figures/fig*.png

Style: one shared look (light surface, hairline solid grid, 2px lines, >= 8px markers
with a surface ring, legends whenever there are two or more series). Seed spread is
always shown (mean with min-max range), and every ablation plot carries the noise floor
as bands: +-2 sigma_seed (darker) and +-2 sigma_pooled (lighter). Colors follow the
validated reference palette's categorical order and are never cycled.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))
from compare import best_lr, comparable, load  # noqa: E402
from microlm.train.config import TrainConfig  # noqa: E402
from microlm.train.schedule import lr_multiplier  # noqa: E402
from tier_a_secondary import curve, summaries  # noqa: E402

OUT = ROOT / "results" / "figures"
RUNS = ROOT / "runs"

# Reference palette (light mode), used in its validated order.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BAND_SEED, BAND_POOL = "#dcdbd4", "#efeee9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

BASE = 3.5908          # baseline mean, 5 seeds (preregistration 001)
SIG_SEED = 0.0036      # sigma_seed (nats)
SIG_POOL = 0.0118      # sigma_pooled over unit-scale stable variants (compare.py --exclude-unstable)
BASE_BPB, SIG_SEED_BPB, SIG_POOL_BPB = 1.3218, 0.0013, 0.0042

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.markersize": 8, "legend.frameon": False, "legend.fontsize": 9,
})
MK = dict(marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)


def rows():
    return comparable(load(str(ROOT / "results" / "all_runs.csv")))


def best(R, name, metric="val_iid_loss"):
    lr, vals = best_lr([r for r in R if r["name"] == name], metric)
    return lr, vals


def best_summaries(R, name):
    lr, _ = best(R, name)
    if name == "opt_muon":  # its key is "adamw_lr/muon_lr"
        muon = float(lr.split("/")[1])
        return [json.loads(p.read_text()) for p in RUNS.glob("opt_muon__*/summary.json")
                if json.loads(p.read_text())["config"]["train"]["muon_lr"] == muon]
    return summaries(name, lr, RUNS)


def noise_bands(ax, horizontal=True, center=0.0, seed=SIG_SEED, pool=SIG_POOL, label=True):
    span = ax.axhspan if horizontal else ax.axvspan
    span(center - 2 * pool, center + 2 * pool, color=BAND_POOL, zorder=0, lw=0,
         label="±2σ pooled" if label else None)
    span(center - 2 * seed, center + 2 * seed, color=BAND_SEED, zorder=0, lw=0,
         label="±2σ seed" if label else None)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# -------------------------------------------------------------------------------------

def fig_noise_floor(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    sweep = sorted((float(r["train.lr"]), float(r["val_iid_loss"])) for r in R
                   if r["name"] == "baseline" and r["train.seed"] == "0")
    stable = [(x, y) for x, y in sweep if y < BASE + 0.5]
    a.plot([x for x, _ in stable], [y for _, y in stable], color=S[0], **MK, label="seed 0")
    seeds = [float(r["val_iid_loss"]) for r in R if r["name"] == "baseline" and r["train.lr"] == "0.004"]
    a.scatter([0.004] * len(seeds), seeds, s=22, color=INK2, zorder=3, label="seeds 0–4 at 4e-3")
    a.set_xscale("log", base=2)
    a.set_xticks([x for x, _ in sweep], [f"{x:g}" for x, _ in sweep])
    a.set_xlabel("peak learning rate")
    a.set_ylabel("val loss (nats/token)")
    a.set_title("Baseline LR sweep")
    for x, y in sweep:
        if y >= BASE + 0.5:
            a.annotate(f"LR {x:g}: {y:.2f}\n(unstable, off scale)", xy=(x, min(y2 for _, y2 in stable) + 0.04),
                       ha="right", va="bottom", color=INK2, fontsize=8)
    a.legend(loc="upper right")

    rep = [float(r["val_iid_loss"]) for r in load(str(ROOT / "results" / "all_runs.csv"))
           if r["name"] == "baseline_repeat"]
    seed0 = [float(r["val_iid_loss"]) for r in R if r["name"] == "baseline" and r["train.lr"] == "0.004" and r["train.seed"] == "0"]
    groups = [("5 seeds", seeds, S[0]), ("seed 0, 3 runs", seed0 + rep, S[1])]
    for i, (label, vals, c) in enumerate(groups):
        b.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=60, color=c,
                  edgecolor=SURFACE, linewidth=1.5, zorder=3)
        m = statistics.mean(vals)
        b.hlines(m, i - 0.25, i + 0.25, color=INK2, lw=1)
        b.annotate(f"σ = {statistics.stdev(vals):.4f}", xy=(i + 0.28, m), va="center", color=INK2, fontsize=9)
    b.set_xticks([0, 1], [g[0] for g in groups])
    b.set_xlim(-0.5, 1.9)
    b.set_ylabel("val loss (nats/token)")
    b.set_title("Noise floor at LR 4e-3")
    b.text(0.0, -0.22, "Same seed, same config: GPU kernel non-determinism alone gives 78% of σ.",
           transform=b.transAxes, color=INK2, fontsize=8.5)
    save(fig, "fig01_noise_floor.png")


GROUPS = [
    ("Positions", [("pos_rope", "RoPE"), ("pos_alibi", "ALiBi"), ("pos_nope", "NoPE"),
                   ("pos_sinusoidal_scaled", "sinusoid, scaled"), ("pos_sinusoidal", "sinusoid, raw")]),
    ("FFN", [("ffn_swiglu", "SwiGLU"), ("ffn_geglu", "GEGLU"), ("ffn_relu2", "ReLU²")]),
    ("Normalization", [("norm_rmsnorm", "RMSNorm"), ("norm_no_final", "no final norm"),
                       ("init_torch", "pre-LN, standard init"), ("norm_post_torchinit", "post-LN, standard init"),
                       ("norm_post", "post-LN, GPT-2 init")]),
    ("Schedule", [("sched_linear", "linear to 0"), ("sched_cosine0", "cosine to 0"), ("sched_wsd", "WSD")]),
    ("Depth × width", [("dw_L2_d640", "2 × 640"), ("dw_L12_d256", "12 × 256"), ("dw_L24_d192", "24 × 192")]),
    ("Optimizer", [("opt_muon", "Muon"), ("opt_lion", "Lion")]),
]


def verdict(delta):
    clears = [abs(delta) >= 2 * SIG_SEED, abs(delta) >= 2 * SIG_POOL]
    if all(clears):
        return "better" if delta < 0 else "worse"
    return "threshold-sensitive" if any(clears) else "inconclusive"


def fig_forest(R):
    items = []
    for g, members in GROUPS:
        items.append((g, None))
        items += [(label, name) for name, label in members]
    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    lo, hi = -0.14, 0.32
    noise_bands(ax, horizontal=False)
    y = 0
    ticks, labels = [], []
    for label, name in items:
        if name is None:
            y -= 0.6
            ax.text(lo, y, label, fontweight="semibold", color=INK, va="center", fontsize=9.5)
            y -= 1
            continue
        _, vals = best(R, name)
        d = [v - BASE for v in vals]
        m = statistics.mean(d)
        if m > hi:
            ax.annotate("off scale", xy=(hi, y), ha="right", va="center", color=INK2, fontsize=8.5)
            v = "worse (unstable)"
        else:
            ax.hlines(y, min(d), max(d), color=S[0], lw=2)
            ax.plot(m, y, color=S[0], **MK)
            v = verdict(m)
        ax.text(hi + 0.005, y, f"{m:+.3f}  {v}", va="center", color=INK2, fontsize=8.5)
        ticks.append(y)
        labels.append(label)
        y -= 1
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(y + 0.3, 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ val loss vs. baseline (nats/token; lower is better)  ·  dot = mean of 3 seeds, bar = range")
    ax.set_title("Every unit-scale variant against the 10.6M baseline (3.5908)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    save(fig, "fig02_forest_plot.png")


POS = [("pos_rope", "RoPE", S[0]), ("pos_alibi", "ALiBi", S[1]), ("pos_nope", "NoPE", S[2]),
       ("pos_sinusoidal_scaled", "sinusoid, scaled", S[3])]


def fig_extrapolation(R):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    Ls = [512, 1024, 2048, 4096]
    for name, label, c in POS:
        ss = best_summaries(R, name)
        M = np.array([[s["extrapolation"]["lengths"][str(L)]["loss"] for L in Ls] for s in ss])
        ax.fill_between(Ls, M.min(0), M.max(0), color=c, alpha=0.10, lw=0)
        ax.plot(Ls, M.mean(0), color=c, **MK, label=label)
    base = [s["extrapolation"]["lengths"]["512"]["loss"] for s in best_summaries(R, "baseline")]
    ax.plot([512], [statistics.mean(base)], color=S[4], linestyle="none", **MK, label="learned (512 only)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls, [str(L) for L in Ls])
    ax.set_xlim(440, 4800)
    ax.axvline(512, color=AXIS, lw=0.8)
    ax.set_xlabel("evaluation context length (tokens); trained at 512")
    ax.set_ylabel("val_long loss (nats/token)")
    ax.set_title("Only ALiBi extrapolates beyond the training length")
    ax.legend(loc="upper left")
    save(fig, "fig03_length_extrapolation.png")


def fig_position_loss(R):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, label, c in POS:
        ss = best_summaries(R, name)
        L = ss[0]["extrapolation"]["lengths"]["4096"]
        edges = L["bucket_edges"]
        x = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
        y = np.mean([s["extrapolation"]["lengths"]["4096"]["bucket_loss"] for s in ss], axis=0)
        ax.plot(x, y, color=c, label=label)
    ax.axvline(512, color=AXIS, lw=0.8)
    ax.annotate("training length", xy=(512, 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("token position within a 4,096-token window")
    ax.set_ylabel("loss at that position (nats)")
    ax.set_title("Where each method breaks: loss by position at L = 4096")
    ax.set_ylim(3.0, 10.5)
    ax.legend(loc="upper left")
    save(fig, "fig04_loss_by_position.png")


def fig_postln(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.4]})
    cells = {("pre", "gpt2"): "baseline", ("post", "gpt2"): "norm_post",
             ("pre", "torch"): "init_torch", ("post", "torch"): "norm_post_torchinit"}
    top = 4.0
    for j, (init, c) in enumerate((("gpt2", S[0]), ("torch", S[1]))):
        xs, ys, bad = [], [], []
        for i, place in enumerate(("pre", "post")):
            _, vals = best(R, cells[(place, init)])
            m = statistics.mean(vals)
            x = i + (j - 0.5) * 0.18
            (bad if m > top else xs).append(x)
            if m <= top:
                ys.append(m)
        label = "GPT-2 init" if init == "gpt2" else "standard (PyTorch) init"
        a.plot(xs, ys, color=c, **MK, label=label)
        if bad:
            a.plot(bad, [top + 0.08] * len(bad), color=c, linestyle="none", marker="X", markersize=9,
                   markeredgecolor=SURFACE, markeredgewidth=1.2)
    a.axhspan(top + 0.03, top + 0.13, color=BAND_POOL, lw=0, zorder=0)
    a.annotate("collapsed (7.34)", xy=(0.02, top + 0.14), xycoords=("axes fraction", "data"), color=INK2,
               fontsize=8, va="bottom")
    a.set_xticks([0, 1], ["pre-LN", "post-LN"])
    a.set_xlim(-0.5, 1.5)
    a.set_ylim(3.5, top + 0.25)
    a.set_ylabel("val loss (nats/token)")
    a.set_title("Norm placement × initialization")
    a.legend(loc="lower right")

    demo = [json.loads(p.read_text()) for p in (RUNS / "warmup_demo").glob("*/summary.json")]
    W = [0, 50, 200, 800]
    ref = {("post", 0.001): 3.8571, ("post", 0.002): 7.3380, ("pre", 0.002): 3.7705}  # protocol seed-0 runs
    series = [("post", 0.001, "post-LN, LR 1e-3", S[0]), ("post", 0.002, "post-LN, LR 2e-3", S[1]),
              ("pre", 0.001, "pre-LN, LR 1e-3", S[2]), ("pre", 0.002, "pre-LN, LR 2e-3", S[3])]
    cap = 4.45
    for k, (place, lr, label, c) in enumerate(series):
        pts = {}
        for s in demo:
            cfg = s["config"]
            if cfg["model"]["norm_position"] == place and cfg["train"]["lr"] == lr:
                pts[cfg["train"]["warmup_steps"]] = s["val_iid_loss"]
        if (place, lr) in ref:
            pts[200] = ref[(place, lr)]
        xs = [W.index(w) for w in sorted(pts)]
        ys = [pts[w] for w in sorted(pts)]
        ok = [(x, y) for x, y in zip(xs, ys) if y < 7]
        bad = [x for x, y in zip(xs, ys) if y >= 7]
        if ok:
            b.plot([x for x, _ in ok], [y for _, y in ok], color=c, **MK, label=label)
        if bad:
            b.plot([x + (k - 0.5) * 0.12 for x in bad], [cap] * len(bad), color=c, linestyle="none", marker="X", markersize=9,
                   markeredgecolor=SURFACE, markeredgewidth=1.2, label=None if ok else label)
    b.axhspan(cap - 0.06, cap + 0.06, color=BAND_POOL, lw=0, zorder=0)
    b.annotate("collapsed (stuck at the unigram loss, 7.34)", xy=(0.02, cap + 0.075), xycoords=("axes fraction", "data"),
               color=INK2, fontsize=8, va="bottom")
    b.set_xticks(range(4), [str(w) for w in W])
    b.set_xlabel("warmup steps (standard init)")
    b.set_ylim(3.6, cap + 0.18)
    b.set_ylabel("val loss (nats/token)")
    b.set_title("Post-LN needs warmup; pre-LN never collapses")
    b.legend(loc="center right", fontsize=8.5)
    save(fig, "fig05_postln_init_warmup.png")


def fig_depth(R):
    sec = json.loads((ROOT / "results" / "depth_matched.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    layers = [2, 6, 12, 24]
    names = {2: "dw_L2_d640", 12: "dw_L12_d256", 24: "dw_L24_d192"}
    noise_bands(a)
    for k, (key, label) in enumerate((("steps", "equal steps"), ("flops", "equal FLOPs"), ("seconds", "equal wallclock"))):
        y = [0.0 if L == 6 else sec[names[L]][key] for L in layers]
        a.plot(layers, y, color=S[k], **MK, label=label)
    a.set_xscale("log", base=2)
    a.set_xticks(layers, [f"{L} × {d}" for L, d in zip(layers, (640, 384, 256, 192))])
    a.set_ylabel("Δ val loss vs. 6 × 384 (nats)")
    a.set_title("6 layers wins under every matching")
    a.legend(loc="upper left")

    for k, (key, label) in enumerate((("induction_copy_accuracy", "copy accuracy"),)):
        means, lo, hi = [], [], []
        for L in layers:
            ss = best_summaries(R, "baseline" if L == 6 else names[L])
            v = [s[key] for s in ss]
            means.append(statistics.mean(v)); lo.append(min(v)); hi.append(max(v))
        b.fill_between(layers, lo, hi, color=S[0], alpha=0.10, lw=0)
        b.plot(layers, means, color=S[0], **MK)
    b.set_xscale("log", base=2)
    b.set_xticks(layers, [f"{L} × {d}" for L, d in zip(layers, (640, 384, 256, 192))])
    b.set_ylabel("copy accuracy on repeated random tokens")
    b.set_title("…but deeper models copy in context better")
    save(fig, "fig06_depth_width.png")


def fig_schedules(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    base = TrainConfig()
    sched = [("cosine to 10% (baseline)", dict(schedule="cosine", min_lr_ratio=0.1)),
             ("cosine to 0", dict(schedule="cosine", min_lr_ratio=0.0)),
             ("linear to 0", dict(schedule="linear", min_lr_ratio=0.0)),
             ("WSD (20% cooldown)", dict(schedule="wsd", decay_frac=0.2, min_lr_ratio=0.0))]
    steps = np.arange(base.total_steps)
    for k, (label, kw) in enumerate(sched):
        cfg = TrainConfig(**{**base.__dict__, **kw})
        a.plot(steps, [lr_multiplier(int(s), cfg) for s in steps], color=S[k], label=label)
    a.set_xlabel("optimizer step")
    a.set_ylabel("LR multiplier")
    a.set_title("The four schedules (peak LR 4e-3, 200 warmup steps)")
    a.legend(loc="lower left", fontsize=8.5)

    noise_bands(b)
    names = [("sched_cosine0", "cosine to 0"), ("sched_linear", "linear to 0"), ("sched_wsd", "WSD")]
    for i, (name, label) in enumerate(names):
        _, vals = best(R, name)
        d = [v - BASE for v in vals]
        b.vlines(i, min(d), max(d), color=S[0], lw=2)
        b.plot(i, statistics.mean(d), color=S[0], **MK)
        b.annotate(f"{statistics.mean(d):+.3f}", xy=(i, statistics.mean(d)), xytext=(10, 0),
                   textcoords="offset points", va="center", color=INK2, fontsize=9)
    b.axhline(0, color=AXIS, lw=0.8)
    b.set_xticks(range(3), [n[1] for n in names])
    b.set_xlim(-0.5, 2.7)
    b.set_ylabel("Δ val loss vs. cosine to 10% (nats)")
    b.set_title("WSD is the only clear win")
    b.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig07_schedules.png")


def fig_vocab(R):
    V = [2048, 4096, 8192, 16384]
    names = {2048: "vocab_2048", 4096: "vocab_4096", 8192: "baseline", 16384: "vocab_16384"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, metric, ylab, title in ((a, "val_iid_bpb", "val bits per byte", "Bits per byte: bigger vocab is better"),
                                    (b, "val_iid_loss", "val nats per token", "Per token: the opposite ranking")):
        m, lo, hi = [], [], []
        for v in V:
            _, vals = best(R, names[v], metric)
            m.append(statistics.mean(vals)); lo.append(min(vals)); hi.append(max(vals))
        if metric == "val_iid_bpb":
            noise_bands(ax, center=BASE_BPB, seed=SIG_SEED_BPB, pool=SIG_POOL_BPB)
            ax.legend(loc="upper right", fontsize=8.5)
        ax.fill_between(V, lo, hi, color=S[0], alpha=0.10, lw=0)
        ax.plot(V, m, color=S[0], **MK)
        for v, y in zip(V, m):
            ax.annotate(f"{y:.3f}", xy=(v, y), xytext=(0, 10), textcoords="offset points", ha="center",
                        color=INK2, fontsize=8.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(V, [f"{v // 1024}k" for v in V])
        ax.set_xlabel("vocabulary size")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    fig.text(0.01, -0.04, "Equal training FLOPs; total parameters matched by depth (2k and 4k: 7 layers, 8k: 6, "
             "16k: 4). Shaded: range over 3 seeds (5 for 8k).", color=INK2, fontsize=8.5)
    save(fig, "fig08_vocab_size.png")


def fig_optimizers(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    # The AdamW baseline and the Muon/Lion runs used two cards of the same model (926k and 952k
    # tokens/s on the throughput benchmark): scale the baseline's seconds to the second card.
    opts = [("baseline", "AdamW", S[0], 926 / 952), ("opt_muon", "Muon", S[1], 1.0), ("opt_lion", "Lion", S[2], 1.0)]
    for name, label, c, scale in opts:
        ss = best_summaries(R, name)
        tok = [[(x / s["flops_per_token"]["total"] / 1e6, y) for x, y in curve(s, "flops")] for s in ss]
        sec = [[(x * scale / 60, y) for x, y in curve(s, "seconds")] for s in ss]
        for ax, cur in ((a, tok), (b, sec)):
            xs = np.mean([[p[0] for p in cv] for cv in cur], axis=0)
            ys = np.mean([[p[1] for p in cv] for cv in cur], axis=0)
            ax.plot(xs, ys, color=c, label=label)
    for ax, xl in ((a, "training tokens (millions)"), (b, "training wallclock (minutes, scaled to one card)")):
        ax.set_xlabel(xl)
        ax.set_ylabel("val loss, periodic eval (nats)")
        ax.set_ylim(3.4, 4.6)
        ax.legend(loc="upper right")
    a.set_title("Muon learns faster per token…")
    b.set_title("…and still wins per second")
    save(fig, "fig09_optimizers.png")


def fig_scaling(R):
    sc = json.loads((ROOT / "results" / "scaling.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, (fam, label) in enumerate((("learned", "learned positions"), ("rope", "RoPE"))):
        P = sc[fam]["points"]
        N = np.array([P[d]["N"] for d in P])
        m = np.array([P[d]["mean"] for d in P])
        lo = np.array([min(P[d]["vals"]) for d in P]); hi = np.array([max(P[d]["vals"]) for d in P])
        ax.vlines(N, lo, hi, color=S[k], lw=2)
        ax.plot(N, m, color=S[k], linestyle="none", **MK, label=label)
        g = np.geomspace(N.min() * 0.85, N.max() * 1.2, 100)
        f = sc[fam]
        ax.plot(g, f["A"] * g ** (-f["alpha"]) + f["E"], color=S[k], lw=1.2, alpha=0.8)
        lo_a, hi_a = f["alpha_ci"]
        ax.annotate(f"α = {f['alpha']:.3f} [{lo_a:.2f}, {hi_a:.2f}]", xy=(g[-1], f["A"] * g[-1] ** (-f["alpha"]) + f["E"]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters (6 layers, width 128–512, ~20 tokens/param)")
    ax.set_ylabel("val loss (nats/token)")
    ax.set_title("Scaling ladder: L(N) = A·N^(−α) + E")
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.legend(loc="upper right")
    save(fig, "fig10_scaling_law.png")


def fig_cross_scale(R):
    sc = json.loads((ROOT / "results" / "scaling.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    noise_bands(ax)
    widths = list(sc["c2"])
    N = [sc["learned"]["points"][w]["N"] for w in widths]
    gap = [sc["c2"][w]["gap"] for w in widths]
    lo = [sc["c2"][w]["ci"][0] for w in widths]; hi = [sc["c2"][w]["ci"][1] for w in widths]
    ax.vlines(N, lo, hi, color=S[0], lw=2)
    ax.plot(N, gap, color=S[0], **MK)
    for n, g in zip(N, gap):
        ax.annotate(f"{g:+.3f}", xy=(n, g), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters")
    ax.set_ylabel("RoPE − learned (nats; < 0 means RoPE better)")
    ax.set_title("RoPE wins at every scale, by less and less")
    ax.legend(loc="lower right")
    save(fig, "fig11_rope_gap_across_scale.png")


def fig_lr_sweeps(R):
    names = [n for _, members in GROUPS for n, _ in members] + ["vocab_2048", "vocab_4096", "vocab_16384"]
    labels = {n: l for _, members in GROUPS for n, l in members} | {f"vocab_{v}": f"vocab {v}" for v in (2048, 4096, 16384)}
    cols = 6
    rows_n = -(-len(names) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(14, 2.1 * rows_n))
    for ax, name in zip(axes.flat, names):
        key = "train.muon_lr" if name == "opt_muon" else "train.lr"
        pts = sorted((float(r[key]), float(r["val_iid_loss"])) for r in R if r["name"] == name and r["train.seed"] == "0")
        ok = [(x, y) for x, y in pts if y < 7]
        if ok:
            ax.plot([x for x, _ in ok], [y for _, y in ok], color=S[0], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            bx, by = min(ok, key=lambda p: p[1])
            ax.plot(bx, by, color=S[1], marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_title(labels[name], fontsize=9)
        if not ok:  # every run collapsed: say so instead of drawing an empty grid
            ax.axis("off")
            ax.text(0.0, 0.55, f"all {len(pts)} LRs collapsed\nto the unigram loss (7.34)", transform=ax.transAxes,
                    color=INK2, fontsize=8.5, va="center")
            continue
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xticks([x for x, _ in pts], [f"{x:g}" for x, _ in pts])
        ax.tick_params(labelsize=7)
        if len(ok) < len(pts):
            ax.text(0.98, 0.95, f"{len(pts) - len(ok)} collapsed", transform=ax.transAxes, ha="right", va="top",
                    color=INK2, fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Seed-0 LR sweep for every variant (orange = the LR its seeds ran at). x: peak LR "
                 "(Muon: muon_lr), y: val loss (nats; vocab variants per their own tokenizer)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "fig13_lr_sweeps.png")


def fig_noise_by_variant(R):
    names = [n for _, members in GROUPS for n, _ in members if n not in ("norm_post", "pos_sinusoidal")]
    labels = {n: l for _, members in GROUPS for n, l in members}
    sd = sorted(((statistics.stdev(best(R, n)[1]), labels[n]) for n in names), reverse=True)
    sd.append((SIG_SEED, "baseline (5 seeds)"))
    sd.sort(reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    y = np.arange(len(sd))[::-1]
    ax.barh(y, [s for s, _ in sd], height=0.55, color=[S[1] if l.startswith("baseline") else S[0] for _, l in sd])
    ax.set_yticks(y, [l for _, l in sd])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("standard deviation over 3 seeds at the variant's best LR (nats)")
    ax.set_title("Noise depends on the variant")
    for yy, (s, _) in zip(y, sd):
        ax.annotate(f"{s:.4f}", xy=(s, yy), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=8)
    save(fig, "fig14_noise_by_variant.png")


def fig_ood(R):
    names = [n for _, members in GROUPS for n, _ in members if n not in ("norm_post", "pos_sinusoidal")]
    labels = {n: l for _, members in GROUPS for n, l in members}
    base_ood = statistics.mean(s["val_ood_bpb"] for s in best_summaries(R, "baseline"))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = []
    for n in names:
        ss = best_summaries(R, n)
        pts.append((statistics.mean(s["val_iid_loss"] for s in ss) - BASE,
                    statistics.mean(s["val_ood_bpb"] for s in ss) - base_ood, labels[n]))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=64, color=S[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    nudge = {"GEGLU": (-34, 8), "WSD": (-8, -14)}  # keep these two clear of neighbouring dots
    for x, y, l in pts:
        if l in ("RoPE", "ALiBi", "NoPE", "Muon", "WSD", "24 × 192", "post-LN, standard init", "GEGLU"):
            ax.annotate(l, xy=(x, y), xytext=nudge.get(l, (6, 4)), textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xlabel("Δ in-distribution val loss (nats; < 0 better)")
    ax.set_ylabel("Δ out-of-distribution bits per byte (Python code; < 0 better)")
    ax.set_title("In-distribution wins do not transfer to code")
    save(fig, "fig15_ood_vs_iid.png")


def main() -> None:
    R = rows()
    for f in (fig_noise_floor, fig_forest, fig_extrapolation, fig_position_loss, fig_postln, fig_depth,
              fig_schedules, fig_vocab, fig_optimizers, fig_scaling, fig_cross_scale, fig_lr_sweeps,
              fig_noise_by_variant, fig_ood):
        f(R)


if __name__ == "__main__":
    main()
