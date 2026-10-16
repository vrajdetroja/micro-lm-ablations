"""C1 scaling-law fit and C2 cross-scale check (preregistration 009).

    python analysis/scaling.py

C1: L(N) = A * N^(-alpha) + E over the width ladder (6 layers, d in {128, 192, 256, 384, 512}),
using each width's seed mean at its best LR and N = N_non_embedding. The fit profiles over E
on a grid below the smallest observed loss; for each E, alpha and A come from linear
regression of log(L - E) on log N, and the E with the smallest squared error in L wins.
95% CIs come from 2,000 bootstrap resamples, each drawing one seed per width.

C2: gap(N) = RoPE mean - learned mean at each width, both at their own best LR.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import best_lr, comparable, load  # noqa: E402

WIDTHS = (128, 192, 256, 384, 512)
FAMILIES = {
    "learned": {d: ("baseline" if d == 384 else f"ladder_d{d}") for d in WIDTHS},
    "rope": {d: ("pos_rope" if d == 384 else f"ladder_d{d}_rope") for d in WIDTHS},
}
SIGMA_SEED = 0.0036  # preregistration 001


def fit(N: np.ndarray, L: np.ndarray, grid: int = 4000) -> tuple[float, float, float, float]:
    """Profile least squares, vectorized over the E grid: returns (alpha, A, E, sse)."""
    E = np.linspace(0.0, L.min() - 1e-4, grid)[:, None]          # (G, 1)
    x = np.log(N)[None, :]                                       # (1, k)
    y = np.log(L[None, :] - E)                                   # (G, k)
    xm, ym = x.mean(), y.mean(axis=1, keepdims=True)
    slope = ((x - xm) * (y - ym)).sum(axis=1) / ((x - xm) ** 2).sum()
    intercept = ym[:, 0] - slope * xm
    alpha, A = -slope, np.exp(intercept)
    pred = A[:, None] * N[None, :] ** (-alpha[:, None]) + E
    sse = ((pred - L[None, :]) ** 2).sum(axis=1)
    i = int(np.argmin(sse))
    return float(alpha[i]), float(A[i]), float(E[i, 0]), float(sse[i])


def points(rows: list[dict], family: str) -> dict[int, dict]:
    out = {}
    for d, name in FAMILIES[family].items():
        rs = [r for r in rows if r["name"] == name]
        lr, vals = best_lr(rs, "val_iid_loss")
        N = float(next(r for r in rs if r["train.lr"] == lr)["params.non_embedding"])
        out[d] = {"name": name, "lr": lr, "vals": vals, "N": N, "mean": statistics.mean(vals)}
    return out


def main() -> None:
    rows = comparable(load("results/all_runs.csv"))
    rng = np.random.default_rng(0)
    results = {}
    for family in FAMILIES:
        P = points(rows, family)
        N = np.array([P[d]["N"] for d in WIDTHS])
        L = np.array([P[d]["mean"] for d in WIDTHS])
        alpha, A, E, sse = fit(N, L)
        boots = []
        for _ in range(2000):
            Lb = np.array([rng.choice(P[d]["vals"]) for d in WIDTHS])
            boots.append(fit(N, Lb))
        ab = np.array([b[0] for b in boots])
        eb = np.array([b[2] for b in boots])
        pred = A * N ** (-alpha) + E
        print(f"\n== C1 {family}: L(N) = {A:.3g} * N^(-{alpha:.3f}) + {E:.3f}")
        print(f"   alpha 95% CI [{np.percentile(ab, 2.5):.3f}, {np.percentile(ab, 97.5):.3f}]   "
              f"E 95% CI [{np.percentile(eb, 2.5):.3f}, {np.percentile(eb, 97.5):.3f}]")
        print(f"   {'width':>6}{'N':>9}{'best lr':>9}{'n':>3}{'mean':>9}{'fit':>9}{'resid':>9}{'seed sd':>9}  resid/sd")
        for i, d in enumerate(WIDTHS):
            sd = statistics.stdev(P[d]["vals"])
            r = L[i] - pred[i]
            print(f"   {d:>6}{N[i] / 1e6:>8.2f}M{P[d]['lr']:>9}{len(P[d]['vals']):>3}{L[i]:>9.4f}{pred[i]:>9.4f}{r:>+9.4f}{sd:>9.4f}  {abs(r) / sd:5.1f}")
        results[family] = {"alpha": alpha, "A": A, "E": E, "alpha_ci": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))],
                           "E_ci": [float(np.percentile(eb, 2.5)), float(np.percentile(eb, 97.5))],
                           "points": {str(d): {k: v for k, v in P[d].items() if k != "vals"} | {"vals": P[d]["vals"]} for d in WIDTHS}}

    # C2: the RoPE gap at each width; seed CI of the difference from a bootstrap over seeds.
    print("\n== C2 RoPE - learned at each width (nats; both at their own best LR)")
    PL, PR = points(rows, "learned"), points(rows, "rope")
    print(f"   {'width':>6}{'N':>9}{'gap':>9}{'95% CI':>20}{'gap / 2 sigma_seed':>21}")
    for d in WIDTHS:
        gap = PR[d]["mean"] - PL[d]["mean"]
        bs = [statistics.mean(rng.choice(PR[d]["vals"], len(PR[d]["vals"]))) -
              statistics.mean(rng.choice(PL[d]["vals"], len(PL[d]["vals"]))) for _ in range(2000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"   {d:>6}{PL[d]['N'] / 1e6:>8.2f}M{gap:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]{gap / (2 * SIGMA_SEED):>21.1f}")
        results.setdefault("c2", {})[str(d)] = {"gap": gap, "ci": [float(lo), float(hi)]}
    Path("results/scaling.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results/scaling.json")


if __name__ == "__main__":
    main()
