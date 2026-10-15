"""Measure real training throughput per config: the first thing to run on a new GPU.

    python experiments/launch/bench_throughput.py configs/baseline.yaml configs/ladder/*.yaml \
        --micro-batch 32 64 128

Runs full optimizer steps (with the config's gradient accumulation) on random tokens, after
compilation, and reports tokens/s, MFU, peak memory, and compile time. Results are written to
results/throughput.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "launch"))

from microlm.model import ModelConfig, Transformer, flops_per_token  # noqa: E402
from microlm.train.config import TrainConfig, load_run_config  # noqa: E402
from microlm.train.optim import build_optimizer  # noqa: E402
from microlm.train.trainer import peak_flops  # noqa: E402
from sweep import vocab_for  # noqa: E402


def bench(run, micro: int, steps: int, warmup: int, device: str, compile_mode: str | None) -> dict:
    tc = TrainConfig(**{**run.train.__dict__, "micro_batch_size": micro})
    cfg = ModelConfig.from_dict({**run.model.to_dict(), "vocab_size": vocab_for(run)})
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    opt = build_optimizer(model, tc)
    fwd = torch.compile(model, mode=compile_mode) if compile_mode is not None else model
    params = [p for p in model.parameters()]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    x = torch.randint(0, cfg.vocab_size, (micro, tc.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro, tc.seq_len), device=device)

    def step():
        for _ in range(tc.grad_accum):
            with autocast:
                _, loss = fwd(x, y)
            (loss / tc.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first = time.perf_counter() - t0  # includes compilation
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    tok_s = tc.tokens_per_step / dt
    peak = peak_flops(device, tc.peak_flops)
    return {
        "name": run.name,
        "micro_batch_size": micro,
        "compile_mode": compile_mode or "eager",
        "step_seconds": dt,
        "tokens_per_sec": tok_s,
        "mfu": flops_per_token(cfg, tc.seq_len)["total"] * tok_s / peak if peak else None,
        "first_step_seconds": first,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "est_run_minutes": tc.total_steps * dt / 60,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--micro-batch", type=int, nargs="+", default=None, help="default: the config's own")
    ap.add_argument("--compile-mode", nargs="+", default=["default"],
                    help="torch.compile modes to try: default, max-autotune-no-cudagraphs, eager")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results/throughput.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_throughput needs a CUDA GPU")

    results = []
    out = REPO / args.out
    if out.exists():
        results = [r for r in json.loads(out.read_text())]
    for path in args.configs:
        run = load_run_config(path)
        micros = args.micro_batch or [run.train.micro_batch_size]
        rows = []
        for micro in micros:
            if run.train.windows_per_step % micro:
                continue
            for mode in args.compile_mode:
                try:
                    r = bench(run, micro, args.steps, args.warmup, "cuda", None if mode == "eager" else mode)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {run.name} micro={micro} {mode}: OOM")
                    torch.cuda.empty_cache()
                    continue
                rows.append(r)
                print(f"  {run.name:<16} micro={micro:<4} {r['compile_mode']:<26} "
                      f"{r['tokens_per_sec'] / 1e3:7.1f}k tok/s  MFU {r['mfu']:.1%}  "
                      f"{r['peak_mem_gb']:5.1f} GB  first step {r['first_step_seconds']:5.1f}s  "
                      f"-> {r['est_run_minutes']:.1f} min/run", flush=True)
                torch._dynamo.reset()
        if rows:
            best = max(rows, key=lambda r: r["tokens_per_sec"])
            for r in rows:
                r["best"] = r is best
            results = [r for r in results if r["name"] != run.name] + rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
