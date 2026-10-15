import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from compare import best_lr  # noqa: E402


def row(lr, seed, loss, diverged=False, out_dir="runs", total_tokens=212336640):
    return {"train.lr": str(lr), "train.optimizer": "adamw", "train.seed": str(seed),
            "val_iid_loss": str(loss), "diverged": str(diverged), "train.out_dir": out_dir,
            "train.total_tokens": str(total_tokens)}


def test_best_lr_follows_the_protocol_not_the_lucky_single_run():
    rows = [row(0.004, 0, 3.50), row(0.004, 1, 3.53), row(0.004, 2, 3.52),
            row(0.008, 0, 3.51),  # loses to 0.004 at seed 0, but beats 0.004's 3-seed mean (3.517)
            row(0.002, 0, 3.60)]
    lr, vals = best_lr(rows, "val_iid_loss")
    assert lr == "0.004" and sorted(vals) == [3.50, 3.52, 3.53]
    _, fresh = best_lr(rows, "val_iid_loss", fresh=True)
    assert sorted(fresh) == [3.52, 3.53]


def test_diverged_runs_never_win():
    rows = [row(0.016, 0, 1.0, diverged=True), row(0.004, 0, 3.5)]
    assert best_lr(rows, "val_iid_loss")[0] == "0.004"


def test_sigma_kernel_and_repeats_stay_out_of_the_variant_table(tmp_path, capsys, monkeypatch):
    import csv

    import compare

    rows = [dict(row(0.004, s, v), name="baseline") for s, v in enumerate([3.50, 3.52, 3.51, 3.53, 3.49])]
    rows += [dict(row(0.004, 0, v, out_dir=f"runs/repeats/r{i}"), name="baseline_repeat")
             for i, v in enumerate((3.505, 3.495))]
    rows += [dict(row(0.004, s, v), name="pos_rope") for s, v in enumerate([3.40, 3.41, 3.42])]
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr("sys.argv", ["compare.py", "--csv", str(path)])
    compare.main()
    out = capsys.readouterr().out
    assert "sigma_kernel (seed 0 repeated, n=3)" in out
    assert "pos_rope" in out and "better" in out
    assert "baseline_repeat " not in out.split("verdict")[1]


def test_resume_check_verdict():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    from check_resume import identical_prefix, mean_abs_diff, verdict

    a = [5.0, 4.0, 3.0, 2.9, 2.8]
    assert identical_prefix(a, [5.0, 4.0, 3.1]) == 2
    assert mean_abs_diff(a, [5.0, 4.0, 3.0, 3.0, 2.9], 3) == pytest.approx(0.1)
    assert verdict(ab=0.0008, ac=0.0006)  # resumed run inside the kernel-noise envelope
    assert not verdict(ab=0.050, ac=0.0006)  # optimizer state lost on resume


def test_special_purpose_runs_never_enter_the_comparison(tmp_path, capsys, monkeypatch):
    # Regression: short resume-check runs share the name "baseline" and the
    # default LR; one of them overwrote the real seed-0 result at that LR, so the protocol script picked
    # the wrong best LR.
    import csv

    import compare

    rows = [dict(row(lr, 0, v), name="baseline") for lr, v in [(0.002, 3.633), (0.004, 3.591), (0.008, 3.601)]]
    rows += [dict(row(0.004, 0, 5.93, out_dir=f"runs/resume_check/{x}", total_tokens=26214400), name="baseline")
             for x in "ab"]
    # a run moved into a subdirectory after the fact is excluded by where it lives
    rows += [dict(row(0.004, 0, 5.93, total_tokens=26214400), name="baseline", run_dir="local/baseline__x")]
    for r in rows:  # where collect.py would find each run
        sub = r["train.out_dir"].removeprefix("runs").strip("/")
        r.setdefault("run_dir", f"{sub}/baseline__y" if sub else "baseline__y")
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr("sys.argv", ["compare.py", "--csv", str(path), "--best-lr", "baseline"])
    compare.main()
    assert capsys.readouterr().out.strip() == "0.004"

    mixed = rows[:3] + [dict(row(0.004, 1, 3.6, total_tokens=26214400), name="baseline")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mixed[0]))
        w.writeheader()
        w.writerows(mixed)
    with pytest.raises(SystemExit, match="different configs"):
        compare.main()


def test_pooled_sigma_and_threshold_sensitive_verdicts(tmp_path, capsys, monkeypatch):
    import csv

    import compare

    def runs(name, vals):
        return [dict(row(0.004, s, v), name=name, run_dir=f"{name}__{s}") for s, v in enumerate(vals)]

    # baseline sigma 0.0036-ish; noisy variants inflate the pooled sigma to ~0.009
    rows = runs("baseline", [3.5909, 3.5883, 3.5964, 3.5915, 3.5870])
    rows += runs("noisy_a", [3.570, 3.585, 3.592]) + runs("noisy_b", [3.600, 3.585, 3.580])
    rows += runs("clear_win", [3.560, 3.561, 3.559])   # delta -0.031: clears both thresholds
    rows += runs("borderline", [3.581, 3.582, 3.580])  # delta -0.0098: clears 2*0.0036 only
    path = tmp_path / "runs.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr("sys.argv", ["compare.py", "--csv", str(path)])
    compare.main()
    out = capsys.readouterr().out
    assert "sigma_pooled" in out and "(12 df)" in out
    lines = {l.split()[0]: l for l in out.splitlines() if l and l.split()[0] in ("clear_win", "borderline")}
    assert lines["clear_win"].rstrip().endswith("better")
    assert lines["borderline"].rstrip().endswith("THRESHOLD-SENSITIVE")
