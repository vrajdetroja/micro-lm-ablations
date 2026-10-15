import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from compare import best_lr  # noqa: E402


def row(lr, seed, loss, diverged=False):
    return {"train.lr": str(lr), "train.optimizer": "adamw", "train.seed": str(seed),
            "val_iid_loss": str(loss), "diverged": str(diverged)}


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
    rows += [dict(row(0.004, 0, v), name="baseline_repeat") for v in (3.505, 3.495)]
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
