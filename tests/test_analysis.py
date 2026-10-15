import sys
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
