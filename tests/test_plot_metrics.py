"""Tests for the metrics plotting module.

These are smoke tests: we don't verify pixel-level content, only that
the plotters consume a realistic metrics.csv and emit non-trivial PNGs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from alphazero.eval.plot_metrics import load_metrics, plot_all


_FIELDS = [
    "iteration",
    "timestamp",
    "elapsed_seconds",
    "buffer_size",
    "train_steps_total",
    "mean_loss",
    "mean_loss_policy",
    "mean_loss_value",
    "self_play_seconds",
    "train_seconds",
    "eval_winrate",
    "promoted",
]


def _write_fake_metrics(path: Path, n_iters: int = 12) -> None:
    """Write a metrics.csv that resembles what the loop produces."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for i in range(1, n_iters + 1):
            # Loss decays roughly, value loss collapses faster than policy.
            total = 2.3 - 0.1 * i + 0.05 * (i % 3)
            policy = 1.9 - 0.05 * i
            value = max(0.05, 0.4 - 0.03 * i)

            do_eval = i % 3 == 0
            winrate = 0.4 + 0.06 * (i / n_iters) if do_eval else None
            promoted = 1 if (do_eval and winrate is not None and winrate > 0.55) else 0

            writer.writerow({
                "iteration": i,
                "timestamp": f"2026-05-25 14:0{i}:00",
                "elapsed_seconds": f"{i * 60:.1f}",
                "buffer_size": 1000 + i * 200,
                "train_steps_total": i * 200,
                "mean_loss": f"{total:.4f}",
                "mean_loss_policy": f"{policy:.4f}",
                "mean_loss_value": f"{value:.4f}",
                "self_play_seconds": "60.0",
                "train_seconds": "5.0",
                "eval_winrate": f"{winrate:.3f}" if winrate is not None else "",
                "promoted": promoted,
            })


def test_load_metrics_parses_blanks_as_nan(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_fake_metrics(csv_path, n_iters=6)
    metrics = load_metrics(csv_path)
    assert metrics["iteration"].shape == (6,)
    # iters 1, 2, 4, 5 had no eval → NaN.
    has_eval = ~np.isnan(metrics["eval_winrate"])
    assert has_eval.tolist() == [False, False, True, False, False, True]


def test_plot_all_writes_nontrivial_pngs(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_fake_metrics(csv_path, n_iters=12)

    out = plot_all(tmp_path)
    loss_png = out["loss"]
    winrate_png = out["winrate"]
    assert loss_png.exists()
    assert winrate_png.exists()
    # PNG headers are 8 bytes; a real plot will be many KB. Use 1KB as the
    # "actually wrote something" threshold.
    assert loss_png.stat().st_size > 1024
    assert winrate_png.stat().st_size > 1024


def test_plot_all_handles_no_evaluations(tmp_path: Path) -> None:
    """Early in training there may be zero eval rows. The plotter should
    still produce a winrate.png (with a 'no evaluations yet' annotation)."""
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for i in (1, 2):
            writer.writerow({
                "iteration": i,
                "timestamp": "—",
                "elapsed_seconds": "0",
                "buffer_size": 100,
                "train_steps_total": 0,
                "mean_loss": "2.0",
                "mean_loss_policy": "1.8",
                "mean_loss_value": "0.2",
                "self_play_seconds": "0",
                "train_seconds": "0",
                "eval_winrate": "",  # never evaluated
                "promoted": 0,
            })
    out = plot_all(tmp_path)
    assert out["winrate"].exists()
    assert out["winrate"].stat().st_size > 1024
