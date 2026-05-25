"""Plot training metrics from a metrics.csv produced by the training loop.

Two views — both designed to be informative even while training is still
running (re-running the CLI updates the PNGs in place):

    * `loss.png`    — total / policy / value losses per iteration. The
                      classic "is anything learning" check.
    * `winrate.png` — evaluation winrate against the current best. Promote
                      events are starred so you can see when the candidate
                      network last took the crown.

The CSV is parsed with stdlib `csv` — no pandas dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

# Non-interactive backend so this works in scripts, SSH sessions, headless CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_FLOAT_FIELDS = (
    "mean_loss",
    "mean_loss_policy",
    "mean_loss_value",
    "self_play_seconds",
    "train_seconds",
    "eval_winrate",
)


def load_metrics(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Load metrics.csv into a dict of numpy arrays.

    Empty/blank fields become NaN; integer columns stay integer.
    """
    csv_path = Path(csv_path)
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    def _floats(key: str) -> np.ndarray:
        return np.array(
            [float(r[key]) if r[key] not in ("", None) else np.nan for r in rows]
        )

    return {
        "iteration": np.array([int(r["iteration"]) for r in rows]),
        "buffer_size": np.array([int(r["buffer_size"]) for r in rows]),
        "train_steps_total": np.array([int(r["train_steps_total"]) for r in rows]),
        "promoted": np.array([int(r["promoted"]) for r in rows]),
        **{key: _floats(key) for key in _FLOAT_FIELDS},
    }


def plot_loss_curves(metrics: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    iters = metrics["iteration"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, metrics["mean_loss"], label="total", lw=2, color="black")
    ax.plot(iters, metrics["mean_loss_policy"], label="policy", lw=1.5, alpha=0.85, color="steelblue")
    ax.plot(iters, metrics["mean_loss_value"], label="value", lw=1.5, alpha=0.85, color="crimson")

    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (mean across iteration's train steps)")
    ax.set_title("Training losses")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def plot_winrate(metrics: dict[str, np.ndarray], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    iters = metrics["iteration"]
    winrate = metrics["eval_winrate"]
    promoted = metrics["promoted"]

    has_eval = ~np.isnan(winrate)
    eval_iters = iters[has_eval]
    eval_wr = winrate[has_eval]

    fig, ax = plt.subplots(figsize=(10, 5))
    if eval_iters.size > 0:
        ax.plot(eval_iters, eval_wr, "o-", color="steelblue", label="winrate vs current best")
    else:
        ax.text(
            0.5, 0.5, "no evaluations yet",
            transform=ax.transAxes, ha="center", va="center", color="gray",
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.axhline(0.55, color="forestgreen", linestyle=":", alpha=0.7, label="promote threshold")

    promo_iters = iters[promoted == 1]
    if promo_iters.size > 0:
        promo_wr = winrate[promoted == 1]
        ax.scatter(promo_iters, promo_wr, color="red", s=140, marker="*", zorder=5, label="promoted")

    ax.set_xlabel("iteration")
    ax.set_ylabel("winrate (draws = ½)")
    ax.set_title("Evaluation winrate vs current best (alternating sides)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def plot_all(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Generate all available plots from a training run. Returns name → path."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(run_dir / "metrics.csv")
    return {
        "loss": plot_loss_curves(metrics, output_dir / "loss.png"),
        "winrate": plot_winrate(metrics, output_dir / "winrate.png"),
    }
