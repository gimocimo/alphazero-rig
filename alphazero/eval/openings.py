"""Opening discovery timeline.

Given a directory of dated checkpoints (e.g. `iter_0010.pt`,
`iter_0020.pt`, …, saved by the training loop when `snapshot_every > 0`),
load each net, query its policy on the empty board, and render the
resulting first-move distributions as a heatmap over time.

What you should expect to see:
    * Early iterations: nearly uniform across columns (random-init priors).
    * Mid training: probability mass starts concentrating on the middle
      column — the centre is dominant in Connect4 because it touches
      more potential 4-in-a-rows than any other column.
    * Late training: heavy mass on column 3, very little elsewhere — the
      agent has discovered the winning first move.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..games.base import Game
from ..nets.pvnet import predict
from ..training.trainer import Trainer

S = TypeVar("S")

_ITER_RE = re.compile(r"iter_(\d+)\.pt$")


def find_snapshots(run_dir: str | Path) -> list[tuple[int, Path]]:
    """Return [(iteration, path), …] sorted by iteration ascending."""
    run_dir = Path(run_dir)
    snapshots: list[tuple[int, Path]] = []
    for p in run_dir.glob("iter_*.pt"):
        m = _ITER_RE.search(p.name)
        if m:
            snapshots.append((int(m.group(1)), p))
    snapshots.sort(key=lambda x: x[0])
    return snapshots


def collect_opening_priors(
    snapshots: list[tuple[int, Path]],
    game: Game[S],
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, list[int]]:
    """Load each snapshot, query the policy on `game.initial_state()`,
    return (priors_matrix, iter_labels) where priors_matrix has shape
    (num_snapshots, action_size)."""
    if not snapshots:
        raise ValueError("no snapshots provided")
    state = game.initial_state()
    priors: list[np.ndarray] = []
    iters: list[int] = []
    for it, path in snapshots:
        trainer = Trainer.from_checkpoint(path)
        net = trainer.net.to(device)
        probs, _ = predict(net, game, state, device=device)
        priors.append(probs.astype(np.float32))
        iters.append(it)
    return np.stack(priors), iters


def plot_opening_timeline(
    priors: np.ndarray,
    iter_labels: list[int],
    output_path: str | Path,
) -> Path:
    """Render the opening-priors-over-time matrix as a heatmap.

    Rows = checkpoints (chronological top → bottom).
    Cols = action indices.
    """
    output_path = Path(output_path)
    n_snaps, n_actions = priors.shape

    # Height scales with number of snapshots, but bounded.
    fig_h = min(10, max(4, 0.4 * n_snaps + 2))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(
        priors,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.5, float(priors.max())),
        interpolation="nearest",
    )

    ax.set_xlabel("action (column)")
    ax.set_xticks(range(n_actions))
    ax.set_yticks(range(n_snaps))
    ax.set_yticklabels([f"iter {it:>4d}" for it in iter_labels])
    ax.set_title("Opening prior — P(first move) per training snapshot")
    fig.colorbar(im, ax=ax, label="probability")

    # Annotate each cell with the rounded percentage for readability when
    # the snapshot count is small.
    if n_snaps <= 20:
        for i in range(n_snaps):
            for j in range(n_actions):
                p = priors[i, j]
                if p < 0.005:
                    continue
                colour = "white" if p > 0.4 else "black"
                ax.text(
                    j, i, f"{int(round(p * 100))}",
                    ha="center", va="center", fontsize=8, color=colour,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def plot_openings_for_run(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> Path:
    """End-to-end: discover snapshots, load priors, write the heatmap.

    Raises if no `iter_*.pt` snapshots are found in `run_dir`.
    """
    from ..games.connect4 import Connect4

    run_dir = Path(run_dir)
    snapshots = find_snapshots(run_dir)
    if not snapshots:
        raise FileNotFoundError(
            f"no iter_*.pt checkpoints in {run_dir} — re-run training with --snapshot-every >= 1"
        )
    output_path = Path(output_path) if output_path else run_dir / "openings.png"
    game = Connect4()
    priors, iters = collect_opening_priors(snapshots, game, device=device)
    return plot_opening_timeline(priors, iters, output_path)
