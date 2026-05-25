"""Value head calibration: does the net know when it's winning?

For each position visited during a series of games, we record:
    predicted  = net.value(state)      — from the to-move player's view
    actual     = outcome * to_move     — +1 if that player went on to win,
                                          -1 if lost, 0 if drew

Buckets of predicted values then get plotted against the mean actual
outcome in each bucket. A perfectly calibrated value head sits on the
y = x diagonal. A random-init net's curve will be roughly flat near zero
(predictions are noise; outcomes are real). A well-trained net's curve
sharpens onto the diagonal.

This is a much more *trustworthy* training-progress signal than loss
because it is interpretable: a positive slope means the net's confidence
correlates with actual results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..agents.base import Agent
from ..games.base import Game
from ..nets.pvnet import PVNet, predict
from ..training.self_play import play_game

S = TypeVar("S")


def collect_calibration_data(
    game: Game[S],
    agent_a: Agent,
    agent_b: Agent,
    net: PVNet,
    num_games: int,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Play `num_games` and gather (predicted_value, actual_outcome) per ply.

    The two arrays returned are aligned: predictions[i] is the net's value
    estimate for the to-move player at position i; outcomes[i] is the
    eventual game result from that same player's perspective.

    Returns (predictions, outcomes), each shape (total_positions,) float32.
    """
    if num_games < 1:
        raise ValueError("num_games must be >= 1")

    predictions: list[float] = []
    outcomes: list[float] = []

    for _ in range(num_games):
        record = play_game(game, agent_a, agent_b)
        if record.outcome is None:
            continue
        for ply, state in enumerate(record.states):
            _, v_pred = predict(net, game, state, device=device)
            predictions.append(v_pred)
            outcomes.append(float(record.outcome * record.players[ply]))

    return (
        np.array(predictions, dtype=np.float32),
        np.array(outcomes, dtype=np.float32),
    )


def plot_calibration(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    output_path: str | Path,
    num_bins: int = 10,
    title_suffix: str = "",
) -> Path:
    """Bucket predictions into `num_bins` equal-width bins on [-1, 1],
    plot mean outcome per bucket vs the bucket centre, and overlay y = x.

    A `n=` annotation above each point reports the sample count, so you
    can spot buckets whose value is noisy due to small sample size.
    """
    output_path = Path(output_path)
    if predictions.size == 0:
        raise ValueError("predictions array is empty — nothing to plot")
    if predictions.shape != outcomes.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape}, outcomes {outcomes.shape}"
        )

    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    bin_idx = np.clip(np.digitize(predictions, bins) - 1, 0, num_bins - 1)

    bucket_means = np.full(num_bins, np.nan)
    bucket_counts = np.zeros(num_bins, dtype=int)
    for b in range(num_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bucket_means[b] = outcomes[mask].mean()
            bucket_counts[b] = int(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="perfect calibration")
    valid = ~np.isnan(bucket_means)
    ax.plot(centers[valid], bucket_means[valid], "o-", color="steelblue", lw=2,
            label="net calibration", markersize=8)
    for c, m, n in zip(centers, bucket_means, bucket_counts):
        if np.isnan(m):
            continue
        ax.annotate(
            f"n={n}", (c, m),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=8, color="gray",
        )

    title = f"Value head calibration ({predictions.size} positions)"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set_xlabel("predicted value (to-move's view)")
    ax.set_ylabel("mean actual outcome (to-move's view)")
    ax.set_title(title)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path
