"""Tests for value calibration data collection + plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alphazero.agents.random_agent import RandomAgent
from alphazero.eval.calibrate import collect_calibration_data, plot_calibration
from alphazero.games.connect4 import Connect4
from alphazero.nets.pvnet import PVNet


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net() -> PVNet:
    torch.manual_seed(0)
    return PVNet()


def test_collect_returns_aligned_arrays(net: PVNet, game: Connect4) -> None:
    preds, outs = collect_calibration_data(
        game, RandomAgent(seed=0), RandomAgent(seed=1), net, num_games=3
    )
    assert preds.shape == outs.shape
    assert preds.ndim == 1
    assert preds.size > 0
    # Outcomes are in {-1, 0, +1}.
    assert set(np.unique(outs).tolist()) <= {-1.0, 0.0, 1.0}
    # Predictions are in [-1, 1] (tanh output).
    assert preds.min() >= -1.0 - 1e-5
    assert preds.max() <= 1.0 + 1e-5


def test_plot_writes_nontrivial_png(net: PVNet, game: Connect4, tmp_path: Path) -> None:
    preds, outs = collect_calibration_data(
        game, RandomAgent(seed=0), RandomAgent(seed=1), net, num_games=3
    )
    output = tmp_path / "calib.png"
    path = plot_calibration(preds, outs, output)
    assert path.exists()
    assert path.stat().st_size > 1024


def test_plot_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_calibration(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            tmp_path / "x.png",
        )


def test_collect_rejects_zero_games(net: PVNet, game: Connect4) -> None:
    with pytest.raises(ValueError, match="num_games"):
        collect_calibration_data(
            game, RandomAgent(), RandomAgent(), net, num_games=0
        )
