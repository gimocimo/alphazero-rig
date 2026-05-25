"""Tests for the opening-discovery timeline.

We don't have a real multi-snapshot training run to test against, so we
construct fake snapshots by saving fresh-init PVNets with different seeds.
That gives us distinct policies per snapshot — enough to exercise the
loading + plotting pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alphazero.eval.openings import (
    collect_opening_priors,
    find_snapshots,
    plot_opening_timeline,
    plot_openings_for_run,
)
from alphazero.games.connect4 import Connect4
from alphazero.nets.pvnet import PVNet, PVNetConfig
from alphazero.training.trainer import Trainer, TrainerConfig


def _save_fake_snapshot(out_dir: Path, iteration: int, seed: int) -> Path:
    """Create a tiny fresh PVNet with a fixed seed and save it as iter_NNNN.pt."""
    torch.manual_seed(seed)
    net = PVNet(PVNetConfig(channels=8, num_blocks=1))
    trainer = Trainer(net, TrainerConfig(device="cpu"))
    path = out_dir / f"iter_{iteration:04d}.pt"
    trainer.save_checkpoint(path)
    return path


def test_find_snapshots_orders_by_iteration(tmp_path: Path) -> None:
    _save_fake_snapshot(tmp_path, 30, seed=2)
    _save_fake_snapshot(tmp_path, 10, seed=0)
    _save_fake_snapshot(tmp_path, 20, seed=1)
    # Also drop an unrelated .pt that should be ignored.
    (tmp_path / "latest.pt").write_bytes(b"")
    found = find_snapshots(tmp_path)
    iters = [it for it, _ in found]
    assert iters == [10, 20, 30]


def test_collect_priors_shape_and_normalisation(tmp_path: Path) -> None:
    snapshots = [
        (it, _save_fake_snapshot(tmp_path, it, seed=it))
        for it in (10, 20, 30)
    ]
    game = Connect4()
    priors, iters = collect_opening_priors(snapshots, game, device="cpu")
    assert priors.shape == (3, game.action_size)
    assert iters == [10, 20, 30]
    # Each row is a valid probability distribution.
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)
    assert (priors >= 0).all()


def test_plot_timeline_writes_png(tmp_path: Path) -> None:
    priors = np.random.default_rng(0).dirichlet(np.ones(7), size=5).astype(np.float32)
    iters = [10, 20, 30, 40, 50]
    out = plot_opening_timeline(priors, iters, tmp_path / "openings.png")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_plot_openings_for_run_end_to_end(tmp_path: Path) -> None:
    for it in (10, 20):
        _save_fake_snapshot(tmp_path, it, seed=it)
    path = plot_openings_for_run(tmp_path, device="cpu")
    assert path.exists()
    assert path.stat().st_size > 1024


def test_plot_openings_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="snapshot"):
        plot_openings_for_run(tmp_path, device="cpu")
