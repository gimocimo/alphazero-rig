"""Smoke test for the full training loop driver.

Runs the loop end-to-end with minimal parameters. Asserts only that the
plumbing works: metrics.csv exists with the expected schema and one row
per iteration, and `latest.pt` is a loadable checkpoint. No claim is made
about learning quality from such a short run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from alphazero.nets.pvnet import PVNet
from alphazero.training.loop import TrainingLoopConfig, train_loop
from alphazero.training.trainer import Trainer

_EXPECTED_FIELDS = {
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
}


@pytest.mark.slow
def test_training_loop_smoke(tmp_path: Path) -> None:
    """End-to-end: tiny config, verify outputs exist and have right shape."""
    config = TrainingLoopConfig(
        iterations=2,
        games_per_iteration=2,
        train_steps_per_iteration=5,
        self_play_simulations=10,
        eval_every=2,
        eval_games=2,
        eval_simulations=10,
        min_buffer_size=1,
        batch_size=4,
        output_dir=str(tmp_path),
        device="cpu",
        seed=0,
    )
    train_loop(config)

    # metrics.csv exists, has 2 rows, schema matches.
    metrics_path = tmp_path / "metrics.csv"
    assert metrics_path.exists()
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == _EXPECTED_FIELDS
    assert len(rows) == 2
    assert int(rows[0]["iteration"]) == 1
    assert int(rows[1]["iteration"]) == 2

    # Latest checkpoint exists and reloads cleanly.
    latest = tmp_path / "latest.pt"
    assert latest.exists()
    reloaded = Trainer.from_checkpoint(latest)
    assert isinstance(reloaded.net, PVNet)
    assert reloaded.step_count == 2 * config.train_steps_per_iteration
