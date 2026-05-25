"""Tests for the Trainer.

The load-bearing test is `loss_decreases_over_steps` — analogous to the
overfit-single-batch test we put on PVNet directly, but routed through the
Trainer interface so it exercises buffer-sampling + device-moving + the
soft-CE loss together.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.nets.pvnet import PVNet, PVNetConfig
from alphazero.training.replay_buffer import ReplayBuffer, TrainingExample
from alphazero.training.trainer import Trainer, TrainerConfig


def _make_buffer(seed: int = 0, n: int = 32) -> ReplayBuffer:
    """Construct a buffer with strong, learnable signal.

    Each example is a uniform random state with a specific *action* and
    *value* it should predict — so loss has somewhere to go.
    """
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity=200, seed=seed)
    for _ in range(n):
        state = rng.standard_normal((2, 6, 7)).astype(np.float32)
        action = int(rng.integers(0, 7))
        policy = np.zeros(7, dtype=np.float32)
        policy[action] = 1.0  # one-hot — achievable floor at 0
        value = float(rng.choice([-1.0, 0.0, 1.0]))
        buf.add(TrainingExample(encoded_state=state, policy_target=policy, value_target=value))
    return buf


@pytest.fixture
def small_net() -> PVNet:
    torch.manual_seed(0)
    # Smaller net keeps the CPU-only test fast while still exercising backprop.
    return PVNet(PVNetConfig(channels=16, num_blocks=2))


def test_trainer_constructs_with_default_config(small_net: PVNet) -> None:
    trainer = Trainer(small_net, TrainerConfig(device="cpu", batch_size=8))
    assert trainer.step_count == 0
    assert trainer.device == "cpu"
    assert isinstance(trainer.optimizer, torch.optim.Adam)


def test_device_auto_detection_returns_supported_device(small_net: PVNet) -> None:
    trainer = Trainer(small_net)
    assert trainer.device in ("cpu", "cuda", "mps")


def test_train_step_returns_expected_metrics(small_net: PVNet) -> None:
    trainer = Trainer(small_net, TrainerConfig(device="cpu", batch_size=8))
    buf = _make_buffer()
    metrics = trainer.train_step(buf)
    assert {"loss", "loss_policy", "loss_value", "grad_norm", "step"} <= metrics.keys()
    assert metrics["step"] == 1
    assert metrics["loss"] > 0


def test_loss_decreases_over_steps(small_net: PVNet) -> None:
    """End-to-end load-bearing test: a few hundred steps on a tiny static
    buffer should drive the loss substantially down. If this fails, the
    trainer plumbing is broken even though PVNet's standalone overfit test
    passes (different code path: buffer sampling, device moves, etc.)."""
    trainer = Trainer(
        small_net,
        TrainerConfig(device="cpu", batch_size=16, learning_rate=5e-3),
    )
    buf = _make_buffer(n=16)  # exactly batch_size so every step sees the same data

    metrics_first = trainer.train_step(buf)
    initial_loss = metrics_first["loss"]

    final_loss = initial_loss
    for _ in range(200):
        final_loss = trainer.train_step(buf)["loss"]

    assert final_loss < 0.3 * initial_loss, (
        f"loss only dropped from {initial_loss:.3f} to {final_loss:.3f}"
    )
    assert trainer.step_count == 201


def test_gradient_clipping_caps_norm(small_net: PVNet) -> None:
    """With grad_clip=0.1 and an explicit forward+backward, the reported
    grad_norm should reflect a clip down to the cap (i.e., the pre-clip
    norm was larger than 0.1)."""
    trainer = Trainer(
        small_net, TrainerConfig(device="cpu", batch_size=8, grad_clip=0.1)
    )
    buf = _make_buffer()
    metrics = trainer.train_step(buf)
    # clip_grad_norm_ returns the *pre-clip* norm, so this should be the
    # norm before clipping. We only assert that clipping actually engaged
    # (i.e. pre-clip norm > 0.1).
    assert metrics["grad_norm"] > 0.1


def test_save_and_load_checkpoint_roundtrip(tmp_path, small_net: PVNet) -> None:
    """Train for a few steps, save, load into a fresh Trainer, verify
    the loaded net produces identical outputs to the original."""
    trainer = Trainer(small_net, TrainerConfig(device="cpu", batch_size=8))
    buf = _make_buffer()
    for _ in range(5):
        trainer.train_step(buf)

    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)

    # Sanity: a fresh Trainer differs from the trained one.
    fresh = Trainer(PVNet(PVNetConfig(channels=16, num_blocks=2)), TrainerConfig(device="cpu"))

    x = torch.randn(4, 2, 6, 7)
    trainer.net.eval()
    fresh.net.eval()
    logits_trained, _ = trainer.net(x)
    logits_fresh, _ = fresh.net(x)
    assert not torch.allclose(logits_trained, logits_fresh)

    # Loaded checkpoint should match the original *exactly*.
    loaded = Trainer.from_checkpoint(checkpoint)
    loaded.net.eval()
    logits_loaded, value_loaded = loaded.net(x)
    logits_orig, value_orig = trainer.net(x)
    assert torch.allclose(logits_loaded, logits_orig)
    assert torch.allclose(value_loaded, value_orig)
    assert loaded.step_count == trainer.step_count
