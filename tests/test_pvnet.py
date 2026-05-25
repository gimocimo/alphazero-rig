"""Tests for the policy/value network.

Three groups:
    1. *Shape / range* — the contract callers rely on: output shapes, value
       in [-1, 1], policy logits finite.
    2. *Inference behaviour* — `predict` masks illegal moves, sums to 1.
    3. *Overfit-single-batch* — the load-bearing sanity check. If a network
       can't memorise a single tiny batch, training will fail silently
       downstream. Catching it here saves hours.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alphazero.games.connect4 import Connect4
from alphazero.nets.pvnet import PVNet, PVNetConfig, predict


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net() -> PVNet:
    torch.manual_seed(0)
    return PVNet()


def test_forward_shapes(net: PVNet) -> None:
    batch = torch.randn(8, 2, 6, 7)
    logits, value = net(batch)
    assert logits.shape == (8, 7)
    assert value.shape == (8,)


def test_value_in_range(net: PVNet) -> None:
    batch = torch.randn(16, 2, 6, 7)
    _, value = net(batch)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_logits_finite(net: PVNet) -> None:
    batch = torch.randn(4, 2, 6, 7)
    logits, _ = net(batch)
    assert torch.all(torch.isfinite(logits))


def test_param_count_reasonable(net: PVNet) -> None:
    n = net.num_parameters()
    # Sanity: nontrivial but not absurd. Connect4 doesn't need a big net.
    assert 5_000 < n < 200_000, f"unexpected param count {n}"


def test_predict_masks_illegal_moves(net: PVNet, game: Connect4) -> None:
    # Fill column 0 so it becomes illegal.
    state = game.initial_state()
    for _ in range(6):
        state = game.step(state, 0)
    # Column 0 is now full → illegal. All other columns legal.
    probs, value = predict(net, game, state)
    assert probs.shape == (7,)
    assert probs[0] == 0.0, "illegal column 0 still has probability"
    assert np.isclose(probs.sum(), 1.0), "probs do not sum to 1"
    assert -1.0 <= value <= 1.0


def test_predict_initial_state_uniform_ish(net: PVNet, game: Connect4) -> None:
    """With a random-init net all columns are legal, so probs should be a
    valid distribution over all 7 columns (each strictly positive)."""
    probs, _ = predict(net, game, game.initial_state())
    assert np.all(probs > 0)
    assert np.isclose(probs.sum(), 1.0)


def test_can_overfit_single_batch(net: PVNet) -> None:
    """The load-bearing test: if the net can't drive the loss to near zero
    on a single 8-example batch, something is broken — bad init, wrong loss
    shape, BatchNorm misconfigured, etc. Catching this here is much cheaper
    than discovering it once the full training loop is wired up.

    One-hot policy targets and clear value targets give an achievable floor
    of zero loss (vs. soft targets, where the floor is the target's entropy).
    """
    torch.manual_seed(42)
    x = torch.randn(8, 2, 6, 7)
    target_actions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    target_value = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.5])

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    net.train()

    for _ in range(500):
        optimizer.zero_grad()
        logits, value = net(x)
        loss_policy = F.cross_entropy(logits, target_actions)
        loss_value = F.mse_loss(value, target_value)
        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()

    final = loss.item()
    assert final < 0.05, (
        f"loss did not collapse to ~0 in 500 steps (got {final:.4f}) — "
        "the net is failing to overfit a tiny batch"
    )
