"""Tests for PUCT-driven self-play data extraction + the new PUCT methods.

Covers:
    * select_action_with_visits returns a valid policy target (sums to 1).
    * argmax of visits matches `select_action` (T=0).
    * Dirichlet noise actually perturbs the root priors.
    * play_self_play_game returns one example per ply, with value targets
      consistent with the eventual outcome.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.games.connect4 import Connect4
from alphazero.mcts.puct import PUCTAgent
from alphazero.nets.pvnet import PVNet
from alphazero.training.replay_buffer import TrainingExample
from alphazero.training.self_play import play_self_play_game


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net() -> PVNet:
    torch.manual_seed(0)
    return PVNet()


def test_select_action_with_visits_returns_valid_policy(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=30, seed=0)
    action, policy = agent.select_action_with_visits(
        game, game.initial_state(), temperature=0.0
    )
    assert 0 <= action < game.action_size
    assert policy.shape == (game.action_size,)
    assert policy.dtype == np.float32
    assert np.isclose(policy.sum(), 1.0)
    assert (policy >= 0).all()


def test_argmax_visits_matches_select_action(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=30, seed=0)
    state = game.initial_state()
    action_a = agent.select_action(game, state)

    agent_b = PUCTAgent(net, simulations=30, seed=0)
    action_b, policy_b = agent_b.select_action_with_visits(game, state, temperature=0.0)
    assert action_a == action_b
    assert action_b == int(np.argmax(policy_b))


def test_temperature_one_samples_stochastically(net: PVNet, game: Connect4) -> None:
    """With T=1 and multiple runs we should see *some* action variety,
    unlike T=0 which is deterministic."""
    state = game.initial_state()
    actions = set()
    for seed in range(20):
        agent = PUCTAgent(net, simulations=30, seed=seed)
        action, _ = agent.select_action_with_visits(game, state, temperature=1.0)
        actions.add(action)
    assert len(actions) > 1, "T=1 produced only one unique action across seeds"


def test_dirichlet_noise_changes_root_priors(net: PVNet, game: Connect4) -> None:
    """A run with ε>0 should give a different policy target than ε=0
    (with the same seed and state)."""
    state = game.initial_state()
    clean = PUCTAgent(net, simulations=30, dirichlet_epsilon=0.0, seed=0)
    noisy = PUCTAgent(
        net, simulations=30, dirichlet_alpha=0.5, dirichlet_epsilon=0.25, seed=0
    )
    _, policy_clean = clean.select_action_with_visits(game, state, temperature=0.0)
    _, policy_noisy = noisy.select_action_with_visits(game, state, temperature=0.0)
    assert not np.allclose(policy_clean, policy_noisy)


def test_play_self_play_game_returns_examples(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=20, dirichlet_epsilon=0.25, seed=0)
    examples = play_self_play_game(game, agent, temperature_threshold=4)
    assert len(examples) > 0
    for ex in examples:
        assert isinstance(ex, TrainingExample)
        assert ex.encoded_state.shape == (2, 6, 7)
        assert ex.policy_target.shape == (7,)
        assert np.isclose(ex.policy_target.sum(), 1.0)
        assert ex.value_target in (-1.0, 0.0, 1.0)


def test_self_play_value_targets_match_outcome(net: PVNet, game: Connect4) -> None:
    """The value target at ply k must be the eventual outcome from the
    perspective of whoever moved at ply k. We can verify this by checking
    that consecutive examples have flipped signs (since players alternate),
    *unless* the outcome was a draw."""
    agent = PUCTAgent(net, simulations=20, dirichlet_epsilon=0.25, seed=0)
    examples = play_self_play_game(game, agent, temperature_threshold=4)
    # The first example's player is +1 (X always starts); its target sign is
    # the absolute outcome.
    if examples[0].value_target == 0.0:
        # Draw: all targets must be zero.
        for ex in examples:
            assert ex.value_target == 0.0
    else:
        signs = [ex.value_target for ex in examples]
        # Adjacent value targets must flip sign (zero-sum, players alternate).
        for a, b in zip(signs[:-1], signs[1:]):
            assert a == -b, f"value targets {a}, {b} do not alternate"
