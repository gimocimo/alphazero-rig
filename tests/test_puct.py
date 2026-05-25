"""Tests for the PUCT agent (network-guided MCTS).

PUCT with a *random-init* network should still play sensibly — the search
compensates for noisy priors. We test the same mechanical + tactical
contracts as pure MCTS, plus a statistical "beats RandomAgent" check.
"""

from __future__ import annotations

import pytest
import torch

from alphazero.agents.random_agent import RandomAgent
from alphazero.games.connect4 import Connect4
from alphazero.mcts.puct import PUCTAgent
from alphazero.nets.pvnet import PVNet
from alphazero.training.self_play import play_game


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net() -> PVNet:
    torch.manual_seed(0)
    return PVNet()


def _play(game: Connect4, actions: list[int]):
    s = game.initial_state()
    for a in actions:
        s = game.step(s, a)
    return s


def test_returns_legal_action(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=20, seed=0)
    action = agent.select_action(game, game.initial_state())
    assert game.legal_actions(game.initial_state())[action]


def test_raises_on_terminal_state(net: PVNet, game: Connect4) -> None:
    state = _play(game, [0, 1, 0, 1, 0, 1, 0])  # X vertical-wins in col 0
    assert game.is_terminal(state)
    agent = PUCTAgent(net, simulations=10, seed=0)
    with pytest.raises(RuntimeError, match="terminal"):
        agent.select_action(game, state)


def test_finds_immediate_winning_move(net: PVNet, game: Connect4) -> None:
    """Same tactical setup as pure-MCTS: X has 3-in-col-0, must play col 0.
    Even a random-init net's noisy priors shouldn't drown out a rollout
    signal this strong (all sims from col 0 hit the terminal value +1)."""
    state = _play(game, [0, 6, 0, 5, 0, 4])
    agent = PUCTAgent(net, simulations=200, seed=0)
    assert agent.select_action(game, state) == 0


def test_blocks_opponent_immediate_win(net: PVNet, game: Connect4) -> None:
    """O has 3-in-col-0, X must block. Search must surface col 0 even
    against the prior of a random net that may favor other columns."""
    state = _play(game, [2, 0, 4, 0, 6, 0])
    agent = PUCTAgent(net, simulations=400, seed=0)
    assert agent.select_action(game, state) == 0


def test_puct_with_random_net_beats_random(net: PVNet, game: Connect4) -> None:
    """Mirrors the pure-MCTS statistical test. With an untrained network
    the priors are roughly uniform and the search budget is doing the heavy
    lifting — should still dominate RandomAgent.
    """
    n_per_side = 4
    wins = 0
    for i in range(n_per_side):
        record = play_game(
            game,
            PUCTAgent(net, simulations=40, seed=i),
            RandomAgent(seed=100 + i),
        )
        if record.outcome == 1:
            wins += 1
    for i in range(n_per_side):
        record = play_game(
            game,
            RandomAgent(seed=200 + i),
            PUCTAgent(net, simulations=40, seed=300 + i),
        )
        if record.outcome == -1:
            wins += 1
    total = 2 * n_per_side
    assert wins >= int(0.75 * total), f"PUCT only won {wins}/{total} vs random"
