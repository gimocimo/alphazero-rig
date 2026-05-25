"""Tests for the game-replay text renderer."""

from __future__ import annotations

import pytest
import torch

from alphazero.agents.random_agent import RandomAgent
from alphazero.eval.replay import render_game_replay
from alphazero.games.connect4 import Connect4
from alphazero.mcts.puct import PUCTAgent
from alphazero.nets.pvnet import PVNet


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def test_replay_random_vs_random_terminates(game: Connect4) -> None:
    out = render_game_replay(game, RandomAgent(seed=0), RandomAgent(seed=1), show_tree=False)
    assert "Initial position" in out
    assert "Result:" in out
    assert "Ply 1:" in out
    # Must end with one of three outcomes.
    assert any(r in out for r in ("X wins", "O wins", "draw"))


def test_replay_with_puct_shows_search_trees(game: Connect4) -> None:
    torch.manual_seed(0)
    net = PVNet()
    agent = PUCTAgent(net, simulations=20, seed=0)
    out = render_game_replay(game, agent, agent, top_k_per_ply=3, max_plies=4)
    # Tree rendering should print "Root:" once per ply.
    assert out.count("Root:") >= 2
    assert "puct(sims=20)" in out


def test_replay_truncates_at_max_plies(game: Connect4) -> None:
    out = render_game_replay(
        game, RandomAgent(seed=0), RandomAgent(seed=1),
        show_tree=False, max_plies=3,
    )
    # Three plies → at least Ply 1, 2, 3 appear; Ply 4 must not.
    assert "Ply 3:" in out
    assert "Ply 4:" not in out
