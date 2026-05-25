"""Tests for the pure MCTS agent.

Three flavours of test:
    1. *Mechanical* — does the agent return legal moves, refuse terminal states.
    2. *Tactical* — given a forced one-move win or threat, does it pick the
       right move? These are the cheapest meaningful check that the algorithm
       isn't broken.
    3. *Statistical* — across many games, does it crush a uniform-random
       opponent? This is the floor any future learned agent must beat.
"""

from __future__ import annotations

import pytest

from alphazero.agents.random_agent import RandomAgent
from alphazero.games.connect4 import Connect4
from alphazero.mcts.pure_mcts import MCTSAgent
from alphazero.training.self_play import play_game


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def _play(game: Connect4, actions: list[int]):
    s = game.initial_state()
    for a in actions:
        s = game.step(s, a)
    return s


def test_returns_legal_action(game: Connect4) -> None:
    state = game.initial_state()
    agent = MCTSAgent(simulations=20, seed=0)
    action = agent.select_action(game, state)
    assert game.legal_actions(state)[action]


def test_raises_on_terminal_state(game: Connect4) -> None:
    # Force a quick X vertical win in col 0.
    state = _play(game, [0, 1, 0, 1, 0, 1, 0])
    assert game.is_terminal(state)
    agent = MCTSAgent(simulations=10, seed=0)
    with pytest.raises(RuntimeError, match="terminal"):
        agent.select_action(game, state)


def test_finds_immediate_winning_move(game: Connect4) -> None:
    """X has three pieces stacked in col 0 with no opposing threat — must drop
    the fourth into col 0. This is the simplest tactical situation: search
    only needs to look one ply ahead, and rollouts from that ply give a 100%
    win signal.
    """
    state = _play(game, [0, 6, 0, 5, 0, 4])
    # Position: X at (5,0),(4,0),(3,0); O at (5,6),(5,5),(5,4). X to move.
    agent = MCTSAgent(simulations=200, seed=0)
    assert agent.select_action(game, state) == 0


def test_blocks_opponent_immediate_win(game: Connect4) -> None:
    """O has three vertical in col 0, X has no winning move of its own.
    Any non-col-0 move from X loses immediately to O playing col 0 next.
    Search must reliably surface col 0 as the only saving move.
    """
    state = _play(game, [2, 0, 4, 0, 6, 0])
    # X at (5,2),(5,4),(5,6) — spread and non-threatening.
    # O at (5,0),(4,0),(3,0) — three vertical, one move from winning.
    agent = MCTSAgent(simulations=400, seed=0)
    assert agent.select_action(game, state) == 0


def test_mcts_dominates_random(game: Connect4) -> None:
    """Across 8 games (4 as X, 4 as O), MCTS with 40 simulations per move
    should win nearly all. Random has no defensive capability at all.
    """
    n_per_side = 4
    wins = 0
    for i in range(n_per_side):
        record = play_game(
            game,
            MCTSAgent(simulations=40, seed=i),
            RandomAgent(seed=100 + i),
        )
        if record.outcome == 1:
            wins += 1
    for i in range(n_per_side):
        record = play_game(
            game,
            RandomAgent(seed=200 + i),
            MCTSAgent(simulations=40, seed=300 + i),
        )
        if record.outcome == -1:
            wins += 1
    total = 2 * n_per_side
    assert wins >= int(0.75 * total), f"MCTS only won {wins}/{total} vs random"
