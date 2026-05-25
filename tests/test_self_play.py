"""Tests for the self-play harness.

These don't test agent strength — only that the harness records valid trajectories
and surfaces sensible outcomes. Statistical tests use enough games that small
fluctuations don't make the suite flaky.
"""

from __future__ import annotations

import pytest

from alphazero.agents.random_agent import RandomAgent
from alphazero.games.connect4 import Connect4
from alphazero.training.self_play import play_game


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def test_random_vs_random_terminates(game: Connect4) -> None:
    record = play_game(game, RandomAgent(seed=0), RandomAgent(seed=1))
    assert record.outcome in (-1, 0, 1)
    assert len(record) > 0
    assert len(record) <= 42  # Connect4 board is 6x7 = 42 cells


def test_trajectory_only_contains_legal_moves(game: Connect4) -> None:
    record = play_game(game, RandomAgent(seed=42), RandomAgent(seed=43))
    for state, action in zip(record.states, record.actions):
        assert game.legal_actions(state)[action], f"illegal move {action}"


def test_players_alternate(game: Connect4) -> None:
    record = play_game(game, RandomAgent(seed=7), RandomAgent(seed=8))
    assert record.players[0] == 1  # X always starts
    for i in range(1, len(record)):
        assert record.players[i] == -record.players[i - 1]


def test_value_target_reflects_outcome(game: Connect4) -> None:
    record = play_game(game, RandomAgent(seed=12), RandomAgent(seed=34))
    if record.outcome == 0:
        # All targets are 0 for a draw.
        for ply in range(len(record)):
            assert record.value_target_for(ply) == 0.0
    else:
        winner = record.outcome
        for ply in range(len(record)):
            expected = 1.0 if record.players[ply] == winner else -1.0
            assert record.value_target_for(ply) == expected


def test_first_mover_has_slight_edge(game: Connect4) -> None:
    """Over many random games X wins more often than O (gets the last move
    in the 42-ply case, plus an extra ply across all odd-length games). This
    is a sanity check that nothing in the harness systematically biases play."""
    n_games = 400
    x_wins = o_wins = draws = 0
    for i in range(n_games):
        record = play_game(game, RandomAgent(seed=2 * i), RandomAgent(seed=2 * i + 1))
        if record.outcome == 1:
            x_wins += 1
        elif record.outcome == -1:
            o_wins += 1
        else:
            draws += 1
    # Empirically X wins ~55% of random Connect4 games. We use loose bounds
    # to avoid flakiness while still catching a broken harness.
    assert x_wins > o_wins, f"X={x_wins} O={o_wins} D={draws} — first-mover edge missing"
    assert x_wins + o_wins + draws == n_games
