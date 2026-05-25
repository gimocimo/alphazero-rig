"""Tests for the head-to-head arena."""

from __future__ import annotations

import pytest

from alphazero.agents.random_agent import RandomAgent
from alphazero.eval.arena import ArenaResult, arena_play
from alphazero.games.connect4 import Connect4
from alphazero.mcts.pure_mcts import MCTSAgent


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def test_arena_result_winrate_with_draws() -> None:
    r = ArenaResult(agent_a_wins=4, agent_b_wins=4, draws=2)
    # (4 + 0.5*2) / 10 = 0.5 — drawn arenas show winrate 0.5.
    assert r.agent_a_winrate == 0.5
    assert r.total == 10


def test_arena_result_clean_winner() -> None:
    r = ArenaResult(agent_a_wins=10, agent_b_wins=0, draws=0)
    assert r.agent_a_winrate == 1.0


def test_arena_result_clean_loser() -> None:
    r = ArenaResult(agent_a_wins=0, agent_b_wins=10, draws=0)
    assert r.agent_a_winrate == 0.0


def test_arena_with_two_random_agents_is_near_half(game: Connect4) -> None:
    """With sides alternated, two random agents should converge to ~0.5
    winrate. Loose bounds keep this from being flaky."""
    a = RandomAgent(seed=0)
    b = RandomAgent(seed=1)
    result = arena_play(game, a, b, num_games=60)
    assert result.total == 60
    assert 0.30 <= result.agent_a_winrate <= 0.70


def test_mcts_crushes_random_in_arena(game: Connect4) -> None:
    """MCTS at 50 sims should win the lion's share against random,
    regardless of which side it plays."""
    mcts = MCTSAgent(simulations=50, seed=0)
    rand = RandomAgent(seed=42)
    result = arena_play(game, mcts, rand, num_games=10)
    assert result.agent_a_winrate >= 0.8, (
        f"MCTS only won {result.agent_a_winrate:.2f} vs Random"
    )


def test_arena_invalid_num_games() -> None:
    with pytest.raises(ValueError, match="num_games"):
        arena_play(Connect4(), RandomAgent(), RandomAgent(), num_games=0)
