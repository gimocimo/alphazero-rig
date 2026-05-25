"""Head-to-head agent evaluation.

The training loop uses this to decide whether to promote a freshly-trained
network over the current best one — promote only if the new net wins enough
games against the old in a noise-free arena.

Two design choices to call out:

1. **Sides are alternated** between games. Without this, an agent that's
   uniformly better at playing white than black would get a misleading
   score. Half the games have agent A as +1, half as -1.

2. **Draws count as half a win for each side.** This is the Elo convention
   — it makes the winrate continuous and avoids penalising agents that
   correctly identify drawn positions. For Connect4 with perfect play
   draws are rare, but the convention is right.

Construct evaluation agents with `dirichlet_epsilon=0` (no noise) and
temperature=0 (argmax). Noise during measurement just adds variance to
the strength estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from ..agents.base import Agent
from ..games.base import Game
from ..training.self_play import play_game

S = TypeVar("S")


@dataclass
class ArenaResult:
    """Tally from agent A's perspective."""

    agent_a_wins: int
    agent_b_wins: int
    draws: int

    @property
    def total(self) -> int:
        return self.agent_a_wins + self.agent_b_wins + self.draws

    @property
    def agent_a_winrate(self) -> float:
        """Winrate with draws counted as half (Elo convention)."""
        if self.total == 0:
            return 0.0
        return (self.agent_a_wins + 0.5 * self.draws) / self.total

    def __repr__(self) -> str:
        return (
            f"ArenaResult(A={self.agent_a_wins} "
            f"B={self.agent_b_wins} "
            f"D={self.draws} "
            f"winrate_A={self.agent_a_winrate:.3f})"
        )


def arena_play(
    game: Game[S],
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    alternate_sides: bool = True,
) -> ArenaResult:
    """Pit two agents against each other for `num_games` and tally outcomes.

    With `alternate_sides=True`, half the games have A as the first player.
    """
    if num_games < 1:
        raise ValueError("num_games must be >= 1")

    a_wins = b_wins = draws = 0
    for i in range(num_games):
        a_plays_first = (not alternate_sides) or (i % 2 == 0)
        if a_plays_first:
            record = play_game(game, agent_a, agent_b)
            if record.outcome == 1:
                a_wins += 1
            elif record.outcome == -1:
                b_wins += 1
            else:
                draws += 1
        else:
            record = play_game(game, agent_b, agent_a)
            # Outcome is from +1's perspective; A is -1 here.
            if record.outcome == -1:
                a_wins += 1
            elif record.outcome == 1:
                b_wins += 1
            else:
                draws += 1

    return ArenaResult(agent_a_wins=a_wins, agent_b_wins=b_wins, draws=draws)
