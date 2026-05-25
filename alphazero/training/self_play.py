"""Self-play harness.

Plays a single game between two agents and records the full trajectory. This
data structure is deliberately the same one we will feed to the trainer later:
when MCTS replaces RandomAgent, all that changes is the `policy_targets` field
gets populated by search visit-counts instead of left as None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np

from ..agents.base import Agent
from ..games.base import Game

S = TypeVar("S")


@dataclass
class GameRecord(Generic[S]):
    """The complete play-by-play of a single game.

    Fields:
        states:           Position before each move. Length N.
        actions:          Move taken from each state. Length N.
        players:          Player-to-move at each state (+1 or -1). Length N.
        outcome:          Final winner: +1, -1, or 0 (draw). Set when the
                          game terminates; raises if accessed before then.
        policy_targets:   Reserved for MCTS-derived distributions over the
                          action space, one per state. Left None for random
                          play; filled in when we plug in PUCT.
    """

    states: list[S] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    outcome: Optional[int] = None
    policy_targets: Optional[list[np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.actions)

    def value_target_for(self, ply: int) -> float:
        """Outcome from the perspective of the player who moved at `ply`.

        AlphaZero's value head learns this signal: +1 if you eventually won
        from this position, -1 if you lost, 0 if drawn.
        """
        if self.outcome is None:
            raise RuntimeError("value_target_for called before game ended")
        return float(self.outcome * self.players[ply])


def play_game(
    game: Game[S],
    agent_plus: Agent,
    agent_minus: Agent,
    max_plies: int | None = None,
) -> GameRecord[S]:
    """Play one game between `agent_plus` (player +1) and `agent_minus` (player -1).

    Args:
        game: The environment.
        agent_plus: Plays as +1 (moves first).
        agent_minus: Plays as -1.
        max_plies: Optional ply limit (safety net for non-terminating games;
            real games like Connect4 always terminate, but useful for debug).

    Returns a `GameRecord` with the trajectory and outcome.
    """
    state = game.initial_state()
    record: GameRecord[S] = GameRecord()

    while not game.is_terminal(state):
        if max_plies is not None and len(record) >= max_plies:
            break

        player = game.current_player(state)
        agent = agent_plus if player == 1 else agent_minus
        action = agent.select_action(game, state)

        legal = game.legal_actions(state)
        if not legal[action]:
            raise RuntimeError(
                f"Agent {agent.name!r} returned illegal action {action} "
                f"(legal mask: {legal.tolist()})"
            )

        record.states.append(state)
        record.actions.append(action)
        record.players.append(player)
        state = game.step(state, action)

    record.outcome = game.winner(state) if game.is_terminal(state) else 0
    return record
