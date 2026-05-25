"""The Agent contract — everything that plays a game implements this."""

from __future__ import annotations

from typing import Protocol, TypeVar

from ..games.base import Game

S = TypeVar("S")


class Agent(Protocol):
    """Anything that can pick a move given a game state.

    Kept deliberately minimal: a RandomAgent, a pure-MCTS agent, and a
    PUCT+network agent will all satisfy this same interface, which is what lets
    the arena pit any pair against each other without special-casing.
    """

    name: str

    def select_action(self, game: Game[S], state: S) -> int:
        """Return a legal action index. Raises if no legal moves exist."""
        ...
