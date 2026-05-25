"""Abstract interface for two-player zero-sum games used by the AlphaZero rig.

A `Game` is a stateless transition system — it never holds the current state itself.
State objects are plain immutable dataclasses owned by callers (MCTS, self-play loops).
This makes the search trivially safe to use across threads/processes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import numpy as np

S = TypeVar("S")


class Game(ABC, Generic[S]):
    """Two-player zero-sum, perfect-information game.

    Convention:
        - Players are +1 and -1.
        - `winner(state)` returns +1, -1, 0 (draw), or None (game ongoing).
        - `encode(state)` always returns the board from the to-move player's
          perspective (own pieces on plane 0, opponent pieces on plane 1). This
          lets a single network handle both players.
    """

    name: str

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Total number of discrete actions in the action space."""

    @abstractmethod
    def initial_state(self) -> S:
        """Return the starting state (player +1 to move)."""

    @abstractmethod
    def legal_actions(self, state: S) -> np.ndarray:
        """Boolean mask of shape `(action_size,)`. True where the action is legal."""

    @abstractmethod
    def step(self, state: S, action: int) -> S:
        """Apply `action` and return the resulting state. Raises on illegal moves."""

    @abstractmethod
    def current_player(self, state: S) -> int:
        """Return +1 or -1 — whichever player is to move."""

    @abstractmethod
    def winner(self, state: S) -> Optional[int]:
        """Return +1 / -1 / 0 if the game is over; None if still in progress."""

    def is_terminal(self, state: S) -> bool:
        return self.winner(state) is not None

    @abstractmethod
    def encode(self, state: S) -> np.ndarray:
        """Encode the state for the policy/value net.

        Shape `(C, H, W)`, dtype float32, from the to-move player's perspective.
        """

    @abstractmethod
    def render(self, state: S) -> str:
        """Human-readable string rendering, used for debugging."""
