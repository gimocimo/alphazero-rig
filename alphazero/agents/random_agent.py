"""Uniform-random baseline agent."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from ..games.base import Game

S = TypeVar("S")


class RandomAgent:
    """Samples uniformly from legal moves. The floor every learned agent must clear."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game[S], state: S) -> int:
        legal = game.legal_actions(state)
        legal_indices = np.flatnonzero(legal)
        if legal_indices.size == 0:
            raise RuntimeError("RandomAgent called on a state with no legal moves")
        return int(self.rng.choice(legal_indices))
