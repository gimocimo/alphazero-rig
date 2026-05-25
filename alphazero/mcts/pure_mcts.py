"""Pure MCTS with UCB1 selection and random rollouts.

The "pure" here means *no neural network* — the value of a leaf is estimated by
playing the rest of the game uniformly at random. This is the classical 2006
algorithm, and it's the baseline the learned agent will eventually have to beat.

Why it matters for AlphaZero understanding:
    - The tree-traversal logic (select → expand → simulate → backprop) is *exactly*
      what PUCT will do. Only the selection formula changes.
    - It shows viscerally that search alone, with no learning, already produces
      strong play — which is the whole motivation for grafting search onto a
      learned policy/value net.

Value convention (worth pinning down once):
    Each node stores `W` — total value seen — *from the perspective of the
    player to move at that node*. A child's Q is the expected outcome for the
    child's to-play player; the parent's gain from moving to that child is
    therefore `-child.Q` (zero-sum). UCB selection negates accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np

from ..games.base import Game

S = TypeVar("S")


@dataclass
class Node(Generic[S]):
    state: S
    to_play: int  # +1 or -1, whoever is about to move from this position
    parent: Optional["Node[S]"] = None
    action_from_parent: Optional[int] = None
    children: dict[int, "Node[S]"] = field(default_factory=dict)
    untried_actions: list[int] = field(default_factory=list)
    N: int = 0  # visit count
    W: float = 0.0  # total value from to_play's perspective

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


class MCTSAgent(Generic[S]):
    """UCB1 + random-rollout MCTS. Plays under the `Agent` protocol."""

    name = "mcts"

    def __init__(
        self,
        simulations: int = 100,
        exploration: float = math.sqrt(2),
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        self.simulations = simulations
        self.c = exploration
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game[S], state: S) -> int:
        if game.is_terminal(state):
            raise RuntimeError("MCTSAgent.select_action called on a terminal state")

        root = self._make_node(game, state, parent=None, action_from_parent=None)
        for _ in range(self.simulations):
            self._simulate(game, root)

        # Robustness over Q-argmax: visit counts are less noisy at low N.
        return max(root.children, key=lambda a: root.children[a].N)

    def _make_node(
        self,
        game: Game[S],
        state: S,
        parent: Optional[Node[S]],
        action_from_parent: Optional[int],
    ) -> Node[S]:
        untried: list[int]
        if game.is_terminal(state):
            untried = []
        else:
            untried = [int(a) for a in np.flatnonzero(game.legal_actions(state))]
        return Node(
            state=state,
            to_play=game.current_player(state),
            parent=parent,
            action_from_parent=action_from_parent,
            untried_actions=untried,
        )

    def _simulate(self, game: Game[S], root: Node[S]) -> None:
        # 1. Selection — walk down the tree using UCB1 until we hit a node
        # that is either terminal or has at least one unexpanded child.
        node = root
        while not game.is_terminal(node.state) and node.is_fully_expanded:
            node = self._select_child(node)

        # 2. Expansion — pick one untried action and create the child.
        if not game.is_terminal(node.state):
            i = int(self.rng.integers(len(node.untried_actions)))
            action = node.untried_actions.pop(i)
            new_state = game.step(node.state, action)
            child = self._make_node(
                game, new_state, parent=node, action_from_parent=action
            )
            node.children[action] = child
            node = child

        # 3. Rollout — play uniformly at random from `node` until the game ends.
        outcome = self._rollout(game, node.state)

        # 4. Backpropagation — walk back to root, updating each node's stats
        # with the outcome viewed from that node's to_play perspective.
        cursor: Optional[Node[S]] = node
        while cursor is not None:
            cursor.N += 1
            cursor.W += outcome * cursor.to_play
            cursor = cursor.parent

    def _select_child(self, node: Node[S]) -> Node[S]:
        """UCB1, evaluated from the parent's (negated) perspective."""
        log_N = math.log(node.N)
        c = self.c
        best_score = -math.inf
        best_child: Optional[Node[S]] = None
        for child in node.children.values():
            exploit = -child.Q  # parent's gain = -child's value
            explore = c * math.sqrt(log_N / child.N)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None  # node is fully expanded ⇒ at least one child
        return best_child

    def _rollout(self, game: Game[S], state: S) -> int:
        """Uniform-random playout. Returns the absolute outcome (+1/-1/0)."""
        while not game.is_terminal(state):
            legal = np.flatnonzero(game.legal_actions(state))
            action = int(legal[self.rng.integers(legal.size)])
            state = game.step(state, action)
        winner = game.winner(state)
        assert winner is not None
        return winner
