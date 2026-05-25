"""PUCT search — the AlphaZero variant of MCTS.

Differences from pure MCTS (in `pure_mcts.py`):

1. **Selection** uses the PUCT formula instead of UCB1::

       score(child) = Q(child, from parent's view)
                    + c_puct * child.prior * sqrt(parent.N) / (1 + child.N)

   The crucial change is `child.prior` — the network's policy output. The
   search is no longer biased only by what's been explored; it is also
   pulled toward moves the net thinks are good. As the net improves, the
   search spends its budget more wisely.

2. **No rollouts.** When we reach a leaf, the value backed up is the
   network's value-head output, not the result of a random playout. Random
   rollouts in pure MCTS are noisy and slow; a (well-trained) value net is
   sharp and one forward pass.

3. **Expansion is one-shot.** When a leaf is first visited, we evaluate the
   net *once* and create *all* legal children at once, each carrying its
   prior P(s, a). Compare to pure MCTS, which expanded one untried action
   per visit.

The tree-traversal skeleton (select → expand → backprop) is otherwise
identical to pure MCTS. Same value-perspective convention: each node's W
stores total value from *the to-play player's perspective at that node*;
parent UCB scoring negates across the edge because zero-sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import numpy as np
import torch

from ..games.base import Game
from ..nets.pvnet import PVNet, predict

S = TypeVar("S")


@dataclass
class PUCTNode(Generic[S]):
    state: S
    to_play: int
    parent: Optional["PUCTNode[S]"] = None
    action_from_parent: Optional[int] = None
    prior: float = 0.0  # P(parent → this), unused at root
    children: dict[int, "PUCTNode[S]"] = field(default_factory=dict)
    N: int = 0  # visit count
    W: float = 0.0  # total value from to_play's perspective
    is_expanded: bool = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class PUCTAgent(Generic[S]):
    """Network-guided MCTS. Plays under the `Agent` protocol."""

    name = "puct"

    def __init__(
        self,
        net: PVNet,
        simulations: int = 100,
        c_puct: float = 1.5,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game[S], state: S) -> int:
        if game.is_terminal(state):
            raise RuntimeError("PUCTAgent.select_action called on a terminal state")

        root = PUCTNode(state=state, to_play=game.current_player(state))
        # Expand root immediately so the first selection has priors to work with.
        self._expand(game, root)

        for _ in range(self.simulations):
            self._simulate(game, root)

        # Most-visited child — robust to noisy Q at low N.
        return max(root.children, key=lambda a: root.children[a].N)

    def _simulate(self, game: Game[S], root: PUCTNode[S]) -> None:
        # 1. Selection: descend until we hit an unexpanded node or a terminal.
        node = root
        while node.is_expanded and not game.is_terminal(node.state):
            node = self._select_child(node)

        # 2. Evaluation: either the terminal outcome or the network's value.
        if game.is_terminal(node.state):
            winner = game.winner(node.state)
            assert winner is not None
            # Convert absolute outcome (+1/-1/0) to leaf-perspective once.
            leaf_value = winner * node.to_play
        else:
            leaf_value = self._expand(game, node)

        # 3. Backpropagation. We hold `leaf_value` in the leaf's to_play
        # frame; to update each ancestor in *its* frame we apply
        # `leaf_value * (leaf.to_play * cursor.to_play)`. Equivalently:
        # convert to absolute frame once, then multiply by each cursor.to_play.
        absolute_value = leaf_value * node.to_play
        cursor: Optional[PUCTNode[S]] = node
        while cursor is not None:
            cursor.N += 1
            cursor.W += absolute_value * cursor.to_play
            cursor = cursor.parent

    def _expand(self, game: Game[S], node: PUCTNode[S]) -> float:
        """Evaluate the net at `node`, create all legal children with priors,
        and return the value (from node.to_play's perspective)."""
        probs, value = predict(self.net, game, node.state, device=self.device)
        for action in range(game.action_size):
            if probs[action] > 0:
                new_state = game.step(node.state, action)
                child = PUCTNode(
                    state=new_state,
                    to_play=game.current_player(new_state),
                    parent=node,
                    action_from_parent=action,
                    prior=float(probs[action]),
                )
                node.children[action] = child
        node.is_expanded = True
        return value

    def _select_child(self, parent: PUCTNode[S]) -> PUCTNode[S]:
        """PUCT selection from parent's perspective.

        Q(child) is stored from the child's to_play frame; the parent's gain
        from moving to that child is `-child.Q` (zero-sum across the edge).
        On the first visit of parent (N=0), the U term vanishes for all
        children — the prior tie-breaks via the dict iteration order, which
        is fine in expectation since subsequent simulations will explore
        based on priors.
        """
        sqrt_N = math.sqrt(parent.N)
        c = self.c_puct
        best_score = -math.inf
        best_child: Optional[PUCTNode[S]] = None
        for child in parent.children.values():
            q = -child.Q if child.N > 0 else 0.0
            u = c * child.prior * sqrt_N / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child
