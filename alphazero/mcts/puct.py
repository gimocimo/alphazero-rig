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
        dirichlet_alpha: float = 0.5,
        dirichlet_epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """PUCT search.

        dirichlet_alpha, dirichlet_epsilon:
            Optional Dirichlet noise mixed into the root priors before search:
                P_root(a) ← (1 - ε) · P(a) + ε · Dir(α)
            Disabled by default (ε = 0). Enable during *self-play* (typically
            ε ≈ 0.25) to ensure the search occasionally explores moves the
            current net would dismiss — essential for discovering improvements.
            Disable during *evaluation*: noise during a tournament would just
            add variance to the strength measurement.
        """
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: Game[S], state: S) -> int:
        """Convenience: argmax-visits action only. For training, use
        `select_action_with_visits` to also get the policy target."""
        action, _ = self.select_action_with_visits(game, state, temperature=0.0)
        return action

    def select_action_with_visits(
        self,
        game: Game[S],
        state: S,
        temperature: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """Run search, return (chosen action, normalised root-visit distribution).

        The visit distribution is the policy training target — search amplifies
        the network's policy, and that amplified signal is what we distill.

        `temperature` controls action selection only (not the returned policy):
            T = 0   → argmax(visits). Deterministic. Use at evaluation time.
            T > 0   → sample from visits^(1/T). Use during self-play; T=1 in
                      the opening encourages diverse data; T→0 in the endgame
                      plays decisively.
        """
        if game.is_terminal(state):
            raise RuntimeError(
                "PUCTAgent.select_action_with_visits called on a terminal state"
            )

        root = PUCTNode(state=state, to_play=game.current_player(state))
        self._expand(game, root)
        self._add_root_dirichlet_noise(root)

        for _ in range(self.simulations):
            self._simulate(game, root)

        visits = np.zeros(game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.N

        total = visits.sum()
        if total <= 0:
            # Can only happen if simulations=0 — guarded by __init__ but be safe.
            legal = game.legal_actions(state)
            visits = legal.astype(np.float32)
            total = visits.sum()

        policy_target = visits / total

        if temperature <= 0:
            chosen = int(np.argmax(visits))
        else:
            # visits^(1/T) — at T=1 this is just the visit distribution.
            sharpened = visits ** (1.0 / temperature)
            probs = sharpened / sharpened.sum()
            chosen = int(self.rng.choice(visits.size, p=probs))

        return chosen, policy_target

    def build_root(
        self, game: Game[S], state: S, add_dirichlet_noise: bool = False
    ) -> PUCTNode[S]:
        """Run search and return the root node (for inspection / viz).

        Distinct from `select_action_with_visits` because it exposes the
        full tree — caller can render it, walk the principal variation,
        compute custom statistics, etc.
        """
        if game.is_terminal(state):
            raise RuntimeError("build_root called on a terminal state")
        root = PUCTNode(state=state, to_play=game.current_player(state))
        self._expand(game, root)
        if add_dirichlet_noise:
            self._add_root_dirichlet_noise(root)
        for _ in range(self.simulations):
            self._simulate(game, root)
        return root

    def _add_root_dirichlet_noise(self, root: PUCTNode[S]) -> None:
        if self.dirichlet_epsilon <= 0:
            return
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])

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
