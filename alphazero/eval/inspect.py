"""Text-based introspection of a PUCT search and the network's outputs.

When you want to ask "what is the agent thinking *right now*?", this is
the module. Three primitives:

    * `render_tree(root)`         — top-N children with N / Q / P columns.
                                    Q is shown from the *parent's* (decision-
                                    maker's) perspective; a child node stores
                                    Q in its own frame, so we negate when
                                    printing.
    * `principal_variation(root)` — the most-visited line of play, the
                                    chain of moves the search "recommends".
    * `render_policy_on_board(...)` — the board, plus the policy
                                      distribution shown as percentages
                                      above each column.

These are pure stdlib + numpy — no graphical dependencies. The CLI/notebook
can render the strings however suits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..games.base import Game

if TYPE_CHECKING:
    from ..mcts.puct import PUCTNode


def render_tree(root: "PUCTNode", top_k: int = 7) -> str:
    """Return a text summary of the search tree rooted at `root`.

    Columns:
        action  — action index leading to the child
        N       — child visit count
        Q       — mean value seen at the child, *from the root's perspective*
                  (we negate the child's stored Q since stored Q is in the
                  child's to-play frame)
        P       — prior assigned to the child by the network's policy head
        U       — the exploration bonus's *direction* at the next selection
                  (c_puct · P · sqrt(N_parent) / (1 + N_child)) is not shown
                  literally because it depends on c_puct; instead we show
                  the dimensionless P · sqrt(N_parent) / (1 + N_child).
    """
    side = "X" if root.to_play == 1 else "O"
    lines = [f"Root: to_play={side}  N={root.N}  Q={root.Q:+.3f}"]

    children = sorted(root.children.items(), key=lambda kv: -kv[1].N)
    if not children:
        lines.append("  (no children — terminal state)")
        return "\n".join(lines)

    shown = min(top_k, len(children))
    lines.append(f"  top {shown} children by visit count:")
    lines.append(
        f"    {'action':>6} {'N':>5} {'Q_root':>8} {'P':>7} {'P·√N/(1+n)':>12}"
    )
    sqrt_parent = (root.N or 1) ** 0.5
    for action, child in children[:shown]:
        q_root = -child.Q if child.N > 0 else 0.0
        u_dim = child.prior * sqrt_parent / (1 + child.N)
        lines.append(
            f"    {action:>6} {child.N:>5} {q_root:>+8.3f} {child.prior:>7.3f} {u_dim:>12.4f}"
        )
    return "\n".join(lines)


def principal_variation(root: "PUCTNode", max_depth: int = 5) -> list[int]:
    """Walk down the tree, picking the most-visited child at each level.

    Returns the action indices, length ≤ max_depth. Stops early if a node
    has no children (unexpanded or terminal).
    """
    actions: list[int] = []
    node = root
    for _ in range(max_depth):
        if not node.children:
            break
        best_action = max(node.children, key=lambda a: node.children[a].N)
        actions.append(best_action)
        node = node.children[best_action]
    return actions


def render_policy_on_board(
    game: Game,
    state,
    policy: np.ndarray,
    value: float | None = None,
) -> str:
    """Show the board with the policy as percentages above each column.

    The policy is rounded to the nearest 1% — enough resolution to see
    where the net is putting its mass, low enough resolution to read
    quickly. Illegal moves (zero probability) are shown as "  -" so a
    blank-looking column is unambiguous.
    """
    parts: list[str] = []

    col_header = " " + " ".join(f"{i:>3d}" for i in range(policy.shape[0]))
    pct_row = " " + " ".join(
        ("  -" if p == 0.0 else f"{int(round(p * 100)):>3d}") for p in policy
    )
    parts.append("column:" + col_header)
    parts.append("policy:" + pct_row + "  (%)")
    parts.append("")
    parts.append(game.render(state))

    if value is not None:
        side = "X" if game.current_player(state) == 1 else "O"
        parts.append(f"value (from {side}'s view): {value:+.3f}")

    return "\n".join(parts)
