"""Pretty-printed game replays.

`render_game_replay(game, agent_a, agent_b)` plays a full game between two
agents and returns a multi-line string showing, for each ply: side to move,
the search-tree summary if the agent is PUCT-style, the chosen move, and
the resulting board. Useful for inspecting *what the agent is thinking*
rather than just *what it played*.

For PUCT-style agents we use `build_root` to run search once and then take
both the action (argmax visits) and the printed tree from the same root —
no double computation.
"""

from __future__ import annotations

from typing import TypeVar

from ..agents.base import Agent
from ..games.base import Game
from .inspect import render_tree

S = TypeVar("S")


def _agent_label(agent: Agent) -> str:
    name = getattr(agent, "name", agent.__class__.__name__)
    sims = getattr(agent, "simulations", None)
    return f"{name}(sims={sims})" if sims is not None else name


def render_game_replay(
    game: Game[S],
    agent_a: Agent,
    agent_b: Agent,
    show_tree: bool = True,
    top_k_per_ply: int = 5,
    max_plies: int | None = None,
) -> str:
    """Play one game and return a text replay.

    `agent_a` plays +1 (X, moves first); `agent_b` plays -1 (O).
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"X: {_agent_label(agent_a)}    O: {_agent_label(agent_b)}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Initial position:")
    state = game.initial_state()
    lines.append(game.render(state))

    ply = 0
    while not game.is_terminal(state):
        if max_plies is not None and ply >= max_plies:
            lines.append("\n(max plies reached, truncating)")
            break

        player = game.current_player(state)
        agent = agent_a if player == 1 else agent_b
        side = "X" if player == 1 else "O"
        ply += 1

        lines.append("")
        lines.append(f"--- Ply {ply}: {side} ({_agent_label(agent)}) ---")

        if show_tree and hasattr(agent, "build_root"):
            root = agent.build_root(game, state)  # type: ignore[attr-defined]
            lines.append(render_tree(root, top_k=top_k_per_ply))
            action = max(root.children, key=lambda a: root.children[a].N)
        else:
            action = agent.select_action(game, state)

        lines.append(f"Move: column {action}")
        state = game.step(state, action)
        lines.append(game.render(state))

    lines.append("")
    if game.is_terminal(state):
        outcome = game.winner(state)
        result = {1: "X wins", -1: "O wins", 0: "draw"}[outcome]  # type: ignore[index]
    else:
        result = "(unfinished)"
    lines.append("=" * 60)
    lines.append(f"Result: {result}   ({ply} plies)")
    lines.append("=" * 60)
    return "\n".join(lines)
