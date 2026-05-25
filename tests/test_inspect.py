"""Tests for the search-tree + policy inspection helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.eval.inspect import (
    principal_variation,
    render_policy_on_board,
    render_tree,
)
from alphazero.games.connect4 import Connect4
from alphazero.mcts.puct import PUCTAgent
from alphazero.nets.pvnet import PVNet


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net() -> PVNet:
    torch.manual_seed(0)
    return PVNet()


def test_render_tree_shows_root_and_top_children(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=50, seed=0)
    root = agent.build_root(game, game.initial_state())
    out = render_tree(root, top_k=4)
    assert "Root" in out
    assert "to_play=X" in out
    assert f"N={root.N}" in out
    # Should list a row per top child.
    assert out.count("\n") >= 4 + 2  # header lines + at least 4 children


def test_render_tree_handles_terminal_state_root(net: PVNet, game: Connect4) -> None:
    """Manually construct a root for a terminal state — search refuses to
    build it, but the renderer should cope gracefully if it's ever handed
    such a node directly (defensive coding)."""
    from alphazero.mcts.puct import PUCTNode

    state = game.initial_state()
    for a in [0, 1, 0, 1, 0, 1, 0]:  # X vertical-wins col 0
        state = game.step(state, a)
    leaf = PUCTNode(state=state, to_play=game.current_player(state))
    out = render_tree(leaf)
    assert "no children" in out


def test_principal_variation_walks_most_visited(net: PVNet, game: Connect4) -> None:
    agent = PUCTAgent(net, simulations=80, seed=0)
    root = agent.build_root(game, game.initial_state())
    pv = principal_variation(root, max_depth=3)
    assert 1 <= len(pv) <= 3
    # First action should be the highest-visited child of root.
    expected_first = max(root.children, key=lambda a: root.children[a].N)
    assert pv[0] == expected_first


def test_render_policy_shows_percentages_and_illegal_marker(
    net: PVNet, game: Connect4
) -> None:
    state = game.initial_state()
    # Fill column 0 so the renderer marks it as "  -".
    for _ in range(6):
        state = game.step(state, 0)

    policy = np.zeros(7, dtype=np.float32)
    policy[1:] = 1 / 6  # uniform over legal cols
    out = render_policy_on_board(game, state, policy, value=0.42)

    assert "policy:" in out
    assert "column:" in out
    assert "  -" in out  # illegal col 0 marker
    assert "+0.420" in out  # value line
    assert "to move" in out  # game.render passthrough


def test_render_policy_without_value(net: PVNet, game: Connect4) -> None:
    state = game.initial_state()
    policy = np.full(7, 1 / 7, dtype=np.float32)
    out = render_policy_on_board(game, state, policy)  # no value
    assert "value" not in out
