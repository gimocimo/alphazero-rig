"""Sanity tests for the Connect4 environment.

These are *rule-correctness* tests — not learning tests. They should run in well
under a second and pin down the game semantics so MCTS and the net can trust
the substrate.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.connect4 import COLS, ROWS, Connect4, Connect4State


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def _play(game: Connect4, actions: list[int]) -> Connect4State:
    state = game.initial_state()
    for a in actions:
        state = game.step(state, a)
    return state


def test_initial_state_is_empty(game: Connect4) -> None:
    s = game.initial_state()
    assert (s.board == 0).all()
    assert s.to_move == 1
    assert game.legal_actions(s).all()
    assert not game.is_terminal(s)


def test_step_drops_to_bottom(game: Connect4) -> None:
    s = game.step(game.initial_state(), 3)
    expected = np.zeros((ROWS, COLS), dtype=np.int8)
    expected[ROWS - 1, 3] = 1
    assert np.array_equal(s.board, expected)
    assert s.to_move == -1


def test_step_stacks_pieces(game: Connect4) -> None:
    s = _play(game, [3, 3, 3])
    # Three pieces stacked in column 3 starting from the bottom.
    assert s.board[ROWS - 1, 3] == 1
    assert s.board[ROWS - 2, 3] == -1
    assert s.board[ROWS - 3, 3] == 1


def test_full_column_is_illegal(game: Connect4) -> None:
    s = _play(game, [0] * ROWS)
    legal = game.legal_actions(s)
    assert not legal[0]
    assert legal[1:].all()


def test_step_into_full_column_raises(game: Connect4) -> None:
    s = _play(game, [0] * ROWS)
    with pytest.raises(ValueError, match="full"):
        game.step(s, 0)


def test_step_out_of_range_raises(game: Connect4) -> None:
    s = game.initial_state()
    with pytest.raises(ValueError, match="out of range"):
        game.step(s, COLS)


def test_vertical_win(game: Connect4) -> None:
    # X plays col 0 four times; O plays col 1 three times in between.
    s = _play(game, [0, 1, 0, 1, 0, 1, 0])
    assert game.winner(s) == 1


def test_horizontal_win(game: Connect4) -> None:
    s = _play(game, [0, 0, 1, 1, 2, 2, 3])
    assert game.winner(s) == 1


def test_backslash_diagonal_win(game: Connect4) -> None:
    # X aims for the \-diagonal: (2,0), (3,1), (4,2), (5,3).
    # Sequence builds support stacks and lands the winning piece at (2,0) last.
    s = _play(game, [3, 0, 1, 1, 1, 2, 2, 0, 6, 0, 0])
    assert game.winner(s) == 1


def test_slash_diagonal_win(game: Connect4) -> None:
    # X aims for the /-diagonal: (5,0), (4,1), (3,2), (2,3).
    s = _play(game, [0, 1, 2, 2, 1, 3, 2, 3, 6, 3, 3])
    assert game.winner(s) == 1


def test_draw_when_board_full_without_winner(game: Connect4) -> None:
    # Hand-craft a full board with no four-in-a-row anywhere. The 2x2-block
    # checkerboard breaks horizontals/verticals at run-length 2, and the
    # vertical period of 2 (paired with horizontal alternation) breaks both
    # diagonals.
    board = np.array(
        [
            [1, -1, 1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1],
            [-1, 1, -1, 1, -1, 1, -1],
            [-1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=np.int8,
    )
    s = Connect4State(board=board, to_move=1)
    assert game.winner(s) == 0
    assert game.is_terminal(s)


def test_encoding_is_from_to_move_perspective(game: Connect4) -> None:
    s = game.step(game.initial_state(), 3)  # X has played; O to move.
    enc = game.encode(s)
    assert enc.shape == (2, ROWS, COLS)
    assert enc.dtype == np.float32
    # Plane 0 = own (O's) pieces — currently zero.
    assert enc[0].sum() == 0
    # Plane 1 = opponent (X's) piece at (5, 3).
    assert enc[1, ROWS - 1, 3] == 1
    assert enc[1].sum() == 1


def test_current_player_alternates(game: Connect4) -> None:
    s = game.initial_state()
    assert game.current_player(s) == 1
    s = game.step(s, 0)
    assert game.current_player(s) == -1
    s = game.step(s, 1)
    assert game.current_player(s) == 1


def test_render_returns_string(game: Connect4) -> None:
    s = game.step(game.initial_state(), 3)
    out = game.render(s)
    assert isinstance(out, str)
    assert "X" in out  # X has played
    assert "to move: O" in out
