"""Connect Four on a 6x7 board.

The board is stored as an int8 numpy array. Row 0 is the *top* row, row 5 is the
*bottom* (gravity pulls pieces down). Players are +1 and -1; empty cells are 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Game

ROWS = 6
COLS = 7
CONNECT = 4


@dataclass(frozen=True)
class Connect4State:
    board: np.ndarray  # shape (ROWS, COLS), dtype int8, values in {-1, 0, +1}
    to_move: int  # +1 or -1

    def __post_init__(self) -> None:
        # Cheap invariants — catches dataclass misuse early.
        assert self.board.shape == (ROWS, COLS)
        assert self.to_move in (1, -1)


class Connect4(Game[Connect4State]):
    name = "connect4"

    @property
    def action_size(self) -> int:
        return COLS

    def initial_state(self) -> Connect4State:
        return Connect4State(board=np.zeros((ROWS, COLS), dtype=np.int8), to_move=1)

    def legal_actions(self, state: Connect4State) -> np.ndarray:
        # A column is playable iff its top cell is still empty.
        return state.board[0] == 0

    def step(self, state: Connect4State, action: int) -> Connect4State:
        if not (0 <= action < COLS):
            raise ValueError(f"action {action} out of range [0, {COLS})")
        col = state.board[:, action]
        empty_rows = np.where(col == 0)[0]
        if empty_rows.size == 0:
            raise ValueError(f"column {action} is full")
        row = int(empty_rows[-1])  # lowest empty row
        new_board = state.board.copy()
        new_board[row, action] = state.to_move
        return Connect4State(board=new_board, to_move=-state.to_move)

    def current_player(self, state: Connect4State) -> int:
        return state.to_move

    def winner(self, state: Connect4State) -> Optional[int]:
        for player in (1, -1):
            if _has_four_in_a_row(state.board, player):
                return player
        if not (state.board == 0).any():
            return 0  # draw — board full, no winner
        return None

    def encode(self, state: Connect4State) -> np.ndarray:
        own = (state.board == state.to_move).astype(np.float32)
        opp = (state.board == -state.to_move).astype(np.float32)
        return np.stack([own, opp])  # (2, ROWS, COLS)

    def render(self, state: Connect4State) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = [
            "|" + "|".join(symbols[int(v)] for v in row) + "|" for row in state.board
        ]
        footer = " " + " ".join(str(i) for i in range(COLS))
        turn = "X" if state.to_move == 1 else "O"
        return "\n".join(lines) + "\n" + footer + f"\nto move: {turn}"


def _has_four_in_a_row(board: np.ndarray, player: int) -> bool:
    """Vectorised check for four consecutive `player` cells in any direction."""
    b = (board == player).astype(np.int8)
    # Horizontal: sum over 4-wide windows in each row.
    if (b[:, :-3] + b[:, 1:-2] + b[:, 2:-1] + b[:, 3:] == CONNECT).any():
        return True
    # Vertical.
    if (b[:-3, :] + b[1:-2, :] + b[2:-1, :] + b[3:, :] == CONNECT).any():
        return True
    # Diagonal down-right (\).
    if (b[:-3, :-3] + b[1:-2, 1:-2] + b[2:-1, 2:-1] + b[3:, 3:] == CONNECT).any():
        return True
    # Diagonal down-left (/).
    if (b[:-3, 3:] + b[1:-2, 2:-1] + b[2:-1, 1:-2] + b[3:, :-3] == CONNECT).any():
        return True
    return False
