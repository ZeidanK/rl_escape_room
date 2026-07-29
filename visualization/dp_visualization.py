"""Streamlit visualization helpers for Value Iteration outputs."""

from collections.abc import Mapping

import numpy as np

from core.types import Action, CellType, Position
from environments.grid_environment import GridEnvironment


# Helpers that translate DP outputs into table/grid formats for Streamlit.
ARROW_SYMBOLS: dict[Action, str] = {
    Action.UP: "↑",
    Action.RIGHT: "→",
    Action.DOWN: "↓",
    Action.LEFT: "←",
}


def build_value_matrix(
    env: GridEnvironment,
    values: Mapping[Position, float],
) -> np.ndarray:
    # Convert sparse value mapping into a 10x10 matrix while leaving walls as
    # NaN so they render differently in tables/heatmaps.
    rows, cols = env.grid_shape
    matrix = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(env.grid[r, c]))
            if cell == CellType.WALL:
                continue
            pos = (r, c)
            if pos in values:
                matrix[r, c] = values[pos]
            else:
                matrix[r, c] = 0.0
    return matrix


def build_policy_symbols(
    env: GridEnvironment,
    policy: Mapping[Position, Action | None],
) -> list[list[str]]:
    # Human-readable policy grid: arrows for actions, S/E/# for special cells.
    rows, cols = env.grid_shape
    symbols: list[list[str]] = []
    for r in range(rows):
        row_symbols: list[str] = []
        for c in range(cols):
            cell = CellType(int(env.grid[r, c]))
            if cell == CellType.WALL:
                row_symbols.append("#")
            elif cell == CellType.START:
                row_symbols.append("S")
            elif env.is_terminal_state((r, c)):
                row_symbols.append("E")
            else:
                pos = (r, c)
                action = policy.get(pos)
                if action is not None:
                    sym = ARROW_SYMBOLS[action]
                else:
                    sym = "·"
                row_symbols.append(sym)
        symbols.append(row_symbols)
    return symbols
