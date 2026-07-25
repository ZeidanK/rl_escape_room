from collections.abc import Mapping

import numpy as np
import streamlit as st

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


def render_trajectory_overlay(
    env: GridEnvironment,
    trajectory: tuple[Position, ...],
) -> np.ndarray:
    # Overlay one rollout path on top of the static map.
    rows, cols = env.grid_shape
    overlay = np.full((rows, cols), "", dtype=object)
    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(env.grid[r, c]))
            if cell == CellType.WALL:
                overlay[r, c] = "#"
            elif cell == CellType.START:
                overlay[r, c] = "S"
            elif env.is_terminal_state((r, c)):
                overlay[r, c] = "E"
            else:
                overlay[r, c] = "·"
    # Mark start of trajectory
    if trajectory:
        first = trajectory[0]
        if first != env.start_position:
            overlay[first] = "●"
        last = trajectory[-1]
        if env.is_terminal_state(last):
            overlay[last] = "★"
    # Mark visited cells
    visited_counts: dict[Position, int] = {}
    for pos in trajectory:
        visited_counts[pos] = visited_counts.get(pos, 0) + 1
    for pos, count in visited_counts.items():
        if count > 1 and pos != first:
            overlay[pos] = str(count)
        elif count == 1 and pos != first and pos != last:
            cell = CellType(int(env.grid[pos]))
            if not env.is_terminal_state(pos) and cell != CellType.START:
                overlay[pos] = "○"
    return overlay
