from collections.abc import Mapping, Sequence

import numpy as np

from core.types import Action, CellType, Position, TrainingEpisodeMetrics
from environments.grid_environment import GridEnvironment


ARROW_SYMBOLS: dict[Action, str] = {
    Action.UP: "↑",
    Action.RIGHT: "→",
    Action.DOWN: "↓",
    Action.LEFT: "←",
}

CELL_SYMBOLS: dict[CellType, str] = {
    CellType.WALL: "#",
    CellType.START: "S",
    CellType.EXIT: "E",
    CellType.SLIPPERY: "I",
    CellType.TRAP: "T",
    CellType.KEY: "K",
    CellType.LOCKED_EXIT: "L",
    CellType.EMPTY: "·",
}


def build_greedy_policy_symbols(
    env: GridEnvironment,
    q_values: Mapping[Position, tuple[float, ...]],
    greedy_policy: Mapping[Position, Action | None],
) -> list[list[str]]:
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
            elif cell == CellType.SLIPPERY:
                policy_action = greedy_policy.get((r, c))
                if policy_action is not None:
                    row_symbols.append("I" + ARROW_SYMBOLS[policy_action])
                else:
                    row_symbols.append("I·")
            elif cell == CellType.TRAP:
                policy_action = greedy_policy.get((r, c))
                if policy_action is not None:
                    row_symbols.append("T" + ARROW_SYMBOLS[policy_action])
                else:
                    row_symbols.append("T·")
            else:
                policy_action = greedy_policy.get((r, c))
                if policy_action is not None:
                    row_symbols.append(ARROW_SYMBOLS[policy_action])
                else:
                    row_symbols.append("·")
        symbols.append(row_symbols)
    return symbols


def build_q_value_tables(
    q_values: Mapping[Position, tuple[float, ...]],
) -> dict[Position, dict[str, float]]:
    return {
        state: {
            "UP": vals[0],
            "RIGHT": vals[1],
            "DOWN": vals[2],
            "LEFT": vals[3],
        }
        for state, vals in q_values.items()
    }


def build_training_dataframe(
    metrics: tuple[TrainingEpisodeMetrics, ...],
) -> dict:
    keys = [
        "episode", "total_reward", "steps", "success", "epsilon",
        "collision_count", "slipped_action_count", "trap_count",
        "mean_abs_td_error", "max_abs_td_error",
    ]
    data = {k: [] for k in keys}
    for m in metrics:
        data["episode"].append(m.episode)
        data["total_reward"].append(m.total_reward)
        data["steps"].append(m.steps)
        data["success"].append(int(m.success))
        data["epsilon"].append(m.epsilon)
        data["collision_count"].append(m.collision_count)
        data["slipped_action_count"].append(m.slipped_action_count)
        data["trap_count"].append(m.trap_count)
        data["mean_abs_td_error"].append(m.mean_abs_td_error)
        data["max_abs_td_error"].append(m.max_abs_td_error)
    return data


def render_sarsa_trajectory_overlay(
    env: GridEnvironment,
    trajectory: tuple[Position, ...],
) -> np.ndarray:
    rows, cols = env.grid_shape
    overlay = np.full((rows, cols), "", dtype=object)
    cell_chars = {int(k): v for k, v in CELL_SYMBOLS.items()}
    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(env.grid[r, c]))
            overlay[r, c] = cell_chars.get(int(cell), "?")
    if trajectory:
        first = trajectory[0]
        last = trajectory[-1]
        overlay[first] = "S"
        if env.is_terminal_state(last):
            overlay[last] = "★"
    visited: dict[Position, int] = {}
    for pos in trajectory:
        visited[pos] = visited.get(pos, 0) + 1
    for pos, count in visited.items():
        if pos == first or pos == last:
            continue
        if count > 1:
            overlay[pos] = str(min(count, 9))
        else:
            overlay[pos] = "○"
    return overlay
