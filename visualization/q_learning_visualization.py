from collections.abc import Mapping

import numpy as np

from core.types import Action, CellType, QLearningEpisodeMetrics, Room3State
from environments.grid_environment import CELL_TO_CHAR
from visualization.sarsa_visualization import (
    ARROW_SYMBOLS,
    render_sarsa_trajectory_overlay,
)


def build_room3_policy_symbols(
    environment,
    policy: Mapping[Room3State, Action | None],
    *,
    has_key: bool,
) -> list[list[str]]:
    env = environment
    rows, cols = 10, 10
    symbols: list[list[str]] = []
    for r in range(rows):
        row_symbols: list[str] = []
        for c in range(cols):
            cell = CellType(int(env.grid[r, c]))
            if cell == CellType.WALL:
                row_symbols.append("#")
                continue

            locked_exit = (r, c) == env.goal_position

            if locked_exit and has_key:
                row_symbols.append("E")
                continue

            state = (r, c, has_key)
            action = policy.get(state)

            if action is not None:
                base = ARROW_SYMBOLS[action]
            elif action is None and state in policy:
                base = "E"
            else:
                base = "·"

            prefix = ""
            if (r, c) == env.start_position and not has_key:
                prefix += "S"
            if (r, c) == env.key_position and not has_key:
                prefix += "K"
            if locked_exit and not has_key:
                prefix += "L"
            if cell == CellType.SLIPPERY:
                prefix += "I"
            if cell == CellType.TRAP:
                prefix += "T"

            row_symbols.append(prefix + base)
        symbols.append(row_symbols)
    return symbols


def build_room3_q_value_table(
    environment,
    q_values: Mapping[Room3State, tuple[float, ...]],
    row: int,
    column: int,
    has_key: bool,
) -> dict:
    state = (row, column, has_key)
    vals = q_values.get(state, (0.0, 0.0, 0.0, 0.0))
    actions_list = [
        {"action": "UP", "value": float(vals[0])},
        {"action": "RIGHT", "value": float(vals[1])},
        {"action": "DOWN", "value": float(vals[2])},
        {"action": "LEFT", "value": float(vals[3])},
    ]
    greedy_idx = int(np.argmax(vals))
    greedy_name = ["UP", "RIGHT", "DOWN", "LEFT"][greedy_idx]
    return {
        "state": str(state),
        "actions": actions_list,
        "greedy_action": greedy_name,
        "is_terminal": environment.is_terminal_state(state),
        "q_values_raw": vals,
    }


def build_q_learning_training_dataframe(
    metrics: tuple[QLearningEpisodeMetrics, ...],
) -> dict:
    keys = [
        "episode", "total_reward", "steps", "success", "terminated", "truncated",
        "epsilon", "key_collected", "key_collection_step", "locked_exit_attempts",
        "collision_count", "slipped_action_count", "trap_count",
        "mean_abs_td_error", "max_abs_td_error",
    ]
    data = {k: [] for k in keys}
    for m in metrics:
        data["episode"].append(m.episode)
        data["total_reward"].append(m.total_reward)
        data["steps"].append(m.steps)
        data["success"].append(int(m.success))
        data["terminated"].append(int(m.terminated))
        data["truncated"].append(int(m.truncated))
        data["epsilon"].append(m.epsilon)
        data["key_collected"].append(int(m.key_collected))
        data["key_collection_step"].append(m.key_collection_step)
        data["locked_exit_attempts"].append(m.locked_exit_attempts)
        data["collision_count"].append(m.collision_count)
        data["slipped_action_count"].append(m.slipped_action_count)
        data["trap_count"].append(m.trap_count)
        data["mean_abs_td_error"].append(m.mean_abs_td_error)
        data["max_abs_td_error"].append(m.max_abs_td_error)
    return data


def render_q_learning_trajectory_overlay(
    environment,
    rollout,
    *,
    has_key: bool,
) -> np.ndarray:
    positions = tuple(step.state for step in rollout.steps)
    return render_sarsa_trajectory_overlay(environment, positions)
