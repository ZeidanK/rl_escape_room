import math

import numpy as np
import streamlit as st

from agents.approximate_sarsa import LinearTileQFunction
from core.types import (
    ApproximateEpisodeMetrics,
    ApproximateSarsaTrainingResult,
    ContinuousRolloutResult,
    ContinuousState,
    VELOCITY_BY_ACTION,
    VelocityAction,
)
from features.tile_coding import TileCoder, TileCodingConfig


@st.cache_data
def build_training_dataframe(
    metrics: tuple[ApproximateEpisodeMetrics, ...],
) -> dict[str, np.ndarray]:
    n = len(metrics)
    arr = {
        "episode": np.zeros(n, dtype=int),
        "total_reward": np.zeros(n, dtype=float),
        "steps": np.zeros(n, dtype=int),
        "success": np.zeros(n, dtype=bool),
        "terminated": np.zeros(n, dtype=bool),
        "truncated": np.zeros(n, dtype=bool),
        "epsilon": np.zeros(n, dtype=float),
        "collision_count": np.zeros(n, dtype=int),
        "distance_travelled_m": np.zeros(n, dtype=float),
        "final_distance_to_exit_m": np.zeros(n, dtype=float),
        "mean_abs_td_error": np.zeros(n, dtype=float),
        "max_abs_td_error": np.zeros(n, dtype=float),
    }
    for i, m in enumerate(metrics):
        arr["episode"][i] = m.episode
        arr["total_reward"][i] = m.total_reward
        arr["steps"][i] = m.steps
        arr["success"][i] = m.success
        arr["terminated"][i] = m.terminated
        arr["truncated"][i] = m.truncated
        arr["epsilon"][i] = m.epsilon
        arr["collision_count"][i] = m.collision_count
        arr["distance_travelled_m"][i] = m.distance_travelled_m
        arr["final_distance_to_exit_m"][i] = m.final_distance_to_exit_m
        arr["mean_abs_td_error"][i] = m.mean_abs_td_error
        arr["max_abs_td_error"][i] = m.max_abs_td_error
    return arr


def render_continuous_trajectory(
    env,
    rollout: ContinuousRolloutResult,
    *,
    max_arrows: int = 30,
) -> dict:
    room_w = env.motion.room_width_m
    room_h = env.motion.room_height_m
    grid_size = 20
    cell_w = room_w / grid_size
    cell_h = room_h / grid_size

    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    # Mark exit
    ex, ey = env.motion.exit_center
    er = env.motion.exit_radius_m
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            if (cx - ex) ** 2 + (cy - ey) ** 2 <= er ** 2:
                grid[row][col] = "E"

    # Mark start
    sx, sy = rollout.start_state[0], rollout.start_state[1]
    sr = int(sy / cell_h)
    sc = int(sx / cell_w)
    if 0 <= sr < grid_size and 0 <= sc < grid_size:
        grid[sr][sc] = "S"

    # Trajectory path
    for step in rollout.trajectory:
        x, y, _, _ = step.state
        r = int(y / cell_h)
        c = int(x / cell_w)
        if 0 <= r < grid_size and 0 <= c < grid_size:
            if grid[r][c] in (".", "S"):
                grid[r][c] = "*"

    # Direction arrows at sampled intervals
    arrow_interval = max(1, len(rollout.trajectory) // max_arrows)
    for idx, step in enumerate(rollout.trajectory):
        if idx % arrow_interval == 0:
            x, y, vx, vy = step.state
            r = int(y / cell_h)
            c = int(x / cell_w)
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r][c] = _arrow_symbol(vx, vy)

    # Mark collisions
    for step in rollout.trajectory:
        if step.collision:
            x, y, _, _ = step.state
            r = int(y / cell_h)
            c = int(x / cell_w)
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r][c] = "X"

    return {"grid": grid, "width": grid_size, "height": grid_size}


def _arrow_symbol(vx: int, vy: int) -> str:
    if vy == 1 and vx == 0:
        return "\u2191"
    elif vy == -1 and vx == 0:
        return "\u2193"
    elif vx == 1 and vy == 0:
        return "\u2192"
    elif vx == -1 and vy == 0:
        return "\u2190"
    elif vx == 1 and vy == 1:
        return "\u2197"
    elif vx == 1 and vy == -1:
        return "\u2198"
    elif vx == -1 and vy == 1:
        return "\u2196"
    elif vx == -1 and vy == -1:
        return "\u2199"
    return "*"


def build_action_field(
    env,
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    fixed_vx: int = 0,
    fixed_vy: int = 0,
    grid_size: int = 10,
) -> np.ndarray:
    tile_coder = TileCoder(tile_coding_config, room_width=env.motion.room_width_m, room_height=env.motion.room_height_m)
    n_actions = 9
    q_func = LinearTileQFunction(tile_coder, n_actions=n_actions)
    q_func._weights = weights.copy()
    field = np.zeros((grid_size, grid_size), dtype=int)
    for row in range(grid_size):
        for col in range(grid_size):
            x = (col + 0.5) * env.motion.room_width_m / grid_size
            y = (row + 0.5) * env.motion.room_height_m / grid_size
            state: ContinuousState = (x, y, fixed_vx, fixed_vy)
            av = q_func.action_values(state)
            max_val = np.max(av)
            tied = np.where(np.abs(av - max_val) < 1e-12)[0]
            field[row, col] = int(tied[0])
    return field


def build_value_surface(
    env,
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    fixed_vx: int = 0,
    fixed_vy: int = 0,
    grid_size: int = 20,
) -> np.ndarray:
    tile_coder = TileCoder(tile_coding_config, room_width=env.motion.room_width_m, room_height=env.motion.room_height_m)
    n_actions = 9
    q_func = LinearTileQFunction(tile_coder, n_actions=n_actions)
    q_func._weights = weights.copy()
    surface = np.zeros((grid_size, grid_size), dtype=float)
    for row in range(grid_size):
        for col in range(grid_size):
            x = (col + 0.5) * env.motion.room_width_m / grid_size
            y = (row + 0.5) * env.motion.room_height_m / grid_size
            state: ContinuousState = (x, y, fixed_vx, fixed_vy)
            av = q_func.action_values(state)
            surface[row, col] = float(np.max(av))
    return surface



