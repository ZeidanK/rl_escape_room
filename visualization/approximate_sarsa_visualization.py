"""Streamlit visualization helpers for Room 4 continuous policies."""

import numpy as np
import streamlit as st

from agents.approximate_sarsa import LinearTileQFunction
from core.types import (
    ApproximateEpisodeMetrics,
    ContinuousState,
)
from features.tile_coding import TileCoder, TileCodingConfig


# Visualization helpers for continuous Room 4.  They sample the learned linear
# Q-function over a grid so a continuous policy can be displayed in 2D.
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


def build_action_field(
    env,
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    fixed_vx: int = 0,
    fixed_vy: int = 0,
    grid_size: int = 10,
) -> np.ndarray:
    # For each sampled (x,y), fix velocity and record the greedy velocity
    # action.  This gives a policy-field approximation.
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
    # Sample max_a Q(s,a) over positions for a coarse value surface.
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
