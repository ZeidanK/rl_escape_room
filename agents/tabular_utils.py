"""Shared epsilon schedules, action selection, and Q-table utilities."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from core.types import Action, EpsilonDecayKind, EpsilonScheduleConfig


# Shared helpers for tabular agents.  SARSA and Q-Learning both use epsilon
# schedules, epsilon-greedy action choice, and immutable Q-table snapshots.
def epsilon_for_episode(
    episode_index: int,
    config: EpsilonScheduleConfig,
) -> float:
    if config.kind == EpsilonDecayKind.CONSTANT:
        return config.start
    elif config.kind == EpsilonDecayKind.EXPONENTIAL:
        return max(config.minimum, config.start * (config.decay ** episode_index))
    elif config.kind == EpsilonDecayKind.LINEAR:
        fraction = min(1.0, episode_index / config.linear_decay_episodes)
        return config.start + (config.minimum - config.start) * fraction
    return config.start


def select_epsilon_greedy_action(
    state: Any,
    *,
    epsilon: float,
    rng: np.random.Generator,
    q_table: dict[Any, np.ndarray],
) -> Action:
    # Explore with probability epsilon; otherwise choose randomly among the
    # best tied actions so training is not biased toward action index 0.
    q_values = q_table[state]
    if rng.random() < epsilon:
        return Action(rng.integers(0, len(q_values)))
    max_val = np.max(q_values)
    tied = np.where(np.abs(q_values - max_val) < 1e-12)[0]
    return Action(int(rng.choice(tied)))


def extract_deterministic_greedy_policy(
    q_values: Mapping[Any, tuple[float, ...]],
) -> dict[Any, Action | None]:
    policy: dict[Any, Action | None] = {}
    for state, vals in q_values.items():
        arr = np.array(vals)
        if not np.all(np.isfinite(arr)):
            policy[state] = None
            continue
        max_val = np.max(arr)
        tied = np.where(np.abs(arr - max_val) < 1e-12)[0]
        policy[state] = Action(int(tied[0]))
    return policy


def freeze_q_table(
    q_table: Mapping[Any, np.ndarray],
) -> Mapping[Any, tuple[float, ...]]:
    # Convert mutable NumPy arrays into tuples so returned results cannot be
    # accidentally mutated by UI or evaluation code.
    return MappingProxyType({
        s: tuple(float(v) for v in arr)
        for s, arr in q_table.items()
    })


def default_snapshot_episodes(total: int) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("total must be positive")
    return tuple(sorted({
        1,
        max(1, round(total * 0.25)),
        max(1, round(total * 0.50)),
        max(1, round(total * 0.75)),
        total,
    }))


def map_signature(grid: np.ndarray) -> str:
    # Saved models store this short hash to detect when a Q-table was trained
    # on a different room layout.
    import hashlib
    raw = grid.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]
