from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from core.types import Action, EpsilonDecayKind, EpsilonScheduleConfig, Position


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
    return MappingProxyType({
        s: tuple(float(v) for v in arr)
        for s, arr in q_table.items()
    })


def validate_q_table(
    q_values: Mapping[Any, tuple[float, ...]],
    *,
    expected_states: set[Any] | None = None,
    expected_action_count: int = 4,
) -> None:
    for state, vals in q_values.items():
        if len(vals) != expected_action_count:
            raise ValueError(
                f"State {state} has {len(vals)} actions; expected {expected_action_count}"
            )
        if not all(np.isfinite(vals)):
            raise ValueError(f"Non-finite Q-values at state {state}")
    if expected_states is not None:
        missing = expected_states - set(q_values.keys())
        if missing:
            raise ValueError(f"Missing states in Q-table: {missing}")
        extra = set(q_values.keys()) - expected_states
        if extra:
            raise ValueError(f"Unexpected states in Q-table: {extra}")


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
    import hashlib
    raw = grid.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]
