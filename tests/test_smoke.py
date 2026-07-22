"""Phase 1 smoke tests.

Verifies the scaffold is correctly set up before any algorithms are implemented.
"""
import numpy as np
import pytest

from core.types import RewardConfig, RoomSpec, RoomKind
from config.rooms import ROOM_SPECS
from environments.base_environment import BaseEnvironment


def test_config_objects_instantiate():
    rewards = RewardConfig()
    assert rewards.step_penalty == -1.0
    assert rewards.exit_reward == 100.0
    assert rewards.wall_penalty == -3.0
    assert rewards.trap_penalty == -20.0
    assert rewards.key_reward == 10.0
    assert rewards.locked_exit_penalty == -5.0
    assert rewards.step_limit_penalty == -30.0
    assert rewards.time_bonus_scale == 0.0


def test_exactly_four_room_specs():
    assert len(ROOM_SPECS) == 4


def test_correct_algorithms_assigned():
    assert "dynamic programming" in ROOM_SPECS["room1_dp"].algorithm.lower() or "value iteration" in ROOM_SPECS["room1_dp"].algorithm.lower()
    assert "sarsa" in ROOM_SPECS["room2_sarsa"].algorithm.lower()
    assert "q-learning" in ROOM_SPECS["room3_qlearning"].algorithm.lower() or "q learning" in ROOM_SPECS["room3_qlearning"].algorithm.lower()
    assert "function approximation" in ROOM_SPECS["room4_continuous"].algorithm.lower() or "tile" in ROOM_SPECS["room4_continuous"].algorithm.lower()


def test_rooms_1_3_are_10x10_grids():
    for room_id in ["room1_dp", "room2_sarsa", "room3_qlearning"]:
        spec = ROOM_SPECS[room_id]
        assert spec.grid_size == (10, 10)
        assert not spec.is_continuous


def test_room_4_is_continuous_and_10x10m():
    spec = ROOM_SPECS["room4_continuous"]
    assert spec.is_continuous
    assert spec.continuous_size == (10.0, 10.0)
    assert spec.grid_size is None


def test_room_4_dt_equals_002():
    assert ROOM_SPECS["room4_continuous"].dt == 0.02


def test_room_4_velocity_values():
    assert ROOM_SPECS["room4_continuous"].velocity_values == (-1, 0, 1)


def test_reward_defaults_valid():
    for room_id, spec in ROOM_SPECS.items():
        r = spec.rewards
        assert r.step_penalty <= 0.0
        assert r.exit_reward > 0.0
        assert r.wall_penalty <= 0.0
        assert r.trap_penalty <= r.step_penalty
        assert r.time_bonus_scale >= 0.0


def test_base_environment_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEnvironment()  # abstract class


def test_all_modules_import_cleanly():
    from core import types as _t
    from config import settings as _s, rooms as _r
    from environments import base_environment as _be, grid_environment as _ge
    from environments import room1_dp as _r1, room2_sarsa as _r2, room3_qlearning as _r3, room4_continuous as _r4
