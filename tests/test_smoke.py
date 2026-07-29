"""Phase 2 comprehensive grid-environment tests."""

import numpy as np
import pytest

from core.types import (
    Action,
    CellType,
    SlipConfig,
)
from environments.grid_environment import (
    GridEnvironment,
    KnownModelGridEnvironment,
    parse_grid_map,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import Room2SARSA
from environments.room3_qlearning import Room3QLearning

# ============================================================
# Map parsing and validation
# ============================================================

VALID_MAP = [
    "##########",
    "#S.......#",
    "#........#",
    "#........#",
    "#........#",
    "#........#",
    "#........#",
    "#........#",
    "#.......E#",
    "##########",
]


def test_parse_valid_map():
    grid = parse_grid_map(VALID_MAP)
    assert grid.shape == (10, 10)
    assert grid[1, 1] == CellType.START
    assert grid[8, 8] == CellType.EXIT


def test_parse_invalid_height():
    with pytest.raises(ValueError, match="10 rows"):
        parse_grid_map(["##########"] * 9)


def test_parse_invalid_width():
    with pytest.raises(ValueError, match="10 characters"):
        parse_grid_map(["#########"] + ["##########"] * 9)


def test_parse_unequal_rows():
    with pytest.raises(ValueError, match="10 characters"):
        parse_grid_map(["##########"] * 5 + ["#########"] * 5)


def test_parse_unknown_symbol():
    with pytest.raises(ValueError, match="Unknown map symbol"):
        lines = list(VALID_MAP)
        lines[3] = lines[3][:5] + "X" + lines[3][6:]
        parse_grid_map(lines)


def test_parse_missing_start():
    lines = [
        "##########",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#.......E#",
        "##########",
    ]
    with pytest.raises(ValueError, match="one START"):  # no START at all → grid env validates
        GridEnvironment(parse_grid_map(lines))


def test_parse_duplicate_start():
    lines = list(VALID_MAP)
    row = list(lines[3])
    row[3] = "S"
    lines[3] = "".join(row)
    with pytest.raises(ValueError, match="one START"):
        GridEnvironment(parse_grid_map(lines))


def test_parse_missing_exit():
    lines = [
        "##########",
        "#S.......#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "##########",
    ]
    with pytest.raises(ValueError, match="terminal cell"):
        GridEnvironment(parse_grid_map(lines))


def test_unreachable_exit():
    lines = [
        "##########",
        "#S......##",
        "##########",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#........#",
        "#.......E#",
        "##########",
    ]
    with pytest.raises(ValueError, match="reachable"):
        GridEnvironment(parse_grid_map(lines))


# ============================================================
# Reset
# ============================================================


def test_reset_returns_start():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    state = env.reset(seed=42)
    assert state == (1, 1)
    assert env.step_count == 0
    assert not env.is_done


def test_reset_clears_counters():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    env.step(Action.RIGHT)
    env.reset(seed=42)
    assert env.step_count == 0
    assert not env.is_done


def test_reset_seed_reproducibility():
    slip_map = [
        "##########",
        "#S.I.....#",
        "#.......E#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ]
    env = GridEnvironment(parse_grid_map(slip_map), slip_config=SlipConfig(), max_steps=100)
    env2 = GridEnvironment(parse_grid_map(slip_map), slip_config=SlipConfig(), max_steps=100)
    env.reset(seed=42)
    env2.reset(seed=42)
    # Move onto slippery cell (1,3)
    env.step(Action.RIGHT)  # (1,2)
    env2.step(Action.RIGHT)
    env.step(Action.RIGHT)  # (1,3) which is I
    env2.step(Action.RIGHT)
    for _ in range(10):
        r1 = env.step(Action.RIGHT)
        r2 = env2.step(Action.RIGHT)
        assert r1.info["effective_action"] == r2.info["effective_action"], (
            f"Seeded sequences diverged at step {env.step_count}"
        )


# ============================================================
# Movement
# ============================================================


def test_each_action_moves_correctly():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    cases = [(Action.RIGHT, (1, 2)), (Action.DOWN, (2, 1)),
             (Action.UP, True), (Action.LEFT, True)]
    for act, expected in cases:
        env.reset(seed=42)
        r = env.step(act)
        if isinstance(expected, bool) and expected:
            assert env.agent_position == (1, 1), f"{act.name} wall should stay at (1,1)"
        else:
            assert env.agent_position == expected, f"{act.name} should move to {expected}"
        assert not r.terminated
        assert not r.truncated


def test_boundary_collision():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    r = env.step(Action.UP)  # (0,1) is wall
    assert env.agent_position == (1, 1)
    assert r.info.get("collision") == "wall"
    r2 = env.step(Action.LEFT)  # (1,0) is wall
    assert env.agent_position == (1, 1)
    assert r2.info.get("collision") == "wall"


def test_wall_collision_additive_reward():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    r = env.step(Action.UP)
    expected = env.reward_config.step_penalty + env.reward_config.wall_penalty
    assert r.reward == expected, f"Expected {expected}, got {r.reward}"


def test_invalid_action_raises():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(99)


# ============================================================
# Exit and timeout
# ============================================================


def test_exit_terminates():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    # Teleport agent to cell next to exit
    env._agent_pos = (8, 7)
    r = env.step(Action.RIGHT)
    assert r.terminated
    assert not r.truncated
    assert r.info.get("event") == "exit"
    assert r.info.get("success") is True


def test_exit_reward_is_additive():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    env._agent_pos = (8, 7)
    r = env.step(Action.RIGHT)
    expected = env.reward_config.step_penalty + env.reward_config.compute_exit_reward(100, 1)
    assert r.reward == expected, f"Expected {expected}, got {r.reward}"


def test_timeout_truncates():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=3)
    env.reset(seed=42)
    for _ in range(3):
        r = env.step(Action.RIGHT)
        if env.is_done:
            break
    assert env.is_done
    assert not r.terminated
    assert r.truncated
    assert r.info.get("event") == "timeout"
    assert r.info.get("success") is False


def test_timeout_adds_penalty():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=1)
    env.reset(seed=42)
    r = env.step(Action.RIGHT)
    expected = env.reward_config.step_penalty + env.reward_config.step_limit_penalty
    assert r.reward == expected, f"Expected {expected}, got {r.reward}"


def test_step_after_done_raises():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=1)
    env.reset(seed=42)
    env.step(Action.RIGHT)
    assert env.is_done
    with pytest.raises(RuntimeError, match="Cannot call step"):
        env.step(Action.RIGHT)


# ============================================================
# Traps
# ============================================================


def test_traversable():
    trap_map = [
        "##########",
        "#S..T...E#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ]
    env = GridEnvironment(parse_grid_map(trap_map), max_steps=100)
    env.reset(seed=42)
    env._agent_pos = (1, 3)
    r = env.step(Action.RIGHT)
    assert env.agent_position == (1, 4)
    assert not r.terminated


def test_trap_penalty_additive():
    trap_map = [
        "##########",
        "#S..T...E#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ]
    env = GridEnvironment(parse_grid_map(trap_map), max_steps=100)
    env.reset(seed=42)
    env._agent_pos = (1, 3)
    r = env.step(Action.RIGHT)
    expected = env.reward_config.step_penalty + env.reward_config.trap_penalty
    assert r.reward == expected, f"Expected {expected}, got {r.reward}"
    assert r.info.get("event") == "trap"


# ============================================================
# Slippery cells
# ============================================================


def test_non_slippery_deterministic():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    r = env.step(Action.RIGHT)
    assert not r.info.get("slipped")
    assert r.info["requested_action"] == r.info["effective_action"]


def test_slippery_info_fields():
    slip_map = [
        "##########",
        "#S.I....E#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ]
    env = GridEnvironment(parse_grid_map(slip_map), slip_config=SlipConfig(), max_steps=100)
    env.reset(seed=42)
    # Move right twice to land on I at (1,3)
    env.step(Action.RIGHT)
    r = env.step(Action.RIGHT)
    assert "requested_action" in r.info
    assert "effective_action" in r.info
    assert "slipped" in r.info


def test_known_model_outcomes():
    """KnownModelGridEnvironment transition distribution sums to 1."""
    slip_map = [
        "##########",
        "#S.I....E#",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
        "##########",
    ]
    env = KnownModelGridEnvironment(parse_grid_map(slip_map), slip_config=SlipConfig(), max_steps=100)
    dist = env.get_transition_distribution((1, 3), Action.RIGHT)
    total = sum(o.probability for o in dist)
    assert abs(total - 1.0) < 1e-10, f"Probabilities sum to {total}, expected 1.0"
    # Non-slippery
    dist2 = env.get_transition_distribution((1, 1), Action.RIGHT)
    assert abs(sum(o.probability for o in dist2) - 1.0) < 1e-10


# ============================================================
# Encapsulation
# ============================================================


def test_input_map_immutable():
    original = parse_grid_map(VALID_MAP)
    env = GridEnvironment(original, max_steps=100)
    original[1, 1] = 99
    assert env.grid[1, 1] == CellType.START


def test_render_immutable():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    rs = env.render()
    rs.grid[1, 1] = 99
    assert env.grid[1, 1] == CellType.START


def test_room_separate_instances():
    r1 = Room1DP()
    r2 = Room1DP()
    assert r1.grid is not r2.grid
    assert np.allclose(r1.grid, r2.grid)


# ============================================================
# Room contracts
# ============================================================


def test_rooms_1_to_3_are_10x10():
    for cls, name in [(Room1DP, "Room1"), (Room2SARSA, "Room2"), (Room3QLearning, "Room3")]:
        env = cls(max_steps=200)
        assert env.grid_shape == (10, 10), f"{name} should be 10x10"


def test_room1_has_slippery_cells():
    env = Room1DP()
    assert np.any(env.grid == CellType.SLIPPERY), "Room1 must have slippery cells"


def test_room2_has_slippery_and_trap():
    env = Room2SARSA()
    assert np.any(env.grid == CellType.SLIPPERY), "Room2 must have slippery cells"
    assert np.any(env.grid == CellType.TRAP), "Room2 must have a trap"


def test_each_room_one_start_and_goal():
    for cls, name in [(Room1DP, "Room1"), (Room2SARSA, "Room2"), (Room3QLearning, "Room3")]:
        env = cls(max_steps=200)
        assert int(np.sum(env.grid == CellType.START)) == 1, f"{name} must have one START"
        goal_types = env._terminal_cell_types()
        count = sum(int(np.sum(env.grid == t)) for t in goal_types)
        assert count >= 1, f"{name} must have at least one terminal cell"


def test_room1_exposes_known_model():
    env = Room1DP()
    dist = env.get_transition_distribution((1, 1), Action.RIGHT)
    assert len(dist) >= 1
    assert abs(sum(o.probability for o in dist) - 1.0) < 1e-10


def test_room2_does_not_expose_known_model():
    env = Room2SARSA()
    assert not isinstance(env, KnownModelGridEnvironment)
    assert not hasattr(env, "get_transition_distribution")


# ============================================================
# Room 3 key mechanics
# ============================================================


def test_room3_state_has_key_flag():
    env = Room3QLearning(max_steps=300)
    state = env.reset(seed=42)
    assert len(state) == 3
    assert state[2] is False


def test_room3_key_collected_once():
    env = Room3QLearning(max_steps=300)
    env.reset(seed=42)
    # Navigate to key at (1,8)
    env._agent_pos = (1, 7)
    r1 = env.step(Action.RIGHT)
    assert r1.info.get("event") == "key"
    assert env._key_collected
    # Step again (cell is now empty)
    r2 = env.step(Action.LEFT)
    assert r2.info.get("event") != "key"
    # Make sure grid cell is now empty
    assert env.grid[1, 8] == CellType.EMPTY


def test_room3_cannot_exit_without_key():
    env = Room3QLearning(max_steps=300)
    env.reset(seed=42)
    # Teleport to locked exit
    env._agent_pos = (8, 8)
    r = env.step(Action.RIGHT)
    assert not r.terminated
    assert not r.truncated
    assert r.info.get("event") == "locked_exit"


def test_room3_exit_with_key():
    env = Room3QLearning(max_steps=300)
    env.reset(seed=42)
    env._key_collected = True
    env._agent_pos = (8, 8)
    r = env.step(Action.RIGHT)
    assert r.terminated
    assert r.info.get("event") == "exit"
    assert r.info.get("success") is True


# ============================================================
# Helper: render_ansi smoke
# ============================================================


def test_render_ansi_includes_agent():
    env = GridEnvironment(parse_grid_map(VALID_MAP), max_steps=100)
    env.reset(seed=42)
    ansi = env.render_ansi()
    assert "A" in ansi
    assert "#" in ansi


# ============================================================
# Regression: Room3QLearning.reset() restores key cell
# ============================================================


def test_room3_reset_restores_key_cell():
    env = Room3QLearning(max_steps=300)
    env.reset(seed=42)
    assert env.grid[1, 8] == CellType.KEY
    assert not env._key_collected
    # Navigate to cell left of key, collect it
    env._agent_pos = (1, 7)
    r = env.step(Action.RIGHT)
    assert r.info.get("event") == "key"
    assert env._key_collected
    assert env.grid[1, 8] == CellType.EMPTY
    # Reset and verify grid restored
    env.reset(seed=42)
    assert env.grid[1, 8] == CellType.KEY, "Reset must restore the key cell"
    assert not env._key_collected, "Reset must clear key_collected flag"


def test_room3_multiple_reset_key_preserved():
    """Key collection and reset can be cycled repeatedly."""
    env = Room3QLearning(max_steps=300)
    for _ in range(3):
        env.reset(seed=42)
        assert env.grid[1, 8] == CellType.KEY
        assert not env._key_collected
        env._agent_pos = (1, 7)
        env.step(Action.RIGHT)
        assert env.grid[1, 8] == CellType.EMPTY
        assert env._key_collected


# ============================================================
# Regression: get_transition_distribution purity
# ============================================================


def test_get_transition_distribution_does_not_mutate_env():
    env = Room1DP(slip_config=SlipConfig(), seed=42, max_steps=200)
    env.reset(seed=42)
    pos_before = env.agent_position
    step_before = env.step_count
    term_before = env._terminated
    trunc_before = env._truncated
    grid_before = env.grid.copy()
    rng_state_before = env.rng.bit_generator.state["state"]

    dist = env.get_transition_distribution(env.agent_position, Action.RIGHT)
    assert len(dist) >= 1

    assert env.agent_position == pos_before, "agent_position changed"
    assert env.step_count == step_before, "step_count changed"
    assert env._terminated == term_before, "_terminated changed"
    assert env._truncated == trunc_before, "_truncated changed"
    assert (env.grid == grid_before).all(), "grid changed"
    assert env.rng.bit_generator.state["state"] == rng_state_before, "rng state changed"


# ============================================================
# Phase 4 pre-checks: Room 2 SARSA
# ============================================================


def test_room2_has_slippery_and_trap():
    env = Room2SARSA(max_steps=200)
    assert np.any(env.grid == CellType.SLIPPERY), "Room2 must have slippery cells"
    assert np.any(env.grid == CellType.TRAP), "Room2 must have a trap"


def test_room2_reset_reproduces_stochastic_sequence():
    """Same seed must produce identical slip/collision outcomes."""
    env1 = Room2SARSA(max_steps=200, slip_config=SlipConfig(0.8, 0.1, 0.1))
    env2 = Room2SARSA(max_steps=200, slip_config=SlipConfig(0.8, 0.1, 0.1))
    env1.reset(seed=42)
    env2.reset(seed=42)
    for _ in range(20):
        r1 = env1.step(Action.RIGHT)
        r2 = env2.step(Action.RIGHT)
        assert r1.info["effective_action"] == r2.info["effective_action"], (
            f"Seeded sequences diverged at step {env1.step_count}"
        )
        if env1.is_done:
            break
