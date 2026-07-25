"""Phase 3 tests: Value Iteration (DP) for Room 1."""

from types import MappingProxyType

import numpy as np
import pytest

from core.types import (
    Action,
    CellType,
    Position,
    RewardConfig,
    RolloutResult,
    SlipConfig,
    TrajectoryStep,
    ValueIterationConfig,
    ValueIterationResult,
)
from environments.grid_environment import KnownModelGridEnvironment, parse_grid_map
from environments.room1_dp import Room1DP
from agents.dynamic_programming import (
    ValueIterationAgent,
    evaluate_policy,
    rollout_policy,
)
from visualization.dp_visualization import (
    build_policy_symbols,
    build_value_matrix,
    render_trajectory_overlay,
)

# ============================================================
# Simple corridor map for deterministic tests
# ============================================================

CORRIDOR_MAP = [
    "##########",
    "#S......E#",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
    "##########",
]

CORRIDOR_GRID = parse_grid_map(CORRIDOR_MAP)


def corridor_env(**kwargs) -> KnownModelGridEnvironment:
    return KnownModelGridEnvironment(
        CORRIDOR_GRID.copy(),
        max_steps=200,
        slip_config=SlipConfig(1.0, 0.0, 0.0),
        **kwargs,
    )


# ============================================================
# 1. Public APIs (4 tests)
# ============================================================


class TestPublicAPIs:
    def test_states_excludes_walls(self):
        env = Room1DP()
        states = env.states
        for s in states:
            assert CellType(int(env.grid[s])) != CellType.WALL, f"State {s} is a wall"
        assert len(states) < env.rows * env.cols

    def test_states_includes_terminal(self):
        env = Room1DP()
        states = env.states
        goal = env.goal_position
        assert goal in states, "Goal must be in states"

    def test_states_immutable(self):
        env = Room1DP()
        states = env.states
        with pytest.raises(TypeError):
            states[0] = (0, 0)

    def test_actions_returns_all_four(self):
        env = Room1DP()
        assert env.actions == (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)

    def test_is_terminal_state_true_for_exit(self):
        env = Room1DP()
        goal = env.goal_position
        assert env.is_terminal_state(goal)

    def test_is_terminal_state_false_for_start(self):
        env = Room1DP()
        assert not env.is_terminal_state(env.start_position)


# ============================================================
# 2. Bellman correctness (5 tests)
# ============================================================


class TestBellmanCorrectness:
    def test_deterministic_one_step_exit(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        state = (1, 7)
        values = {s: 0.0 for s in env.states}
        q = agent.calculate_action_value(state, Action.RIGHT, values)
        expected = env.reward_config.step_penalty + env.reward_config.exit_reward
        assert q == pytest.approx(expected, rel=1e-10)

    def test_stochastic_expectation(self):
        slip = SlipConfig(0.8, 0.1, 0.1)
        env = Room1DP(slip_config=slip)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        # Slippery cell at (1,5): RIGHT has 80% intended, 10% left (UP), 10% right (DOWN)
        state = (1, 5)
        q = agent.calculate_action_value(state, Action.RIGHT, {s: 0.0 for s in env.states})
        # With zero values, the UP slip collides with a wall and matches step():
        # 0.8*(-1) + 0.1*(-4) + 0.1*(-1) = -1.3
        assert q == pytest.approx(-1.3, abs=1e-10)

    def test_terminal_no_bootstrap(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        state = (1, 7)
        outcomes = env.get_transition_distribution(state, Action.RIGHT)
        for o in outcomes:
            if o.terminated:
                q_via_outcome = o.probability * o.reward
                break
        q = agent.calculate_action_value(state, Action.RIGHT, {s: 0.0 for s in env.states})
        assert q == pytest.approx(q_via_outcome, rel=1e-10)

    def test_non_terminal_bootstraps(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        state = (1, 5)
        values = {s: 10.0 for s in env.states}
        q = agent.calculate_action_value(state, Action.RIGHT, values)
        expected = env.reward_config.step_penalty + 0.95 * 10.0
        assert q == pytest.approx(expected, rel=1e-10)

    def test_bellman_additive_rewards(self):
        trap_map = [
            "##########",
            "#S.T....E#",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ]
        env = KnownModelGridEnvironment(
            parse_grid_map(trap_map), slip_config=SlipConfig(1.0, 0.0, 0.0)
        )
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        state = (1, 2)
        values = {s: 5.0 for s in env.states}
        q = agent.calculate_action_value(state, Action.RIGHT, values)
        # Trap reward = step_penalty + trap_penalty + gamma * next_value
        expected = -1.0 + (-20.0) + 0.95 * 5.0
        assert q == pytest.approx(expected, rel=1e-10)


# ============================================================
# 3. Value Iteration (6 tests)
# ============================================================


class TestValueIteration:
    def test_converges_on_simple_map(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        assert result.converged
        assert result.iterations > 0

    def test_terminal_value_zero(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        result = agent.solve()
        for s in env.states:
            if env.is_terminal_state(s):
                assert result.values[s] == pytest.approx(0.0, abs=1e-10)

    def test_start_value_positive(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        result = agent.solve()
        assert result.start_state_value > 0.0

    def test_finite_values(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0))
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.99, tolerance=1e-6))
        result = agent.solve()
        for v in result.values.values():
            assert np.isfinite(v)

    def test_synchronous_sweep(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        assert len(result.delta_history) == result.iterations

    def test_non_convergence_flag(self):
        env = corridor_env()
        agent = ValueIterationAgent(
            env, ValueIterationConfig(gamma=0.95, tolerance=1e-20, max_iterations=5)
        )
        result = agent.solve()
        assert not result.converged
        assert result.iterations == 5


# ============================================================
# 4. Policy extraction (5 tests)
# ============================================================


class TestPolicyExtraction:
    def test_highest_action_selected(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        # On corridor, RIGHT should give highest Q-value for non-terminal
        policy = agent.extract_policy({s: 10.0 for s in env.states})
        for s in env.states:
            if not env.is_terminal_state(s):
                assert policy[s] is not None

    def test_tie_breaking_order(self):
        # On start of corridor at (1,1), wall actions are lower value (-4).
        # RIGHT is the first optimal non-collision move.
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tie_tolerance=1e-12))
        equal_values = {s: 0.0 for s in env.states}
        policy = agent.extract_policy(equal_values)
        assert policy[(1, 1)] == Action.RIGHT

    def test_terminal_policy_none(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        result = agent.solve()
        for s in env.states:
            if env.is_terminal_state(s):
                assert result.policy[s] is None

    def test_non_terminal_gets_action(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        result = agent.solve()
        non_terminal = [s for s in env.states if not env.is_terminal_state(s)]
        for s in non_terminal:
            assert result.policy[s] is not None, f"No policy for non-terminal {s}"

    def test_extract_policy_does_not_mutate(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95))
        values = {s: 10.0 for s in env.states}
        orig = dict(values)
        agent.extract_policy(values)
        assert values == orig


# ============================================================
# 5. Environment purity (1 test, 6 assertions)
# ============================================================


class TestEnvironmentPurity:
    def test_solve_does_not_mutate_env(self):
        env = Room1DP(slip_config=SlipConfig(), seed=42, max_steps=200)
        env.reset(seed=42)
        pos_before = env.agent_position
        step_before = env.step_count
        term_before = env._terminated
        trunc_before = env._truncated
        grid_before = env.grid.copy()
        rng_before = env.rng.bit_generator.state["state"]

        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        agent.solve()

        assert env.agent_position == pos_before, "agent_position changed"
        assert env.step_count == step_before, "step_count changed"
        assert env._terminated == term_before, "_terminated changed"
        assert env._truncated == trunc_before, "_truncated changed"
        assert (env.grid == grid_before).all(), "grid changed"
        assert env.rng.bit_generator.state["state"] == rng_before, "rng state changed"

    @pytest.mark.parametrize(
        ("state", "action"),
        [
            ((1, 0), Action.LEFT),  # boundary collision
            ((1, 0), Action.UP),    # wall collision
        ],
    )
    def test_known_model_collision_rewards_match_step(self, state, action):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0), seed=42, max_steps=200)
        model_outcome = env.get_transition_distribution(state, action)[0]

        env.reset(seed=42)
        env._agent_pos = state
        step_result = env.step(action)

        assert model_outcome.next_state == step_result.next_state == state
        assert model_outcome.reward == step_result.reward


# ============================================================
# 6. Rollout (5 tests)
# ============================================================


class TestRollout:
    def test_deterministic_reaches_exit(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        assert roll.success
        assert roll.terminated

    def test_trajectory_length_matches_steps(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        assert len(roll.steps) == roll.total_steps

    def test_cumulative_reward_matches(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        expected_reward = sum(s.reward for s in roll.steps)
        assert roll.total_reward == pytest.approx(expected_reward, rel=1e-10)

    def test_stops_on_termination(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        assert roll.terminated
        assert not roll.truncated

    def test_same_seed_same_trajectory(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        r1 = rollout_policy(env, result.policy, seed=42)
        r2 = rollout_policy(env, result.policy, seed=42)
        assert r1.total_steps == r2.total_steps
        assert r1.total_reward == pytest.approx(r2.total_reward, rel=1e-10)
        for s1, s2 in zip(r1.steps, r2.steps):
            assert s1.state == s2.state
            assert s1.effective_action == s2.effective_action


# ============================================================
# 7. Visualization (4 tests)
# ============================================================


class TestVisualization:
    def test_value_matrix_shape(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0))
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        vmat = build_value_matrix(env, result.values)
        assert vmat.shape == (10, 10)

    def test_value_matrix_walls_nan(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0))
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        vmat = build_value_matrix(env, result.values)
        for r in range(10):
            for c in range(10):
                if CellType(int(env.grid[r, c])) == CellType.WALL:
                    assert np.isnan(vmat[r, c]), f"Wall at ({r},{c}) is not NaN"

    def test_policy_symbols_shape(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0))
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        psym = build_policy_symbols(env, result.policy)
        assert len(psym) == 10
        assert all(len(row) == 10 for row in psym)

    def test_policy_goal_has_no_arrow(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0))
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        psym = build_policy_symbols(env, result.policy)
        goal = env.goal_position
        assert psym[goal[0]][goal[1]] == "E"

    def test_visualization_no_env_mutation(self):
        env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0), seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        grid_before = env.grid.copy()
        pos_before = env.agent_position
        build_value_matrix(env, result.values)
        assert (env.grid == grid_before).all()
        assert env.agent_position == pos_before
        build_policy_symbols(env, result.policy)
        assert (env.grid == grid_before).all()
        assert env.agent_position == pos_before


# ============================================================
# 8. Experiment runner (3 tests)
# ============================================================


class TestExperimentRunner:
    def test_correct_config_count(self):
        from training.dp_experiments import GAMMA_VALUES, TOLERANCE_VALUES, SLIP_CONFIGS
        expected = len(GAMMA_VALUES) * len(TOLERANCE_VALUES) * len(SLIP_CONFIGS)
        assert expected == 36

    def test_all_records_have_required_fields(self):
        from training.dp_experiments import run_room1_experiments
        results = run_room1_experiments()
        required = [
            "gamma", "tolerance", "slip_config", "converged", "iterations",
            "final_delta", "start_state_value", "success_rate", "mean_return",
            "mean_steps", "mean_successful_steps",
        ]
        for r in results:
            for field in required:
                assert field in r, f"Missing field {field} in result"
        assert len(results) == 36

    def test_deterministic_ranking(self):
        from training.dp_experiments import run_room1_experiments
        results = run_room1_experiments()
        # Best config should be deterministic, converged, high success
        best = results[0]
        assert best["converged"] is True


# ============================================================
# 9. Immutable result types (2 tests)
# ============================================================


class TestImmutability:
    def test_value_iteration_result_immutable(self):
        env = corridor_env()
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        with pytest.raises(TypeError):
            result.values[(1, 1)] = 100.0
        with pytest.raises(TypeError):
            result.policy[(1, 1)] = Action.UP

    def test_rollout_result_immutable(self):
        env = corridor_env(seed=42)
        agent = ValueIterationAgent(env, ValueIterationConfig(gamma=0.95, tolerance=1e-6))
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        with pytest.raises(AttributeError):
            roll.steps = ()


# ============================================================
# 10. ValueIterationConfig validation (3 tests)
# ============================================================


class TestValueIterationConfig:
    def test_gamma_must_be_less_than_one(self):
        with pytest.raises(ValueError, match="gamma"):
            ValueIterationConfig(gamma=1.0)

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ValueError, match="tolerance"):
            ValueIterationConfig(tolerance=0.0)

    def test_max_iterations_must_be_positive(self):
        with pytest.raises(ValueError, match="max_iterations"):
            ValueIterationConfig(max_iterations=0)
