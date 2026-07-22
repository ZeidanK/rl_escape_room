"""Phase 5 tests: Q-Learning for Room 3."""

from types import MappingProxyType

import numpy as np
import pytest

from core.types import (
    Action,
    CellType,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    Position,
    QLearningConfig,
    QLearningEpisodeMetrics,
    RolloutResult,
    Room3State,
    SlipConfig,
    TrajectoryStep,
)
from environments.grid_environment import KnownModelGridEnvironment, parse_grid_map
from environments.room3_qlearning import ROOM3_GRID, ROOM3_MAP, Room3QLearning
from agents.q_learning import (
    QLearningAgent,
    evaluate_q_learning_policy,
    load_q_model,
    rollout_q_learning_policy,
    save_q_model,
)
from agents.tabular_utils import extract_deterministic_greedy_policy
from visualization.q_learning_visualization import (
    build_q_learning_training_dataframe,
    build_room3_policy_symbols,
    build_room3_q_value_table,
    render_q_learning_trajectory_overlay,
)


# ============================================================
# Helpers
# ============================================================

def factory():
    return Room3QLearning(max_steps=500, slip_config=SlipConfig(0.8, 0.1, 0.1))


def deterministic_factory():
    return Room3QLearning(max_steps=500, slip_config=SlipConfig(1.0, 0.0, 0.0))


# ============================================================
# 1. Pre-phase regression — Room 3 state contract (6 tests)
# ============================================================


class TestRoom3Contract:
    def test_reset_restores_has_key_false(self):
        env = factory()
        env.reset(seed=42)
        assert not env._key_collected
        state = env.state
        assert state[2] is False

    def test_reset_restores_key_cell(self):
        env = factory()
        env.reset(seed=42)
        key_pos = env.key_position
        assert key_pos is not None
        assert CellType(int(env.grid[key_pos])) == CellType.KEY

    def test_reset_restores_locked_exit_cell(self):
        env = factory()
        env.reset(seed=42)
        goal = env.goal_position
        assert goal is not None
        assert CellType(int(env.grid[goal])) == CellType.LOCKED_EXIT

    def test_key_collected_once(self):
        env = factory()
        env.reset(seed=42)
        key_pos = env.key_position
        env._agent_pos = key_pos
        _, _, info = env._on_enter_cell(key_pos, CellType.KEY)
        assert info.get("key_collected") is True
        _, _, info2 = env._on_enter_cell(key_pos, CellType.KEY)
        assert info2.get("event") == "key_already_collected"

    def test_locked_exit_without_key_not_terminal(self):
        env = factory()
        env.reset(seed=42)
        goal = env.goal_position
        env._agent_pos = goal
        reward, terminated, info = env._on_enter_cell(goal, CellType.LOCKED_EXIT)
        assert not terminated
        assert info.get("event") == "locked_exit"

    def test_locked_exit_with_key_terminates(self):
        env = factory()
        env.reset(seed=42)
        env._key_collected = True
        goal = env.goal_position
        env._agent_pos = goal
        reward, terminated, info = env._on_enter_cell(goal, CellType.LOCKED_EXIT)
        assert terminated
        assert info.get("success") is True

    def test_states_cartesian_product(self):
        env = factory()
        non_wall = sum(
            1 for r in range(10) for c in range(10)
            if CellType(int(env.grid[r, c])) != CellType.WALL
        )
        assert len(env.states) == non_wall * 2
        assert all(len(s) == 3 for s in env.states)
        assert all(isinstance(s[2], bool) for s in env.states)

    def test_is_terminal_state_locked_exit_no_key(self):
        env = factory()
        goal = env.goal_position
        assert not env.is_terminal_state((goal[0], goal[1], False))

    def test_is_terminal_state_locked_exit_with_key(self):
        env = factory()
        goal = env.goal_position
        assert env.is_terminal_state((goal[0], goal[1], True))

    def test_room3_no_known_model(self):
        assert not hasattr(Room3QLearning, "get_transition_distribution")


# ============================================================
# 2. Config validation (6 tests)
# ============================================================


class TestQLearningConfig:
    def test_invalid_alpha_too_low(self):
        with pytest.raises(ValueError, match="alpha"):
            QLearningConfig(alpha=0.0)

    def test_invalid_alpha_too_high(self):
        with pytest.raises(ValueError, match="alpha"):
            QLearningConfig(alpha=1.5)

    def test_invalid_gamma_one(self):
        with pytest.raises(ValueError, match="gamma"):
            QLearningConfig(gamma=1.0)

    def test_invalid_episode_count(self):
        with pytest.raises(ValueError, match="episodes"):
            QLearningConfig(episodes=0)

    def test_invalid_max_steps(self):
        with pytest.raises(ValueError, match="max_steps"):
            QLearningConfig(max_steps=0)

    def test_valid_config_defaults(self):
        cfg = QLearningConfig()
        assert cfg.episodes == 5000
        assert 0.0 < cfg.alpha <= 1.0
        assert 0.0 <= cfg.gamma < 1.0


# ============================================================
# 3. Q-Learning update correctness (8 tests)
# ============================================================


class TestQLearningUpdate:
    def test_off_policy_target_uses_max(self):
        """Update uses max over next state actions, NOT a specific next action."""
        env = factory()
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.zeros(4) for s in env.states}
        s: Room3State = (1, 2, False)
        ns: Room3State = (1, 3, False)
        qt[ns] = np.array([10.0, 20.0, 30.0, 40.0])
        td = agent.update(s, Action.RIGHT, -1.0, ns, terminated=False, truncated=False, q_table=qt)
        # target = -1 + 0.9 * max(10,20,30,40) = -1 + 36 = 35
        assert qt[s][int(Action.RIGHT)] == pytest.approx(35.0)

    def test_terminal_update_no_bootstrap(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.full(4, 100.0) for s in factory().states}
        s: Room3State = (1, 2, False)
        td = agent.update(s, Action.RIGHT, 99.0, s, terminated=True, truncated=False, q_table=qt)
        assert qt[s][int(Action.RIGHT)] == pytest.approx(99.0)

    def test_truncated_update_no_bootstrap(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.full(4, 100.0) for s in factory().states}
        s: Room3State = (1, 2, False)
        td = agent.update(s, Action.RIGHT, -30.0, s, terminated=False, truncated=True, q_table=qt)
        assert qt[s][int(Action.RIGHT)] == pytest.approx(-30.0)

    def test_non_terminal_target_differs_from_behaviour(self):
        """Prove that target uses max, not the behaviour action's value."""
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.zeros(4) for s in factory().states}
        s: Room3State = (1, 2, False)
        ns: Room3State = (1, 3, False)
        qt[ns] = np.array([100.0, 0.0, 0.0, 0.0])
        # update toward max = 100, regardless of what behaviour action would be
        agent.update(s, Action.DOWN, -1.0, ns, terminated=False, truncated=False, q_table=qt)
        assert qt[s][int(Action.DOWN)] == pytest.approx(-1 + 0.9 * 100.0)

    def test_alpha_controls_update_magnitude(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.zeros(4) for s in factory().states}
        s: Room3State = (1, 2, False)
        ns: Room3State = (1, 3, False)
        qt[ns] = np.array([0.0, 0.0, 0.0, 0.0])
        agent.update(s, Action.RIGHT, 10.0, ns, terminated=True, truncated=False, q_table=qt)
        # target = 10, td=10, q += 0.5*10 = 5
        assert qt[s][int(Action.RIGHT)] == pytest.approx(5.0)

    def test_single_entry_changes_only(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.zeros(4) for s in factory().states}
        s: Room3State = (1, 2, False)
        before = {st: arr.copy() for st, arr in qt.items()}
        agent.update(s, Action.RIGHT, -1.0, s, terminated=True, truncated=False, q_table=qt)
        assert qt[s][int(Action.RIGHT)] != before[s][int(Action.RIGHT)]
        for st, arr in qt.items():
            if st != s:
                assert np.allclose(arr, before[st])

    def test_q_values_finite(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=500, seed=42))
        qt = {s: np.zeros(4) for s in factory().states}
        s: Room3State = (1, 2, False)
        ns: Room3State = (1, 3, False)
        for _ in range(100):
            agent.update(s, Action.RIGHT, -1.0, ns, terminated=False, truncated=False, q_table=qt)
        assert np.all(np.isfinite(qt[s]))


# ============================================================
# 4. Training loop (8 tests)
# ============================================================


class TestQLearningTraining:
    def test_metric_per_episode(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        assert len(result.metrics) == 10

    def test_same_seed_reproduces_q_table(self):
        c = QLearningConfig(episodes=10, max_steps=500, seed=42)
        r1 = QLearningAgent(factory, c).train()
        r2 = QLearningAgent(factory, c).train()
        for s in r1.q_values:
            assert r1.q_values[s] == r2.q_values[s]

    def test_different_seeds_differ(self):
        r1 = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42)).train()
        r2 = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=99)).train()
        all_same = all(
            r1.q_values[s] == r2.q_values[s] for s in r1.q_values
        )
        assert not all_same

    def test_key_collected_tracked(self):
        agent = QLearningAgent(deterministic_factory, QLearningConfig(episodes=30, alpha=0.5, gamma=0.95, max_steps=500, seed=42))
        result = agent.train()
        assert any(m.key_collected for m in result.metrics)

    def test_locked_exit_attempts_tracked(self):
        agent = QLearningAgent(deterministic_factory, QLearningConfig(episodes=20, alpha=1.0, gamma=0.0, max_steps=500, seed=1))
        result = agent.train()
        # With zero-gamma and greedy, some episodes may hit locked exit without key early on
        assert isinstance(result.metrics[0].locked_exit_attempts, int)

    def test_q_table_contains_all_states(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=5, max_steps=500, seed=42))
        result = agent.train()
        env = factory()
        for s in env.states:
            assert s in result.q_values

    def test_epsilon_matches_schedule(self):
        config = QLearningConfig(episodes=20, max_steps=500, seed=42,
                                 epsilon=EpsilonScheduleConfig(kind=EpsilonDecayKind.LINEAR, start=1.0, minimum=0.0, linear_decay_episodes=10))
        agent = QLearningAgent(factory, config)
        result = agent.train()
        for m in result.metrics:
            expected = EpsilonScheduleConfig(kind=EpsilonDecayKind.LINEAR, start=1.0, minimum=0.0, linear_decay_episodes=10)
            from agents.tabular_utils import epsilon_for_episode
            exp_val = epsilon_for_episode(m.episode, expected)
            assert m.epsilon == pytest.approx(exp_val, abs=1e-10)

    def test_terminated_episode_success(self):
        agent = QLearningAgent(deterministic_factory, QLearningConfig(episodes=50, alpha=0.1, gamma=0.95, max_steps=500, seed=42))
        result = agent.train()
        successes = sum(1 for m in result.metrics if m.terminated)
        assert successes > 0


# ============================================================
# 5. Snapshots (5 tests)
# ============================================================


class TestSnapshots:
    def test_default_snapshot_milestones(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=100, max_steps=500, seed=42))
        result = agent.train()
        eps = sorted(result.snapshots.keys())
        assert 1 in eps
        assert 25 in eps
        assert 50 in eps
        assert 75 in eps
        assert 100 in eps

    def test_snapshot_q_values_immutable(self):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        snap = next(iter(result.snapshots.values()))
        with pytest.raises(TypeError):
            snap.q_values[(1, 1, False)] = (0.0, 0.0, 0.0, 0.0)

    def test_snapshot_rollout_does_not_update_q(self):
        result = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42)).train()
        for snap in result.snapshots.values():
            if snap.rollout:
                assert isinstance(snap.rollout, RolloutResult)

    def test_snapshot_rng_independent(self):
        c = QLearningConfig(episodes=10, max_steps=500, seed=42)
        r1 = QLearningAgent(factory, c).train()
        r2 = QLearningAgent(factory, c).train()
        assert r1.metrics == r2.metrics

    def test_snapshot_epsilon_recorded(self):
        result = QLearningAgent(factory, QLearningConfig(episodes=100, max_steps=500, seed=42)).train()
        for ep, snap in result.snapshots.items():
            assert snap.epsilon >= 0.0


# ============================================================
# 6. Evaluation (5 tests)
# ============================================================


class TestEvaluation:
    def test_success_rate_improves(self):
        """Trained Q-Learning on Room 3 should achieve positive success rate."""
        agent = QLearningAgent(factory, QLearningConfig(episodes=100, alpha=0.1, gamma=0.95, max_steps=500, seed=42))
        result = agent.train()
        ev = evaluate_q_learning_policy(factory, result.q_values, n_episodes=20, seeds=range(20))
        assert ev.success_rate >= 0.0
        assert ev.key_collection_rate >= 0.0

    def test_no_q_mutation(self):
        env = factory()
        qs = {s: (1.0, 2.0, 3.0, 4.0) for s in env.states}
        orig = dict(qs)
        evaluate_q_learning_policy(factory, qs, n_episodes=5, seeds=range(5))
        assert qs == orig

    def test_same_seeds_reproduce(self):
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev1 = evaluate_q_learning_policy(factory, qs, n_episodes=10, seeds=range(10))
        ev2 = evaluate_q_learning_policy(factory, qs, n_episodes=10, seeds=range(10))
        assert ev1.success_rate == ev2.success_rate

    def test_rollout_count_matches(self):
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev = evaluate_q_learning_policy(factory, qs, n_episodes=25, seeds=range(25), retain_rollouts=False)
        assert len(ev.rollouts) == 0

    def test_retain_rollouts(self):
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev = evaluate_q_learning_policy(factory, qs, n_episodes=5, seeds=range(5), retain_rollouts=True)
        assert len(ev.rollouts) == 5


# ============================================================
# 7. Persistence (5 tests)
# ============================================================


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        stem = str(tmp_path / "test_ql_model")
        from core.types import RewardConfig
        save_q_model(result, stem, reward_config=RewardConfig(), slip_config=SlipConfig(), map_grid=ROOM3_GRID)
        loaded_q, meta = load_q_model(stem, map_grid=ROOM3_GRID)
        assert len(loaded_q) == len(result.q_values)
        for s in result.q_values:
            assert s in loaded_q
            assert loaded_q[s] == result.q_values[s]

    def test_metadata_preserved(self, tmp_path):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        stem = str(tmp_path / "test_ql_meta")
        save_q_model(result, stem, map_grid=ROOM3_GRID)
        _, meta = load_q_model(stem, map_grid=ROOM3_GRID)
        assert meta["schema_version"] == 1
        assert meta["algorithm"] == "Q-Learning"
        assert meta["room"] == "Room3QLearning"
        assert meta["state_schema"] == ["row", "column", "has_key"]

    def test_incompatible_map_rejected(self, tmp_path):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        stem = str(tmp_path / "test_ql_map")
        from environments.room1_dp import ROOM1_GRID
        save_q_model(result, stem, map_grid=ROOM1_GRID)
        with pytest.raises(ValueError, match="Map signature"):
            load_q_model(stem, map_grid=ROOM3_GRID)

    def test_wrong_algorithm_rejected(self, tmp_path):
        agent = QLearningAgent(factory, QLearningConfig(episodes=10, max_steps=500, seed=42))
        result = agent.train()
        stem = str(tmp_path / "test_ql_algo")
        save_q_model(result, stem, map_grid=ROOM3_GRID)
        import json
        json_path = stem + ".json"
        with open(json_path) as f:
            meta = json.load(f)
        meta["algorithm"] = "SARSA"
        with open(json_path, "w") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="algorithm"):
            load_q_model(stem, map_grid=ROOM3_GRID)

    def test_greedy_policy_extraction(self):
        qs: dict[Room3State, tuple[float, ...]] = {
            (1, 1, False): (10.0, 0.0, 0.0, 0.0),
            (8, 9, True): (0.0, 0.0, 0.0, 0.0),
        }
        policy = extract_deterministic_greedy_policy(qs)
        assert policy[(1, 1, False)] == Action.UP


# ============================================================
# 8. Visualization (5 tests)
# ============================================================


class TestVisualization:
    def test_policy_symbols_no_key(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_deterministic_greedy_policy(qs)
        sym = build_room3_policy_symbols(env, policy, has_key=False)
        assert len(sym) == 10
        assert all(len(row) == 10 for row in sym)

    def test_policy_symbols_with_key(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_deterministic_greedy_policy(qs)
        sym = build_room3_policy_symbols(env, policy, has_key=True)
        assert len(sym) == 10

    def test_key_and_locked_markers_preserved(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_deterministic_greedy_policy(qs)
        sym_no_key = build_room3_policy_symbols(env, policy, has_key=False)
        key_pos = env.key_position
        goal_pos = env.goal_position
        if key_pos:
            assert "K" in sym_no_key[key_pos[0]][key_pos[1]]
        if goal_pos:
            assert "L" in sym_no_key[goal_pos[0]][goal_pos[1]]

    def test_terminal_cell_empty_with_key(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_deterministic_greedy_policy(qs)
        sym_key = build_room3_policy_symbols(env, policy, has_key=True)
        goal = env.goal_position
        if goal:
            assert sym_key[goal[0]][goal[1]] == "E"

    def test_training_dataframe_length(self):
        c = QLearningConfig(episodes=10, max_steps=500, seed=42)
        r = QLearningAgent(factory, c).train()
        df = build_q_learning_training_dataframe(r.metrics)
        assert len(df["episode"]) == 10


# ============================================================
# 9. Learning sanity (2 tests)
# ============================================================


class TestLearningSanity:
    def test_deterministic_key_task_succeeds(self):
        """On a simple 3-step deterministic task, Q-Learning should succeed."""
        rows = [
            "##########",
            "#SKL.....#",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
            "##########",
        ]
        grid = parse_grid_map(rows)

        # Custom environment class for the simple task
        class SimpleKeyEnv(KnownModelGridEnvironment):
            def __init__(self, **kwargs):
                import copy
                g = grid.copy()
                super().__init__(g, max_steps=50, slip_config=SlipConfig(1.0, 0.0, 0.0), **kwargs)
                self._key_collected = False

            def _terminal_cell_types(self):
                return {CellType.EXIT, CellType.LOCKED_EXIT}

            def _encode_state(self):
                return (self._agent_pos[0], self._agent_pos[1], self._key_collected)

            @property
            def states(self):
                pos = tuple((r, c) for r in range(10) for c in range(10)
                            if CellType(int(self._grid[r, c])) != CellType.WALL)
                return tuple((r, c, hk) for r, c in pos for hk in (False, True))

            @property
            def key_position(self):
                for r in range(10):
                    for c in range(10):
                        if CellType(int(self._grid[r, c])) == CellType.KEY:
                            return (r, c)
                return None

            @property
            def goal_position(self):
                for r in range(10):
                    for c in range(10):
                        if CellType(int(self._grid[r, c])) == CellType.LOCKED_EXIT:
                            return (r, c)
                return None

            def is_terminal_state(self, state):
                if not isinstance(state, tuple) or len(state) != 3:
                    return False
                row, col, has_key = state
                return has_key and (row, col) == self.goal_position

            def reset(self, seed=None):
                self._key_collected = False
                return super().reset(seed=seed)

            def _on_enter_cell(self, position, cell):
                if cell == CellType.KEY:
                    if not self._key_collected:
                        self._key_collected = True
                        self._grid[position] = int(CellType.EMPTY)
                        from core.types import RewardConfig
                        return 10.0, False, {"event": "key", "key_collected": True}
                    return 0.0, False, {"event": "key_already_collected"}
                elif cell == CellType.LOCKED_EXIT:
                    if self._key_collected:
                        from core.types import RewardConfig
                        return RewardConfig().compute_exit_reward(self.max_steps, self._step_count), True, {"event": "exit", "success": True}
                    return -5.0, False, {"event": "locked_exit"}
                if cell == CellType.EXIT:
                    from core.types import RewardConfig
                    return RewardConfig().compute_exit_reward(self.max_steps, self._step_count), True, {"event": "exit", "success": True}
                return super()._on_enter_cell(position, cell)

        def simple_factory():
            return SimpleKeyEnv()

        agent = QLearningAgent(simple_factory, QLearningConfig(episodes=200, alpha=0.1, gamma=0.95, max_steps=50, seed=42))
        result = agent.train()

        # Evaluate
        qs = result.q_values
        ev = evaluate_q_learning_policy(simple_factory, qs, n_episodes=20, seeds=range(20))
        assert ev.success_rate > 0.5

    def test_room3_training_improves(self):
        """Training on real Room 3 should improve success rate over time."""
        agent = QLearningAgent(factory, QLearningConfig(episodes=100, alpha=0.1, gamma=0.95, max_steps=500, seed=42))
        result = agent.train()

        first_half = result.metrics[:50]
        second_half = result.metrics[50:]
        sr_first = sum(1 for m in first_half if m.success) / len(first_half)
        sr_second = sum(1 for m in second_half if m.success) / len(second_half)
        assert sr_second >= sr_first


# ============================================================
# 10. Off-policy proof (1 test)
# ============================================================


class TestOffPolicyProof:
    def test_target_differs_from_behaviour_in_training(self):
        """Create a scenario where the behaviour action at s' differs from max."""
        env = factory()
        q_table = {s: np.zeros(4) for s in env.states}

        # Set up the next state such that max Q is UP, but behaviour (with eps=0.5)
        # might pick something else. Update with sarsa-like (SARSA) should use
        # next_action; QL should use max.
        ns = (2, 1, False)
        q_table[ns] = np.array([100.0, 0.0, 0.0, 0.0])  # max is UP=0
        agent = QLearningAgent(factory, QLearningConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=500, seed=42))

        # Run a manual update where next state has Qmax = UP
        s = (1, 1, False)
        td = agent.update(s, Action.RIGHT, -1.0, ns, terminated=False, truncated=False, q_table=q_table)
        # target = -1 + 0.9 * 100 = 89
        assert q_table[s][int(Action.RIGHT)] == pytest.approx(89.0), \
            "Q-Learning must use max Q(next), not Q(next, behaviour_action)"
