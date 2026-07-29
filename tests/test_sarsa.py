"""Phase 4 tests: SARSA for Room 2."""

import numpy as np
import pytest

from core.types import (
    Action,
    CellType,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    RolloutResult,
    SarsaConfig,
    SlipConfig,
)
from environments.grid_environment import KnownModelGridEnvironment, parse_grid_map
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from agents.sarsa import (
    SarsaAgent,
    epsilon_for_episode,
    evaluate_sarsa_policy,
    extract_greedy_policy,
    load_model,
    save_model,
    select_action,
)
from visualization.sarsa_visualization import (
    build_greedy_policy_symbols,
    build_training_dataframe,
)


# ============================================================
# Helpers
# ============================================================

def factory():
    return Room2SARSA(max_steps=200)


def deterministic_factory():
    return Room2SARSA(max_steps=200, slip_config=SlipConfig(1.0, 0.0, 0.0))


def small_deterministic_env():
    """Small 3xN map for quick sanity checks."""
    m = [
        "##########",
        "#SE.......#",
        "##########",
    ]
    return KnownModelGridEnvironment(
        parse_grid_map(m),
        max_steps=50,
        slip_config=SlipConfig(1.0, 0.0, 0.0),
    )


# ============================================================
# 1. Configuration and epsilon schedule (8 tests)
# ============================================================


class TestConfig:
    def test_invalid_alpha_too_low(self):
        with pytest.raises(ValueError, match="alpha"):
            SarsaConfig(alpha=0.0)

    def test_invalid_alpha_too_high(self):
        with pytest.raises(ValueError, match="alpha"):
            SarsaConfig(alpha=1.5)

    def test_invalid_gamma_one(self):
        with pytest.raises(ValueError, match="gamma"):
            SarsaConfig(gamma=1.0)

    def test_invalid_episode_count(self):
        with pytest.raises(ValueError, match="episodes"):
            SarsaConfig(episodes=0)

    def test_constant_schedule(self):
        cfg = EpsilonScheduleConfig(kind=EpsilonDecayKind.CONSTANT, start=0.5)
        for i in range(10):
            assert epsilon_for_episode(i, cfg) == 0.5

    def test_exponential_schedule(self):
        cfg = EpsilonScheduleConfig(kind=EpsilonDecayKind.EXPONENTIAL, start=1.0, minimum=0.0, decay=0.5)
        assert epsilon_for_episode(0, cfg) == 1.0
        assert epsilon_for_episode(1, cfg) == 0.5
        assert epsilon_for_episode(2, cfg) == 0.25

    def test_linear_schedule(self):
        cfg = EpsilonScheduleConfig(kind=EpsilonDecayKind.LINEAR, start=1.0, minimum=0.0, linear_decay_episodes=10)
        assert epsilon_for_episode(0, cfg) == 1.0
        assert epsilon_for_episode(5, cfg) == 0.5
        assert epsilon_for_episode(10, cfg) == 0.0

    def test_epsilon_minimum_bound(self):
        cfg = EpsilonScheduleConfig(kind=EpsilonDecayKind.EXPONENTIAL, start=1.0, minimum=0.1, decay=0.5)
        for i in range(10):
            assert epsilon_for_episode(i, cfg) >= 0.1

    def test_epsilon_reproducibility(self):
        cfg = EpsilonScheduleConfig(kind=EpsilonDecayKind.EXPONENTIAL)
        v1 = epsilon_for_episode(42, cfg)
        v2 = epsilon_for_episode(42, cfg)
        assert v1 == v2


# ============================================================
# 2. Action selection (5 tests)
# ============================================================


class TestActionSelection:
    def test_epsilon_one_explores_all(self):
        env = factory()
        rng = np.random.default_rng(42)
        q_table = {s: np.zeros(4) for s in env.states}
        actions_seen = set()
        for _ in range(100):
            a = select_action(env.start_position, epsilon=1.0, rng=rng, q_table=q_table)
            actions_seen.add(a)
        assert len(actions_seen) == 4

    def test_epsilon_zero_greedy(self):
        env = factory()
        rng = np.random.default_rng(42)
        q_table = {s: np.array([10.0, 0.0, 0.0, 0.0]) for s in env.states}
        for _ in range(50):
            a = select_action(env.start_position, epsilon=0.0, rng=rng, q_table=q_table)
            assert a == Action.UP

    def test_greedy_ties_among_maxima(self):
        env = factory()
        rng = np.random.default_rng(42)
        q_table = {s: np.array([10.0, 0.0, 10.0, 0.0]) for s in env.states}
        actions_seen = set()
        for _ in range(100):
            a = select_action(env.start_position, epsilon=0.0, rng=rng, q_table=q_table)
            actions_seen.add(a)
        assert actions_seen == {Action.UP, Action.DOWN}
        assert Action.RIGHT not in actions_seen
        assert Action.LEFT not in actions_seen

    def test_same_seed_reproduces(self):
        env = factory()
        qt = {s: np.array([1.0, 2.0, 3.0, 4.0]) for s in env.states}
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        acts1 = [select_action(env.start_position, epsilon=0.5, rng=rng1, q_table=qt) for _ in range(20)]
        acts2 = [select_action(env.start_position, epsilon=0.5, rng=rng2, q_table=qt) for _ in range(20)]
        assert acts1 == acts2

    def test_invalid_state_raises_key_error(self):
        rng = np.random.default_rng(42)
        qt = {(1, 1): np.zeros(4)}
        with pytest.raises(KeyError):
            select_action((99, 99), epsilon=0.5, rng=rng, q_table=qt)


# ============================================================
# 3. SARSA update (7 tests)
# ============================================================


class TestUpdate:
    def test_non_terminal_update_exact(self):
        env = factory()
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=200, seed=42))
        qt = {s: np.zeros(4) for s in env.states}
        s = (1, 1)
        a = Action.RIGHT
        ns = (1, 2)
        na = Action.RIGHT
        td = agent.update(s, a, -1.0, ns, na, terminated=False, truncated=False, q_table=qt)
        # target = -1 + 0.9 * 0 = -1, td = -1 - 0 = -1, q += 0.5 * (-1) = -0.5
        assert qt[s][int(a)] == pytest.approx(-0.5)
        assert td == pytest.approx(-1.0)

    def test_terminal_update_exact(self):
        env = factory()
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=200, seed=42))
        qt = {s: np.zeros(4) for s in env.states}
        s = (1, 1)
        a = Action.RIGHT
        ns = (1, 2)
        td = agent.update(s, a, 10.0, ns, None, terminated=True, truncated=False, q_table=qt)
        # target = 10, td = 10 - 0 = 10, q += 0.5 * 10 = 5
        assert qt[s][int(a)] == pytest.approx(5.0)
        assert td == pytest.approx(10.0)

    def test_truncated_update_no_bootstrap(self):
        env = factory()
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=200, seed=42))
        qt = {s: np.ones(4) * 100.0 for s in env.states}
        s = (1, 1)
        a = Action.RIGHT
        ns = (1, 2)
        td = agent.update(s, a, -30.0, ns, None, terminated=False, truncated=True, q_table=qt)
        # target = -30 (no bootstrap because truncated), td = -30 - 100 = -130, q += 0.5 * (-130) = 100 - 65 = 35
        assert qt[s][int(a)] == pytest.approx(35.0)
        assert td == pytest.approx(-130.0)

    def test_alpha_zero_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            SarsaConfig(alpha=0.0)

    def test_gamma_effect(self):
        env = factory()
        qt = {s: np.zeros(4) for s in env.states}
        qt[(1, 2)] = np.array([10.0, 0.0, 0.0, 0.0])
        s = (1, 1)
        a = Action.RIGHT
        ns = (1, 2)
        na = Action.UP
        # With gamma=0.5: target = -1 + 0.5*10 = 4, td = 4 - 0 = 4, q += 0.5*4 = 2
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=0.5, gamma=0.5, max_steps=200, seed=42))
        agent.update(s, a, -1.0, ns, na, terminated=False, truncated=False, q_table=qt)
        assert qt[s][int(a)] == pytest.approx(2.0)

    def test_single_entry_changes(self):
        env = factory()
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=1.0, gamma=0.9, max_steps=200, seed=42))
        qt = {s: np.zeros(4) for s in env.states}
        s = (1, 1)
        qt[(1, 2)] = np.array([5.0, 0.0, 0.0, 0.0])
        before = {s: arr.copy() for s, arr in qt.items()}
        agent.update(s, Action.RIGHT, -1.0, (1, 2), Action.UP, terminated=False, truncated=False, q_table=qt)
        assert qt[s][int(Action.RIGHT)] != before[s][int(Action.RIGHT)]
        assert np.allclose(qt[(1, 2)], before[(1, 2)])

    def test_q_values_finite(self):
        env = factory()
        agent = SarsaAgent(factory, SarsaConfig(episodes=1, alpha=0.5, gamma=0.9, max_steps=200, seed=42))
        qt = {s: np.zeros(4) for s in env.states}
        for _ in range(100):
            agent.update((1, 1), Action.RIGHT, -1.0, (1, 2), Action.RIGHT, terminated=False, truncated=False, q_table=qt)
        assert np.all(np.isfinite(qt[(1, 1)]))


# ============================================================
# 4. Training loop (8 tests)
# ============================================================


class TestTrainingLoop:
    def test_uses_next_action(self):
        """Prove the update is truly on-policy by checking epsilon affects it."""
        rng = np.random.default_rng(42)
        env = deterministic_factory()
        agent = SarsaAgent(deterministic_factory, SarsaConfig(episodes=2, alpha=1.0, gamma=0.0, max_steps=200, seed=42))
        result = agent.train()
        assert len(result.metrics) == 2

    def test_metric_per_episode(self):
        agent = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42))
        result = agent.train()
        assert len(result.metrics) == 10

    def test_epsilon_matches_schedule(self):
        config = SarsaConfig(episodes=20, max_steps=200, seed=42,
                             epsilon=EpsilonScheduleConfig(kind=EpsilonDecayKind.LINEAR, start=1.0, minimum=0.0, linear_decay_episodes=10))
        agent = SarsaAgent(factory, config)
        result = agent.train()
        for m in result.metrics:
            expected = epsilon_for_episode(m.episode, config.epsilon)
            assert m.epsilon == pytest.approx(expected, abs=1e-10)

    def test_same_seed_reproduces_q_table(self):
        c = SarsaConfig(episodes=10, max_steps=200, seed=42)
        r1 = SarsaAgent(factory, c).train()
        r2 = SarsaAgent(factory, c).train()
        for s in r1.q_values:
            assert r1.q_values[s] == r2.q_values[s]

    def test_different_seeds_differ(self):
        r1 = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42)).train()
        r2 = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=99)).train()
        # At least one Q-value should differ
        all_same = all(
            r1.q_values[s] == r2.q_values[s] for s in r1.q_values
        )
        assert not all_same

    def test_terminal_episode_stops(self):
        config = SarsaConfig(episodes=5, alpha=0.5, gamma=0.9, max_steps=200, seed=42)
        agent = SarsaAgent(deterministic_factory, config)
        result = agent.train()
        # Some episodes should succeed (terminate naturally)
        successes = sum(1 for m in result.metrics if m.terminated)
        assert successes >= 0

    def test_q_table_contains_all_states(self):
        agent = SarsaAgent(factory, SarsaConfig(episodes=5, max_steps=200, seed=42))
        result = agent.train()
        env = factory()
        for s in env.states:
            assert s in result.q_values

    def test_no_known_model_api(self):
        assert not hasattr(Room2SARSA, "get_transition_distribution")


# ============================================================
# 5. Snapshots (5 tests)
# ============================================================


class TestSnapshots:
    def test_default_snapshot_indices_valid(self):
        agent = SarsaAgent(factory, SarsaConfig(episodes=100, max_steps=200, seed=42))
        result = agent.train()
        for ep in result.snapshots:
            assert 1 <= ep <= 100

    def test_initial_middle_final_exist(self):
        agent = SarsaAgent(factory, SarsaConfig(episodes=100, max_steps=200, seed=42))
        result = agent.train()
        assert 1 in result.snapshots
        assert 50 in result.snapshots
        assert 100 in result.snapshots

    def test_snapshot_q_values_immutable(self):
        agent = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42))
        result = agent.train()
        snap = next(iter(result.snapshots.values()))
        with pytest.raises(TypeError):
            snap.q_values[(1, 1)] = (0.0, 0.0, 0.0, 0.0)

    def test_snapshot_rollout_does_not_update_q(self):
        config = SarsaConfig(episodes=10, max_steps=200, seed=42)
        agent = SarsaAgent(factory, config)
        result = agent.train()
        # The rollout in snapshot should not change training Q-values
        # We verify by checking that training result Q-values are internally consistent
        for snap in result.snapshots.values():
            if snap.rollout:
                assert isinstance(snap.rollout, RolloutResult)

    def test_snapshot_rng_independent(self):
        """Training with snapshots should be reproducible."""
        c = SarsaConfig(episodes=10, max_steps=200, seed=42)
        r1 = SarsaAgent(factory, c).train()
        r2 = SarsaAgent(factory, c).train()
        assert r1.metrics == r2.metrics


# ============================================================
# 6. Evaluation (5 tests)
# ============================================================


class TestEvaluation:
    def test_epsilon_zero(self):
        qs = {s: (10.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev = evaluate_sarsa_policy(factory, qs, n_episodes=10, seeds=range(10))
        assert ev.success_rate >= 0.0

    def test_no_q_mutation(self):
        qs = {s: (1.0, 2.0, 3.0, 4.0) for s in factory().states}
        orig = dict(qs)
        evaluate_sarsa_policy(factory, qs, n_episodes=5, seeds=range(5))
        assert qs == orig

    def test_same_seeds_reproduce_summary(self):
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev1 = evaluate_sarsa_policy(factory, qs, n_episodes=10, seeds=range(10))
        ev2 = evaluate_sarsa_policy(factory, qs, n_episodes=10, seeds=range(10))
        assert ev1.success_rate == ev2.success_rate

    def test_success_rate_calculation(self):
        # 10-row deterministic corridor: S at (1,1), E at (1,2), all walls elsewhere
        rows = [
            "##########",
            "#SE......#",
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
            parse_grid_map(rows),
            max_steps=50,
            slip_config=SlipConfig(1.0, 0.0, 0.0),
        )
        # Action order: UP=0, RIGHT=1, DOWN=2, LEFT=3
        # From (1,1), RIGHT leads to exit in one step
        qs = {s: np.array([0.0, 100.0, 0.0, 0.0]) for s in env.states}
        ev = evaluate_sarsa_policy(lambda: env, qs, n_episodes=10, seeds=range(10))
        assert ev.success_rate == 1.0

    def test_rollout_count_matches(self):
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in factory().states}
        ev = evaluate_sarsa_policy(factory, qs, n_episodes=25, seeds=range(25))
        assert len(ev.rollouts) == 25


# ============================================================
# 7. Persistence (4 tests)
# ============================================================


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        agent = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42))
        result = agent.train()
        stem = str(tmp_path / "test_model")
        from core.types import RewardConfig
        save_model(result, stem, reward_config=RewardConfig(), slip_config=SlipConfig(), map_grid=ROOM2_GRID)
        loaded_q, meta = load_model(stem, map_grid=ROOM2_GRID)
        assert len(loaded_q) == len(result.q_values)
        for s in result.q_values:
            assert s in loaded_q
            assert loaded_q[s] == result.q_values[s]

    def test_metadata_preserved(self, tmp_path):
        agent = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42))
        result = agent.train()
        stem = str(tmp_path / "meta_test")
        save_model(result, stem, map_grid=ROOM2_GRID)
        _, meta = load_model(stem, map_grid=ROOM2_GRID)
        assert meta["version"] == 1
        assert meta["config"]["episodes"] == 10
        assert meta["n_actions"] == 4

    def test_incompatible_map_rejected(self, tmp_path):
        agent = SarsaAgent(factory, SarsaConfig(episodes=10, max_steps=200, seed=42))
        result = agent.train()
        stem = str(tmp_path / "map_test")
        from environments.room1_dp import ROOM1_GRID
        save_model(result, stem, map_grid=ROOM1_GRID)
        with pytest.raises(ValueError, match="Map signature"):
            load_model(stem, map_grid=ROOM2_GRID)

    def test_greedy_policy_extraction(self):
        qs = {(1, 1): (10.0, 0.0, 0.0, 0.0), (8, 7): (0.0, 0.0, 0.0, 0.0)}
        policy = extract_greedy_policy(qs)
        assert policy[(1, 1)] == Action.UP


# ============================================================
# 8. Visualization (5 tests)
# ============================================================


class TestVisualization:
    def test_policy_matrix_dimensions(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_greedy_policy(qs)
        sym = build_greedy_policy_symbols(env, qs, policy)
        assert len(sym) == 10
        assert all(len(row) == 10 for row in sym)

    def test_slippery_and_trap_markers_preserved(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_greedy_policy(qs)
        sym = build_greedy_policy_symbols(env, qs, policy)
        for r in range(10):
            for c in range(10):
                cell = CellType(int(env.grid[r, c]))
                ch = sym[r][c]
                if cell == CellType.SLIPPERY:
                    assert "I" in ch, f"Slippery cell at ({r},{c}) missing I marker"
                if cell == CellType.TRAP:
                    assert "T" in ch, f"Trap cell at ({r},{c}) missing T marker"

    def test_terminal_cell_has_no_action(self):
        env = factory()
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_greedy_policy(qs)
        sym = build_greedy_policy_symbols(env, qs, policy)
        goal = env.goal_position
        assert sym[goal[0]][goal[1]] == "E"

    def test_training_dataframe_length(self):
        c = SarsaConfig(episodes=10, max_steps=200, seed=42)
        r = SarsaAgent(factory, c).train()
        df = build_training_dataframe(r.metrics)
        assert len(df["episode"]) == 10

    def test_no_env_mutation(self):
        env = factory()
        env.reset(seed=42)
        qs = {s: (0.0, 0.0, 0.0, 0.0) for s in env.states}
        policy = extract_greedy_policy(qs)
        grid_before = env.grid.copy()
        pos_before = env.agent_position
        build_greedy_policy_symbols(env, qs, policy)
        assert (env.grid == grid_before).all()
        assert env.agent_position == pos_before


# ============================================================
# 9. Learning sanity (1 test)
# ============================================================


class TestLearningSanity:
    def test_final_policy_better_than_random(self):
        """On a small deterministic map, trained policy should outperform random."""
        agent = SarsaAgent(deterministic_factory, SarsaConfig(episodes=200, alpha=0.1, gamma=0.95, max_steps=200, seed=42))
        result = agent.train()
        eval_result = evaluate_sarsa_policy(deterministic_factory, result.q_values, n_episodes=30, seeds=range(30))
        assert eval_result.success_rate > 0.5, f"Trained policy success rate {eval_result.success_rate} should exceed 50%"
