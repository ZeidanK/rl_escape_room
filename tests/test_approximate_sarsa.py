import hashlib
import hmac
import math

import numpy as np
import pytest

from core.types import (
    ApproximateSarsaConfig,
    ContinuousRenderState,
    ContinuousRewardConfig,
    ContinuousRolloutResult,
    ContinuousState,
    ContinuousTrajectoryStep,
    EpsilonScheduleConfig,
    FIXED_UNSEEN_STARTS,
    Room4MotionConfig,
    StartMode,
    TileCodingConfig,
    VELOCITY_BY_ACTION,
    VelocityAction,
)
from environments.room4_continuous import Room4Continuous
from features.tile_coding import TileCoder
from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    LinearTileQFunction,
    evaluate_approximate_policy,
    evaluate_approximate_policy_all_categories,
    load_approximate_model,
    rollout_approximate_policy,
    save_approximate_model,
)

# ============================================================
# Test helpers
# ============================================================

_DEFAULT_MOTION = Room4MotionConfig()
_DEFAULT_REWARDS = ContinuousRewardConfig()


def _make_env(start_mode=StartMode.FIXED, seed=None, max_steps=750):
    return Room4Continuous(
        motion_config=_DEFAULT_MOTION,
        reward_config=_DEFAULT_REWARDS,
        max_steps=max_steps,
        start_mode=start_mode,
        seed=seed,
    )


def _make_tile_coder(config=None):
    if config is None:
        config = TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True)
    return TileCoder(config, room_width=10.0, room_height=10.0)


# ============================================================
# 1. Environment tests
# ============================================================

class TestRoom4Environment:
    def test_state_shape_and_types(self):
        env = _make_env()
        s = env.reset(seed=42)
        assert isinstance(s, tuple) and len(s) == 4
        x, y, vx, vy = s
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(vx, int)
        assert isinstance(vy, int)

    def test_nine_actions(self):
        env = _make_env()
        assert len(env.actions) == 9
        for a in env.actions:
            assert a in (int(v) for v in VelocityAction)

    def test_dt_is_0_02(self):
        assert _DEFAULT_MOTION.time_step_s == 0.02

    def test_cardinal_movement(self):
        env = _make_env()
        env.reset(seed=42)
        # Move EAST
        r = env.step(int(VelocityAction.EAST))
        x, y, vx, vy = r.next_state
        assert abs(x - 0.5 - 0.02) < 1e-10
        assert abs(y - 0.5) < 1e-10
        assert vx == 1 and vy == 0

    def test_diagonal_movement(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.step(int(VelocityAction.NORTH_EAST))
        x, y, vx, vy = r.next_state
        assert abs(x - 0.5 - 0.02) < 1e-10
        assert abs(y - 0.5 - 0.02) < 1e-10
        assert vx == 1 and vy == 1

    def test_stop_action(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.step(int(VelocityAction.STOP))
        x, y, vx, vy = r.next_state
        assert abs(x - 0.5) < 1e-10
        assert abs(y - 0.5) < 1e-10
        assert vx == 0 and vy == 0

    def test_boundary_clipping(self):
        env = _make_env()
        env.reset(seed=42)
        for _ in range(100):
            r = env.step(int(VelocityAction.WEST))
        x, y = r.next_state[0], r.next_state[1]
        assert x >= 0.0

    def test_boundary_velocity_zeroed(self):
        env = _make_env()
        env.reset(seed=42)
        for _ in range(100):
            r = env.step(int(VelocityAction.WEST))
        assert r.info.get("collision") == "boundary"

    def test_exit_detection(self):
        env = _make_env()
        env.reset(seed=42)
        # Teleport near exit for testing
        env.pos = np.array([9.4, 9.4], dtype=float)
        r = env.step(int(VelocityAction.NORTH_EAST))
        assert r.terminated
        assert r.info["event"] == "exit"
        assert r.info["success"]

    def test_timeout(self):
        env = _make_env(max_steps=10)
        env.reset(seed=42)
        for _ in range(10):
            r = env.step(int(VelocityAction.STOP))
        assert r.truncated
        assert r.info["event"] == "timeout"

    def test_reset_reproducibility(self):
        s1 = _make_env().reset(seed=42)
        s2 = _make_env().reset(seed=42)
        assert s1 == s2

    def test_custom_start_state_validated(self):
        env = _make_env()
        env.reset(seed=42, start_state=(5.0, 5.0, 0, 0))
        x, y, vx, vy = env.state
        assert abs(x - 5.0) < 1e-10
        assert abs(y - 5.0) < 1e-10

    def test_custom_start_inside_exit_rejected(self):
        env = _make_env()
        import pytest
        with pytest.raises(ValueError, match="inside exit radius"):
            env.reset(seed=42, start_state=(9.5, 9.5, 0, 0))

    def test_custom_start_outside_bounds_rejected(self):
        env = _make_env()
        import pytest
        with pytest.raises(ValueError, match="outside room bounds"):
            env.reset(seed=42, start_state=(15.0, 5.0, 0, 0))

    def test_render_returns_continuous_render_state(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.render()
        assert isinstance(r, ContinuousRenderState)
        assert isinstance(r.x, float)
        assert isinstance(r.y, float)
        assert isinstance(r.vx, int)
        assert isinstance(r.vy, int)
        assert isinstance(r.trajectory, tuple)
        assert len(r.trajectory) > 0

    def test_render_state_immutable(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.render()
        with pytest.raises(Exception):
            r.x = 99.0

    def test_render_trajectory_immutable(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.render()
        with pytest.raises(Exception):
            r.trajectory = ()

    def test_render_repeated_idempotent(self):
        env = _make_env()
        env.reset(seed=42)
        r1 = env.render()
        r2 = env.render()
        assert r1.x == r2.x
        assert r1.step_count == r2.step_count
        assert r1.trajectory == r2.trajectory

    def test_reward_components_exposed(self):
        env = _make_env()
        env.reset(seed=42)
        r = env.step(int(VelocityAction.EAST))
        assert "step_penalty" in r.info
        assert "boundary_penalty" in r.info
        assert "progress_reward" in r.info
        assert "exit_reward" in r.info
        assert "timeout_penalty" in r.info

    def test_random_lower_left_start(self):
        env = _make_env(start_mode=StartMode.RANDOM_LOWER_LEFT)
        s = env.reset(seed=42)
        x, y = s[0], s[1]
        assert 0.25 <= x <= 3.0
        assert 0.25 <= y <= 3.0
        assert s[2] == 0 and s[3] == 0

    def test_random_room_start_outside_exit(self):
        env = _make_env(start_mode=StartMode.RANDOM_ROOM)
        s = env.reset(seed=42)
        x, y = s[0], s[1]
        ex, ey = env.motion.exit_center
        er = env.motion.exit_radius_m
        assert (x - ex) ** 2 + (y - ey) ** 2 > er * er

    def test_step_after_done_raises(self):
        env = _make_env(max_steps=1)
        env.reset(seed=42)
        env.step(int(VelocityAction.STOP))
        import pytest
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(int(VelocityAction.STOP))


# ============================================================
# 2. Tile coder tests
# ============================================================

class TestTileCoder:
    def test_feature_count(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True))
        assert tc.feature_count == 4 * 4 * 4 * 3 * 3  # 576

    def test_exactly_one_active_per_tiling(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=8, tiles_x=10, tiles_y=10, include_velocity=True))
        feats = tc.active_features((5.0, 5.0, 0, 0))
        assert len(feats) == 8

    def test_deterministic_indexing(self):
        tc = _make_tile_coder()
        f1 = tc.active_features((5.0, 5.0, 0, 0))
        f2 = tc.active_features((5.0, 5.0, 0, 0))
        assert f1 == f2

    def test_no_out_of_range_indices(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True))
        feats = tc.active_features((5.0, 5.0, 1, -1))
        for f in feats:
            assert 0 <= f < tc.feature_count

    def test_nearby_states_overlap(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=8, tiles_x=10, tiles_y=10, include_velocity=True))
        f1 = tc.active_features((5.0, 5.0, 0, 0))
        f2 = tc.active_features((5.05, 5.05, 0, 0))
        overlap = len(set(f1) & set(f2))
        assert overlap > 0

    def test_distant_states_separate(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True))
        f1 = tc.active_features((0.5, 0.5, 0, 0))
        f2 = tc.active_features((9.5, 9.5, 0, 0))
        overlap = len(set(f1) & set(f2))
        assert overlap == 0

    def test_boundary_values(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=10, tiles_y=10, include_velocity=True))
        f1 = tc.active_features((0.0, 0.0, 0, 0))
        f2 = tc.active_features((10.0, 10.0, 0, 0))
        assert len(f1) == 4
        assert len(f2) == 4
        for f in f1 + f2:
            assert 0 <= f < tc.feature_count

    def test_velocity_categories(self):
        tc = _make_tile_coder()
        f1 = tc.active_features((5.0, 5.0, -1, -1))
        f2 = tc.active_features((5.0, 5.0, 0, 0))
        f3 = tc.active_features((5.0, 5.0, 1, 1))
        assert f1 != f2
        assert f1 != f3
        assert f2 != f3

    def test_no_duplicate_active(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True))
        feats = tc.active_features((3.0, 7.0, 1, -1))
        assert len(set(feats)) == len(feats)

    def test_same_state_same_features(self):
        tc = _make_tile_coder()
        for _ in range(10):
            f = tc.active_features((2.718, 3.141, 1, -1))
            assert len(f) == tc.config.num_tilings

    def test_feature_count_no_velocity(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=False))
        assert tc.feature_count == 4 * 4 * 4 * 1 * 1  # 64


# ============================================================
# 3. Linear approximator tests
# ============================================================

class TestLinearTileQFunction:
    def test_zero_initialization(self):
        tc = _make_tile_coder()
        q = LinearTileQFunction(tc, n_actions=9)
        assert q.weights.shape == (9, tc.feature_count)
        assert np.all(q.weights == 0)

    def test_rng_independence(self):
        tc = _make_tile_coder()
        q1 = LinearTileQFunction(tc, n_actions=9)
        q2 = LinearTileQFunction(tc, n_actions=9)
        assert np.allclose(q1.weights, q2.weights)

    def test_initial_values_zero(self):
        tc = _make_tile_coder()
        q = LinearTileQFunction(tc, n_actions=9)
        for a in VelocityAction:
            assert q.value((5.0, 5.0, 0, 0), a) == 0.0
        av = q.action_values((5.0, 5.0, 0, 0))
        assert np.all(av == 0.0)

    def test_action_values_callable(self):
        tc = _make_tile_coder()
        q = LinearTileQFunction(tc, n_actions=9)
        av = q.action_values((5.0, 5.0, 0, 0))
        assert av.shape == (9,)

    def test_update_affects_selected_action_only(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=1, tiles_x=2, tiles_y=2, include_velocity=True))
        q = LinearTileQFunction(tc, n_actions=9)
        v_before = q.value((5.0, 5.0, 0, 0), VelocityAction.EAST)
        q.update((5.0, 5.0, 0, 0), VelocityAction.EAST, 1.0)
        v_after = q.value((5.0, 5.0, 0, 0), VelocityAction.EAST)
        assert abs(v_after - v_before - 1.0) < 1e-10, f"Expected {v_before + 1.0}, got {v_after}"
        # Other actions unaffected
        v_other_before = q.value((5.0, 5.0, 0, 0), VelocityAction.NORTH)
        q.update((5.0, 5.0, 0, 0), VelocityAction.NORTH, 0.0)
        v_other_after = q.value((5.0, 5.0, 0, 0), VelocityAction.NORTH)
        assert abs(v_other_after - v_other_before) < 1e-10

    def test_finite_weights(self):
        tc = _make_tile_coder()
        q = LinearTileQFunction(tc, n_actions=9)
        assert np.all(np.isfinite(q.weights))

    def test_read_only_public_copy(self):
        tc = _make_tile_coder()
        q = LinearTileQFunction(tc, n_actions=9)
        w = q.weights
        # Modifying the copy should not affect internal weights
        old_internal = q.weights
        w[0, 0] = 999.0
        new_internal = q.weights
        assert np.allclose(old_internal, new_internal)


# ============================================================
# 4. Semi-gradient SARSA tests
# ============================================================

class TestSemiGradientSARSA:
    def test_exact_non_terminal_update(self):
        config = ApproximateSarsaConfig(
            episodes=1, alpha=1.0, gamma=0.9, max_steps=10, seed=42,
            epsilon=EpsilonScheduleConfig(start=0.0, minimum=0.0, decay=1.0, kind="constant"),
            tile_coding=TileCodingConfig(num_tilings=1, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        agent = ApproximateSarsaAgent(factory, config)
        result = agent.train()
        assert len(result.metrics) == 1

    def test_terminal_no_bootstrap(self):
        env = _make_env()
        state_before = env.reset(seed=42)
        tc = _make_tile_coder(TileCodingConfig(num_tilings=1, tiles_x=4, tiles_y=4, include_velocity=True))
        q = LinearTileQFunction(tc, n_actions=9)
        action = VelocityAction.STOP
        v_before = q.value(state_before, action)
        # Force exit
        env.pos = np.array([9.4, 9.4], dtype=float)
        r = env.step(int(action))
        # No bootstrap = target = reward only
        target = r.reward
        scaled = (target - v_before) * 0.1  # alpha=0.1, num_tilings=1
        q.update(state_before, action, scaled)
        v_after = q.value(state_before, action)
        assert abs(v_after - v_before - scaled) < 1e-10, f"Expected {v_before + scaled}, got {v_after}"

    def test_truncated_no_bootstrap(self):
        env = _make_env(max_steps=1)
        env.reset(seed=42)
        tc = _make_tile_coder(TileCodingConfig(num_tilings=1, tiles_x=4, tiles_y=4, include_velocity=True))
        q = LinearTileQFunction(tc, n_actions=9)
        v_before = q.value(env.state, VelocityAction.STOP)
        r = env.step(int(VelocityAction.STOP))
        assert r.truncated
        assert abs(r.reward - _DEFAULT_REWARDS.step - _DEFAULT_REWARDS.timeout) < 1e-10

    def test_next_behaviour_action_used(self):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=1.0, gamma=0.9, max_steps=5, seed=0,
            epsilon=EpsilonScheduleConfig(start=0.0, minimum=0.0, decay=1.0, kind="constant"),
            tile_coding=TileCodingConfig(num_tilings=1, tiles_x=2, tiles_y=2, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        agent = ApproximateSarsaAgent(factory, config)
        result = agent.train()
        assert len(result.metrics) == 2

    def test_alpha_normalization_by_tilings(self):
        tc = _make_tile_coder(TileCodingConfig(num_tilings=1, tiles_x=4, tiles_y=4, include_velocity=True))
        q = LinearTileQFunction(tc, n_actions=9)
        v_before = q.value((5.0, 5.0, 0, 0), VelocityAction.EAST)
        # With 1 tiling, each feature gets the full scaled_td_error
        q.update((5.0, 5.0, 0, 0), VelocityAction.EAST, 1.0)
        v_after = q.value((5.0, 5.0, 0, 0), VelocityAction.EAST)
        assert abs(v_after - v_before - 1.0) < 1e-10

    def test_reproducibility(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        r1 = ApproximateSarsaAgent(factory, config).train()
        r2 = ApproximateSarsaAgent(factory, config).train()
        assert np.allclose(r1.weights, r2.weights)
        assert len(r1.metrics) == len(r2.metrics)
        for m1, m2 in zip(r1.metrics, r2.metrics):
            assert m1.total_reward == m2.total_reward

    def test_snapshot_independence(self):
        config = ApproximateSarsaConfig(
            episodes=10, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            snapshot_episodes=(5,),
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        assert 5 in result.snapshots
        snap = result.snapshots[5]
        assert snap.rollout is not None
        assert snap.weights.shape == result.weights.shape

    def test_no_tabular_q_table(self):
        config = ApproximateSarsaConfig(
            episodes=1, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        assert not hasattr(result, "q_values")
        assert hasattr(result, "weights")

    def test_snapshot_one_episode_has_snapshot(self):
        config = ApproximateSarsaConfig(
            episodes=1, alpha=0.1, gamma=0.9, max_steps=5, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        assert 1 in result.snapshots

    def test_snapshot_includes_final_episode(self):
        config = ApproximateSarsaConfig(
            episodes=10, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        assert 10 in result.snapshots

    def test_snapshot_custom_final_accepted(self):
        config = ApproximateSarsaConfig(
            episodes=10, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            snapshot_episodes=(10,),
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        assert 10 in result.snapshots

    def test_snapshot_does_not_alter_reproducibility(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        r_nosnap = ApproximateSarsaAgent(factory, config).train()
        config2 = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            snapshot_episodes=(1, 3, 5),
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        r_snap = ApproximateSarsaAgent(factory, config2).train()
        assert np.allclose(r_nosnap.weights, r_snap.weights)


# ============================================================
# 5. Evaluation tests
# ============================================================

class TestApproximateEvaluation:
    def test_fixed_start_metrics(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        ev = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=3, start_mode=StartMode.FIXED,
        )
        assert ev.n_episodes == 3
        assert 0 <= ev.success_rate <= 1.0

    def test_no_weight_mutation(self):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        w_before = result.weights.copy()
        evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=2, start_mode=StartMode.FIXED,
        )
        assert np.allclose(result.weights, w_before)

    def test_random_start_reproducibility(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        seeds = range(5)
        ev1 = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=5, seeds=seeds, start_mode=StartMode.FIXED,
        )
        ev2 = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=5, seeds=seeds, start_mode=StartMode.FIXED,
        )
        assert abs(ev1.success_rate - ev2.success_rate) < 1e-10
        assert abs(ev1.mean_return - ev2.mean_return) < 1e-10

    def test_rollout_retention(self):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        ev = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=3, start_mode=StartMode.FIXED,
        )
        assert len(ev.rollouts) == 3

    def test_generalization_summary(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        ev = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=3, start_mode=StartMode.RANDOM_LOWER_LEFT,
        )
        assert hasattr(ev, "start_category")
        assert ev.start_category == StartMode.RANDOM_LOWER_LEFT.value

    def test_unseen_starts_are_valid(self):
        for start in FIXED_UNSEEN_STARTS:
            sx, sy, svx, svy = start
            assert 0.0 <= sx <= 10.0
            assert 0.0 <= sy <= 10.0
            ex, ey = _DEFAULT_MOTION.exit_center
            er = _DEFAULT_MOTION.exit_radius_m
            assert (sx - ex) ** 2 + (sy - ey) ** 2 > er * er

    def test_multi_category_evaluation_all_keys_present(self):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=10, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        cats = evaluate_approximate_policy_all_categories(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=5,
        )
        for key in ("fixed_training_start", "fixed_unseen_starts",
                     "random_lower_left", "random_room"):
            assert key in cats, f"Missing category: {key}"
        for start in FIXED_UNSEEN_STARTS:
            label = f"unseen_start_{start[0]}_{start[1]}_{start[2]}_{start[3]}".replace(".", "_")
            assert label in cats, f"Missing per-start result: {label}"

    def test_multi_category_evaluation_reproducible(self):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=10, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        weights_copy = result.weights.copy()
        cats1 = evaluate_approximate_policy_all_categories(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=5,
        )
        cats2 = evaluate_approximate_policy_all_categories(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=5,
        )
        for key in cats1:
            if key.startswith("unseen_start_"):
                continue
            e1, e2 = cats1[key], cats2[key]
            assert abs(e1.success_rate - e2.success_rate) < 1e-10
            assert abs(e1.mean_return - e2.mean_return) < 1e-10
        assert np.allclose(result.weights, weights_copy), "weights mutated"


# ============================================================
# 6. Persistence tests
# ============================================================

class TestApproximatePersistence:
    def test_save_load_round_trip(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_model")
        save_approximate_model(result, stem)
        loaded_weights, meta = load_approximate_model(stem)
        assert np.allclose(result.weights, loaded_weights)
        assert meta["algorithm"] == "Semi-gradient SARSA with tile coding"
        assert meta["room"] == "Room4Continuous"

    def test_metadata_preserved(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=5, seed=42,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_meta")
        save_approximate_model(result, stem)
        _, meta = load_approximate_model(stem)
        assert meta["training_seed"] == 42
        assert meta["feature_count"] > 0
        assert meta["action_count"] == 9
        assert meta["state_schema"] == ["X", "Y", "Vx", "Vy"]

    def test_incompatible_tile_config_rejected(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_bad_tc")
        save_approximate_model(result, stem)
        bad_tc = TileCodingConfig(num_tilings=8, tiles_x=10, tiles_y=10, include_velocity=True)
        import pytest
        with pytest.raises(ValueError, match="Tile coding mismatch"):
            load_approximate_model(stem, expected_tile_coding=bad_tc)

    def test_wrong_weight_shape_rejected(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_shape")
        save_approximate_model(result, stem)
        # Tamper with metadata
        import json
        meta_path = stem + ".json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["action_count"] = 5
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        import pytest
        with pytest.raises(ValueError, match="Expected 9 actions"):
            load_approximate_model(stem)

    def test_non_finite_weights_rejected(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_nan")
        save_approximate_model(result, stem)
        # Tamper with npz — also update checksum to avoid checksum rejection
        npz_path = stem + ".npz"
        json_path = stem + ".json"
        loaded = np.load(npz_path)
        bad_w = loaded["weights"].copy()
        bad_w[0, 0] = float("nan")
        np.savez_compressed(npz_path, weights=bad_w)
        # Update checksum in metadata to prevent checksum mismatch
        import json
        with open(json_path) as f:
            meta = json.load(f)
        sha = hashlib.sha256()
        with open(npz_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        meta["weights_sha256"] = sha.hexdigest()
        with open(json_path, "w") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="Non-finite weights"):
            load_approximate_model(stem)

    def test_checksum_round_trip(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=3, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_checksum")
        save_approximate_model(result, stem)
        _, meta = load_approximate_model(stem)
        assert "weights_sha256" in meta
        assert len(meta["weights_sha256"]) == 64

    def test_checksum_rejects_modified_npz(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_mod_npz")
        save_approximate_model(result, stem)
        npz_path = stem + ".npz"
        loaded = np.load(npz_path)
        bad_w = loaded["weights"].copy()
        bad_w[0, 0] = 999.0
        np.savez_compressed(npz_path, weights=bad_w)
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_approximate_model(stem)

    def test_checksum_rejects_modified_checksum(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_mod_cs")
        save_approximate_model(result, stem)
        import json
        json_path = stem + ".json"
        with open(json_path) as f:
            meta = json.load(f)
        meta["weights_sha256"] = "0" * 64
        with open(json_path, "w") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_approximate_model(stem)

    def test_tmp_files_not_valid_model(self, tmp_path):
        config = ApproximateSarsaConfig(
            episodes=2, alpha=0.1, gamma=0.9, max_steps=5, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        stem = str(tmp_path / "test_tmp")
        save_approximate_model(result, stem)
        # No .tmp files remain after successful save
        import glob
        remaining = glob.glob(str(tmp_path / "*.tmp"))
        assert len(remaining) == 0


# ============================================================
# 7. Visualization tests (basic structural)
# ============================================================

class TestApproximateVisualization:
    def test_action_field_shape(self):
        config = TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True)
        tc = _make_tile_coder(config)
        q = LinearTileQFunction(tc, n_actions=9)
        from visualization.approximate_sarsa_visualization import build_action_field
        env = _make_env()
        field = build_action_field(env, q.weights, config, fixed_vx=0, fixed_vy=0, grid_size=5)
        assert field.shape == (5, 5)

    def test_value_surface_shape(self):
        config = TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True)
        tc = _make_tile_coder(config)
        q = LinearTileQFunction(tc, n_actions=9)
        from visualization.approximate_sarsa_visualization import build_value_surface
        env = _make_env()
        surface = build_value_surface(env, q.weights, config, fixed_vx=0, fixed_vy=0, grid_size=10)
        assert surface.shape == (10, 10)
        assert np.all(np.isfinite(surface))

    def test_training_dataframe(self):
        config = ApproximateSarsaConfig(
            episodes=5, alpha=0.1, gamma=0.9, max_steps=10, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
        )
        factory = lambda: _make_env()
        result = ApproximateSarsaAgent(factory, config).train()
        from visualization.approximate_sarsa_visualization import build_training_dataframe
        df = build_training_dataframe(result.metrics)
        assert len(df["episode"]) == 5
        assert "total_reward" in df
        assert "steps" in df
        assert "success" in df

    def test_no_env_mutation(self):
        config = TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True)
        tc = _make_tile_coder(config)
        q = LinearTileQFunction(tc, n_actions=9)
        from visualization.approximate_sarsa_visualization import build_action_field, build_value_surface
        env = _make_env()
        s_before = env.state
        build_action_field(env, q.weights, config, fixed_vx=0, fixed_vy=0, grid_size=5)
        assert env.state == s_before
        build_value_surface(env, q.weights, config, fixed_vx=0, fixed_vy=0, grid_size=5)
        assert env.state == s_before


# ============================================================
# 8. Learning sanity test
# ============================================================

class TestApproximateLearningSanity:
    def test_trained_outperforms_random(self):
        config = ApproximateSarsaConfig(
            episodes=20, alpha=0.2, gamma=0.99, max_steps=50, seed=42,
            epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.05, decay=0.9),
            tile_coding=TileCodingConfig(num_tilings=4, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env(start_mode=StartMode.FIXED)
        agent = ApproximateSarsaAgent(factory, config)
        result = agent.train()

        # Compare first vs last 5 episodes
        early_rewards = [m.total_reward for m in result.metrics[:5]]
        late_rewards = [m.total_reward for m in result.metrics[-5:]]
        assert np.mean(late_rewards) > np.mean(early_rewards), (
            f"Late rewards {np.mean(late_rewards):.1f} not > early {np.mean(early_rewards):.1f}"
        )

    def test_seeded_non_flaky(self):
        config = ApproximateSarsaConfig(
            episodes=10, alpha=0.1, gamma=0.9, max_steps=20, seed=0,
            tile_coding=TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4, include_velocity=True),
            start_mode=StartMode.FIXED,
        )
        factory = lambda: _make_env()
        r1 = ApproximateSarsaAgent(factory, config).train()
        r2 = ApproximateSarsaAgent(factory, config).train()
        assert np.allclose(r1.weights, r2.weights)
        for m1, m2 in zip(r1.metrics, r2.metrics):
            assert m1.total_reward == m2.total_reward
