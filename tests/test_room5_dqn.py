"""Room 5 dynamic-obstacle environment and NumPy DQN regressions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from agents.dqn import (
    DQNAgent,
    DQNNetwork,
    ReplayBuffer,
    evaluate_dqn_policy,
    load_dqn_model,
    rollout_dqn_policy,
    save_dqn_model,
)
from core.types import (
    DQNConfig,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    Room4MotionConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
    VelocityAction,
)
from environments.room5_obstacles import Room5Obstacles


def _tiny_dqn_config(seed: int = 7, episodes: int = 4) -> DQNConfig:
    return DQNConfig(
        episodes=episodes,
        learning_rate=0.005,
        gamma=0.95,
        max_steps=12,
        seed=seed,
        epsilon=EpsilonScheduleConfig(
            kind=EpsilonDecayKind.EXPONENTIAL,
            start=0.4,
            minimum=0.05,
            decay=0.8,
        ),
        replay_capacity=64,
        batch_size=4,
        warmup_steps=4,
        target_update_interval=2,
        hidden_units=8,
    )


def _empty_fast_factory(max_steps: int = 12):
    motion = Room4MotionConfig(time_step_s=1.0, exit_radius_m=0.8)
    obstacle = Room5ObstacleConfig(min_obstacles=0, max_obstacles=0, observation_distance_m=2.5)
    reward = Room5RewardConfig(exit=50.0, timeout=-5.0, distance_progress_scale=5.0)
    return lambda: Room5Obstacles(
        motion_config=motion,
        obstacle_config=obstacle,
        reward_config=reward,
        max_steps=max_steps,
    )


class TestRoom5Environment:
    def test_observation_contract_and_fixed_width(self):
        env = Room5Obstacles(obstacle_config=Room5ObstacleConfig(fixed_layout=True))
        obs = env.reset(seed=1, layout_seed=2)

        assert len(obs) == 22
        assert env.obstacle_width_m == pytest.approx(0.5)
        assert all(o.width_m == pytest.approx(0.5) for o in env.obstacles)

    def test_obstacle_width_must_be_exactly_half_meter(self):
        with pytest.raises(ValueError, match="exactly 0.5"):
            Room5ObstacleConfig(obstacle_width_m=0.6)

    def test_seeded_layout_reproducibility_and_count_range(self):
        cfg = Room5ObstacleConfig(min_obstacles=2, max_obstacles=4, layout_seed=123)
        env_a = Room5Obstacles(obstacle_config=cfg)
        env_b = Room5Obstacles(obstacle_config=cfg)

        env_a.reset(seed=1, layout_seed=999)
        env_b.reset(seed=99, layout_seed=999)

        assert env_a.obstacles == env_b.obstacles
        assert env_a.layout_signature() == env_b.layout_signature()
        assert 2 <= len(env_a.obstacles) <= 4

    def test_generated_layout_avoids_start_exit_and_obstacle_overlap(self):
        env = Room5Obstacles(obstacle_config=Room5ObstacleConfig(min_obstacles=5, max_obstacles=5))
        env.reset(seed=1, layout_seed=2026)

        sx, sy = env.motion.start_position
        ex, ey = env.motion.exit_center
        half = env.obstacle_width_m / 2
        for obstacle in env.obstacles:
            assert not (abs(sx - obstacle.center_x) <= half and abs(sy - obstacle.center_y) <= half)
            assert not (abs(ex - obstacle.center_x) <= half and abs(ey - obstacle.center_y) <= half)
        for i, left in enumerate(env.obstacles):
            for right in env.obstacles[i + 1:]:
                assert abs(left.center_x - right.center_x) >= env.obstacle_width_m or (
                    abs(left.center_y - right.center_y) >= env.obstacle_width_m
                )

    def test_observation_distance_uses_center_to_center_measurement(self):
        cfg = Room5ObstacleConfig(fixed_layout=True, observation_distance_m=1.0)
        env = Room5Obstacles(obstacle_config=cfg)

        obs = env.reset(seed=1, layout_seed=1, start_state=(2.0, 4.2, 0, 0))
        assert obs[6] == pytest.approx(1.0)
        assert obs[7] == pytest.approx(1.0)
        assert obs[9] == pytest.approx(1.0)

        obs = env.reset(seed=1, layout_seed=1, start_state=(1.99, 4.2, 0, 0))
        assert obs[6] == pytest.approx(0.0)

    def test_obstacle_collision_terminates_as_failure(self):
        env = Room5Obstacles(obstacle_config=Room5ObstacleConfig(fixed_layout=True), max_steps=20)
        env.reset(seed=1, layout_seed=1, start_state=(2.70, 4.2, 0, 0))

        result = env.step(VelocityAction.EAST)

        assert result.terminated
        assert not result.truncated
        assert result.info["event"] == "obstacle_collision"
        assert result.info["collision"] == "obstacle"
        assert result.info["success"] is False

    def test_timeout_and_render_immutability(self):
        env = Room5Obstacles(
            obstacle_config=Room5ObstacleConfig(min_obstacles=0, max_obstacles=0),
            max_steps=1,
        )
        env.reset(seed=1, layout_seed=1)

        result = env.step(VelocityAction.STOP)
        render_state = env.render()

        assert result.truncated
        assert result.info["event"] == "timeout"
        with pytest.raises(FrozenInstanceError):
            render_state.x = 5.0


class TestDQNMechanics:
    def test_replay_buffer_samples_recent_transitions(self):
        buffer = ReplayBuffer(capacity=3, input_dim=2)
        for i in range(5):
            state = np.array([i, i + 1], dtype=float)
            buffer.add(
                SimpleNamespace(
                    state=state,
                    action=i % 2,
                    reward=float(i),
                    next_state=state + 1,
                    done=False,
                )
            )

        states, actions, rewards, next_states, dones = buffer.sample(3, np.random.default_rng(1))

        assert len(buffer) == 3
        assert states.shape == (3, 2)
        assert actions.shape == rewards.shape == dones.shape == (3,)
        assert next_states.shape == (3, 2)

    def test_td_update_moves_selected_action_toward_target(self):
        net = DQNNetwork(input_dim=3, hidden_units=4, action_count=2, rng=np.random.default_rng(1))
        states = np.ones((8, 3), dtype=float)
        actions = np.zeros(8, dtype=int)
        before = net.predict(states)[0, 0]
        targets = np.full(8, before + 2.0)

        loss, mean_td, max_td = net.train_batch(states, actions, targets, learning_rate=0.01)
        after = net.predict(states)[0, 0]

        assert loss > 0
        assert mean_td > 0
        assert max_td > 0
        assert after > before

    def test_training_is_reproducible_and_weights_are_finite(self):
        factory = _empty_fast_factory()
        cfg = _tiny_dqn_config(seed=123, episodes=5)

        first = DQNAgent(factory, cfg).train()
        second = DQNAgent(factory, cfg).train()

        assert first.metrics[0].epsilon == pytest.approx(cfg.epsilon.start)
        assert first.final_epsilon < cfg.epsilon.start
        for key in first.weights:
            assert np.all(np.isfinite(first.weights[key]))
            assert np.allclose(first.weights[key], second.weights[key])

    def test_save_load_roundtrip_validates_metadata(self, tmp_path):
        factory = _empty_fast_factory()
        result = DQNAgent(factory, _tiny_dqn_config(seed=5, episodes=3)).train()
        stem = tmp_path / "room5_dqn_test"

        save_dqn_model(result, str(stem), environment_factory=factory)
        network, metadata = load_dqn_model(str(stem))

        assert metadata["algorithm"] == "NumPy DQN"
        assert metadata["room"] == "Room5Obstacles"
        assert metadata["input_dim"] == result.input_dim
        assert np.allclose(network.predict(factory().reset(seed=1, layout_seed=1)),
                           DQNNetwork.from_weights(dict(result.weights)).predict(factory().reset(seed=1, layout_seed=1)))

    def test_trained_policy_beats_random_rollouts_on_fast_empty_room(self):
        factory = _empty_fast_factory(max_steps=14)
        cfg = DQNConfig(
            episodes=70,
            learning_rate=0.005,
            gamma=0.95,
            max_steps=14,
            seed=77,
            epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.05, decay=0.94),
            replay_capacity=2000,
            batch_size=16,
            warmup_steps=16,
            target_update_interval=10,
            hidden_units=32,
        )

        result = DQNAgent(factory, cfg).train()
        trained = evaluate_dqn_policy(factory, result, n_episodes=8, seeds=range(8), max_steps=14)
        random_net = DQNNetwork(result.input_dim, result.config.hidden_units, result.action_count,
                                rng=np.random.default_rng(99))
        random_rollouts = [
            rollout_dqn_policy(factory, random_net, seed=i, layout_seed=i, epsilon=1.0, max_steps=14)
            for i in range(8)
        ]

        assert trained.mean_return > np.mean([r.total_reward for r in random_rollouts])
