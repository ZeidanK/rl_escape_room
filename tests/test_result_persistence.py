"""Saved training-history persistence regressions."""

import os

import numpy as np

from agents.approximate_sarsa import ApproximateSarsaAgent, save_approximate_model
from agents.dqn import DQNAgent, save_dqn_model
from agents.q_learning import QLearningAgent, save_q_model
from agents.sarsa import SarsaAgent, save_model
from agents.dynamic_programming import ValueIterationAgent, evaluate_policy, rollout_policy
from core.types import (
    ApproximateSarsaConfig,
    DQNConfig,
    EpsilonScheduleConfig,
    QLearningConfig,
    Room4MotionConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
    SarsaConfig,
    SlipConfig,
    StartMode,
    ValueIterationConfig,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from environments.room4_continuous import Room4Continuous
from environments.room5_obstacles import Room5Obstacles
from features.tile_coding import TileCodingConfig
from training.result_persistence import (
    SAVED_OUTPUTS_KEY,
    deserialize_approximate_metrics,
    deserialize_dqn_metrics,
    deserialize_policy_evaluation,
    deserialize_sarsa_metrics,
    deserialize_q_learning_metrics,
    deserialize_grid_rollout,
    load_room1_run,
    read_json,
    save_room1_run,
)


def _room5_fast_factory(max_steps: int = 8):
    motion = Room4MotionConfig(time_step_s=1.0, exit_radius_m=0.8)
    obstacle = Room5ObstacleConfig(min_obstacles=0, max_obstacles=0)
    reward = Room5RewardConfig(exit=50.0, timeout=-5.0, distance_progress_scale=5.0)
    return lambda: Room5Obstacles(
        motion_config=motion,
        obstacle_config=obstacle,
        reward_config=reward,
        max_steps=max_steps,
    )


def test_room2_sarsa_save_stores_training_metrics(tmp_path):
    result = SarsaAgent(
        lambda: Room2SARSA(max_steps=20),
        SarsaConfig(episodes=3, max_steps=20, seed=1),
    ).train()
    stem = str(tmp_path / "room2_run")

    save_model(result, stem, slip_config=Room2SARSA().slip_config, map_grid=ROOM2_GRID)
    metadata = read_json(stem + ".json")
    restored = deserialize_sarsa_metrics(metadata["training_metrics"])

    assert len(restored) == len(result.metrics)
    assert restored[-1].total_reward == result.metrics[-1].total_reward


def test_room3_q_learning_save_stores_training_metrics(tmp_path):
    result = QLearningAgent(
        lambda: Room3QLearning(max_steps=20),
        QLearningConfig(episodes=3, max_steps=20, seed=2),
    ).train()
    stem = str(tmp_path / "room3_run")

    save_q_model(result, stem, slip_config=Room3QLearning().slip_config, map_grid=ROOM3_GRID)
    metadata = read_json(stem + ".json")
    restored = deserialize_q_learning_metrics(metadata["training_metrics"])

    assert len(restored) == len(result.metrics)
    assert restored[-1].locked_exit_attempts == result.metrics[-1].locked_exit_attempts


def test_room4_approximate_save_stores_training_metrics(tmp_path):
    tile_config = TileCodingConfig(num_tilings=2, tiles_x=4, tiles_y=4)
    motion = Room4MotionConfig()
    result = ApproximateSarsaAgent(
        lambda: Room4Continuous(motion_config=motion, max_steps=4, start_mode=StartMode.FIXED),
        ApproximateSarsaConfig(
            episodes=2,
            max_steps=4,
            seed=3,
            tile_coding=tile_config,
            start_mode=StartMode.FIXED,
        ),
    ).train()
    stem = str(tmp_path / "room4_run")

    save_approximate_model(result, stem, tile_coding_config=tile_config, motion_config=motion)
    metadata = read_json(stem + ".json")
    restored = deserialize_approximate_metrics(metadata["training_metrics"])

    assert len(restored) == len(result.metrics)
    assert restored[-1].final_distance_to_exit_m == result.metrics[-1].final_distance_to_exit_m


def test_room5_dqn_save_stores_training_metrics(tmp_path):
    factory = _room5_fast_factory()
    result = DQNAgent(
        factory,
        DQNConfig(
            episodes=2,
            max_steps=8,
            seed=4,
            epsilon=EpsilonScheduleConfig(start=0.4, minimum=0.05, decay=0.8),
            replay_capacity=32,
            batch_size=4,
            warmup_steps=4,
            target_update_interval=2,
            hidden_units=8,
        ),
    ).train()
    stem = str(tmp_path / "room5_run")

    save_dqn_model(result, stem, environment_factory=factory)
    metadata = read_json(stem + ".json")
    restored = deserialize_dqn_metrics(metadata["training_metrics"])

    assert len(restored) == len(result.metrics)
    assert restored[-1].mean_loss == result.metrics[-1].mean_loss


def test_room1_saved_run_roundtrip_includes_graph_and_outputs(tmp_path):
    env = Room1DP(slip_config=SlipConfig(1.0, 0.0, 0.0), max_steps=30)
    config = ValueIterationConfig(gamma=0.9, tolerance=1e-2, max_iterations=1000)
    result = ValueIterationAgent(env, config).solve()
    rollout = rollout_policy(env, result.policy, seed=5)
    summary = evaluate_policy(env, result.policy, n_episodes=2, seeds=range(2))
    stem = str(tmp_path / "room1_run")

    save_room1_run(
        result,
        stem,
        config=config,
        slip_config=env.slip_config,
        map_grid=env.grid,
        rollout=rollout,
        evaluation=summary,
    )
    loaded, metadata = load_room1_run(stem, map_grid=env.grid)
    saved_outputs = metadata[SAVED_OUTPUTS_KEY]

    assert loaded.delta_history == result.delta_history
    assert np.isclose(loaded.start_state_value, result.start_state_value)
    assert deserialize_grid_rollout(saved_outputs["rollout"]).total_steps == rollout.total_steps
    assert deserialize_policy_evaluation(saved_outputs["evaluation_summary"]).episodes == 2
    assert os.path.exists(stem + ".json")
