"""Train local Room 2-5 models for the Streamlit showcase.

This utility intentionally writes only to storage/models. It is for local demos;
final grading experiments still live under storage/experiments/final.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.approximate_sarsa import ApproximateSarsaAgent, save_approximate_model
from agents.dqn import DQNAgent, save_dqn_model
from agents.q_learning import QLearningAgent, save_q_model
from agents.sarsa import SarsaAgent, save_model
from core.types import (
    ApproximateSarsaConfig,
    ApproximateSarsaTrainingResult,
    DQNConfig,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    QLearningTrainingResult,
    QLearningConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
    SarsaConfig,
    SarsaTrainingResult,
    StartMode,
    TileCodingConfig,
)
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from environments.room4_continuous import ContinuousRewardConfig, Room4Continuous, Room4MotionConfig
from environments.room5_obstacles import Room5Obstacles


STAGE_TARGETS: tuple[tuple[str, float | None], ...] = (
    ("beginning", 0.0),
    ("25", 0.25),
    ("50", 0.50),
    ("75", 0.75),
    ("final", None),
)


def _epsilon(decay: float, minimum: float = 0.05) -> EpsilonScheduleConfig:
    # All generated showcase models use exponential exploration decay.
    return EpsilonScheduleConfig(
        kind=EpsilonDecayKind.EXPONENTIAL,
        start=1.0,
        minimum=minimum,
        decay=decay,
    )


def _stem(output_root: Path, subdir: str, showcase_name: str, local_prefix: str, showcase: bool) -> Path:
    # Showcase mode uses stable filenames checked by the Streamlit app; normal
    # local mode writes timestamped files so older models are not overwritten.
    filename = showcase_name if showcase else f"{local_prefix}_local_{_timestamp()}"
    return output_root / subdir / filename


def _target_episode(total: int, fraction: float | None) -> int:
    if fraction is None:
        return total
    if fraction <= 0:
        return 1
    return max(1, min(total, round(total * fraction)))


def _nearest_snapshot_episode(result, target: int) -> int:
    available = sorted(result.snapshots.keys())
    if not available:
        raise ValueError("training result has no snapshots to save")
    if target in result.snapshots:
        return target
    before = [episode for episode in available if episode <= target]
    return before[-1] if before else available[0]


def _save_room2_stages(result: SarsaTrainingResult, stage_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for slug, fraction in STAGE_TARGETS:
        if fraction is None:
            stage_result = SarsaTrainingResult(
                config=result.config,
                q_values=result.q_values,
                metrics=result.metrics,
                snapshots={},
                final_epsilon=result.final_epsilon,
                training_seed=result.training_seed,
            )
        else:
            episode = _nearest_snapshot_episode(result, _target_episode(result.config.episodes, fraction))
            snap = result.snapshots[episode]
            stage_result = SarsaTrainingResult(
                config=replace(result.config, episodes=episode),
                q_values=snap.q_values,
                metrics=result.metrics[:episode],
                snapshots={},
                final_epsilon=snap.epsilon,
                training_seed=result.training_seed,
            )
        stem = stage_dir / slug
        paths.append(Path(save_model(stage_result, str(stem), slip_config=Room2SARSA().slip_config, map_grid=ROOM2_GRID)))
    return paths


def _save_room3_stages(result: QLearningTrainingResult, stage_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for slug, fraction in STAGE_TARGETS:
        if fraction is None:
            stage_result = QLearningTrainingResult(
                config=result.config,
                q_values=result.q_values,
                metrics=result.metrics,
                snapshots={},
                final_epsilon=result.final_epsilon,
                training_seed=result.training_seed,
            )
        else:
            episode = _nearest_snapshot_episode(result, _target_episode(result.config.episodes, fraction))
            snap = result.snapshots[episode]
            stage_result = QLearningTrainingResult(
                config=replace(result.config, episodes=episode),
                q_values=snap.q_values,
                metrics=result.metrics[:episode],
                snapshots={},
                final_epsilon=snap.epsilon,
                training_seed=result.training_seed,
            )
        stem = stage_dir / slug
        paths.append(Path(save_q_model(stage_result, str(stem), slip_config=Room3QLearning().slip_config, map_grid=ROOM3_GRID)))
    return paths


def _save_room4_stages(
    result: ApproximateSarsaTrainingResult,
    stage_dir: Path,
    *,
    tile_config: TileCodingConfig,
    motion_config: Room4MotionConfig,
    reward_config: ContinuousRewardConfig,
) -> list[Path]:
    paths: list[Path] = []
    for slug, fraction in STAGE_TARGETS:
        if fraction is None:
            stage_result = ApproximateSarsaTrainingResult(
                config=result.config,
                weights=result.weights,
                metrics=result.metrics,
                snapshots={},
                final_epsilon=result.final_epsilon,
                training_seed=result.training_seed,
            )
        else:
            episode = _nearest_snapshot_episode(result, _target_episode(result.config.episodes, fraction))
            snap = result.snapshots[episode]
            stage_result = ApproximateSarsaTrainingResult(
                config=replace(result.config, episodes=episode),
                weights=snap.weights,
                metrics=result.metrics[:episode],
                snapshots={},
                final_epsilon=snap.epsilon,
                training_seed=result.training_seed,
            )
        stem = stage_dir / slug
        paths.append(
            Path(
                save_approximate_model(
                    stage_result,
                    str(stem),
                    tile_coding_config=tile_config,
                    motion_config=motion_config,
                    reward_config=reward_config,
                )
            )
        )
    return paths


def train_room2(output_root: Path, episodes: int, seed: int, *, showcase: bool = False) -> Path:
    # Train the SARSA model that the Room 2 showcase can load quickly.
    env_factory = lambda: Room2SARSA(max_steps=200)
    config = SarsaConfig(
        episodes=episodes,
        alpha=0.05,
        gamma=0.95,
        max_steps=200,
        seed=seed,
        epsilon=_epsilon(0.99),
    )
    result = SarsaAgent(env_factory, config).train()
    stem = _stem(output_root, "room2_sarsa", "showcase_sarsa", "sarsa", showcase)
    save_model(result, str(stem), slip_config=Room2SARSA().slip_config, map_grid=ROOM2_GRID)
    if showcase:
        _save_room2_stages(result, output_root / "room2_sarsa" / "showcase_stages")
    return stem.with_suffix(".json")


def train_room3(output_root: Path, episodes: int, seed: int, *, showcase: bool = False) -> Path:
    # Train the Q-Learning model that includes the key-collection state flag.
    env_factory = lambda: Room3QLearning(max_steps=300)
    config = QLearningConfig(
        episodes=episodes,
        alpha=0.50,
        gamma=0.99,
        max_steps=300,
        seed=seed,
        epsilon=_epsilon(0.999),
    )
    result = QLearningAgent(env_factory, config).train()
    stem = _stem(output_root, "room3_q_learning", "showcase_ql", "ql", showcase)
    save_q_model(result, str(stem), slip_config=Room3QLearning().slip_config, map_grid=ROOM3_GRID)
    if showcase:
        _save_room3_stages(result, output_root / "room3_q_learning" / "showcase_stages")
    return stem.with_suffix(".json")


def train_room4(output_root: Path, episodes: int, seed: int, *, showcase: bool = False) -> Path:
    # Train the tile-coded linear model used by the continuous Room 4 showcase.
    motion = Room4MotionConfig()
    reward = ContinuousRewardConfig(distance_progress_scale=1.0)
    tile_config = TileCodingConfig(num_tilings=16, tiles_x=16, tiles_y=16)
    env_factory = lambda: Room4Continuous(
        motion_config=motion,
        reward_config=reward,
        max_steps=750,
        start_mode=StartMode.RANDOM_LOWER_LEFT,
    )
    config = ApproximateSarsaConfig(
        episodes=episodes,
        alpha=0.05,
        gamma=0.99,
        max_steps=750,
        seed=seed,
        epsilon=_epsilon(0.995, minimum=0.02),
        tile_coding=tile_config,
        start_mode=StartMode.RANDOM_LOWER_LEFT,
    )
    result = ApproximateSarsaAgent(env_factory, config).train()
    stem = _stem(output_root, "room4_approximate_sarsa", "showcase_approx", "approx", showcase)
    save_approximate_model(
        result,
        str(stem),
        tile_coding_config=tile_config,
        motion_config=motion,
        reward_config=reward,
    )
    if showcase:
        _save_room4_stages(
            result,
            output_root / "room4_approximate_sarsa" / "showcase_stages",
            tile_config=tile_config,
            motion_config=motion,
            reward_config=reward,
        )
    return stem.with_suffix(".json")


def train_room5(output_root: Path, episodes: int, seed: int, *, showcase: bool = False) -> Path:
    # Train the optional DQN model for the obstacle room.
    motion = Room4MotionConfig(time_step_s=0.05)
    obstacle = Room5ObstacleConfig(min_obstacles=3, max_obstacles=5, observation_distance_m=2.5, layout_seed=seed)
    reward = Room5RewardConfig(distance_progress_scale=2.0)
    env_factory = lambda: Room5Obstacles(
        motion_config=motion,
        obstacle_config=obstacle,
        reward_config=reward,
        max_steps=260,
    )
    config = DQNConfig(
        episodes=episodes,
        learning_rate=0.001,
        gamma=0.99,
        max_steps=260,
        seed=seed,
        epsilon=_epsilon(0.995, minimum=0.05),
        replay_capacity=20_000,
        batch_size=64,
        warmup_steps=128,
        target_update_interval=100,
        hidden_units=64,
    )
    result = DQNAgent(env_factory, config).train()
    stem = _stem(output_root, "room5_dqn", "showcase_dqn", "dqn", showcase)
    save_dqn_model(result, str(stem), environment_factory=env_factory)
    return stem.with_suffix(".json")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    # Command-line entry point.  --smoke is useful when checking that saving and
    # loading work without waiting for full training.
    parser = argparse.ArgumentParser(description="Train local models for Streamlit showcase rooms.")
    parser.add_argument("--output-root", default="storage/models", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--room2-episodes", default=500, type=int)
    parser.add_argument("--room3-episodes", default=500, type=int)
    parser.add_argument("--room4-episodes", default=500, type=int)
    parser.add_argument("--room5-episodes", default=300, type=int)
    parser.add_argument("--showcase", action="store_true", help="Write deterministic showcase_* artifact names.")
    parser.add_argument(
        "--rooms",
        nargs="+",
        choices=["room2", "room3", "room4", "room5"],
        default=["room2", "room3", "room4", "room5"],
        help="Subset of rooms to generate.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use tiny episode counts for wiring checks.")
    args = parser.parse_args()

    if args.smoke:
        args.room2_episodes = 2
        args.room3_episodes = 2
        args.room4_episodes = 1
        args.room5_episodes = 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    if "room2" in args.rooms:
        paths.append(train_room2(args.output_root, args.room2_episodes, args.seed, showcase=args.showcase))
    if "room3" in args.rooms:
        paths.append(train_room3(args.output_root, args.room3_episodes, args.seed, showcase=args.showcase))
    if "room4" in args.rooms:
        paths.append(train_room4(args.output_root, args.room4_episodes, args.seed, showcase=args.showcase))
    if "room5" in args.rooms:
        paths.append(train_room5(args.output_root, args.room5_episodes, args.seed, showcase=args.showcase))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
