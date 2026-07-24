"""Train local Room 2-4 models for the Streamlit showcase.

This utility intentionally writes only to storage/models. It is for local demos;
final grading experiments still live under storage/experiments/final.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.approximate_sarsa import ApproximateSarsaAgent, save_approximate_model
from agents.q_learning import QLearningAgent, save_q_model
from agents.sarsa import SarsaAgent, save_model
from core.types import (
    ApproximateSarsaConfig,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    QLearningConfig,
    SarsaConfig,
    StartMode,
    TileCodingConfig,
)
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from environments.room4_continuous import ContinuousRewardConfig, Room4Continuous, Room4MotionConfig


def _epsilon(decay: float, minimum: float = 0.05) -> EpsilonScheduleConfig:
    return EpsilonScheduleConfig(
        kind=EpsilonDecayKind.EXPONENTIAL,
        start=1.0,
        minimum=minimum,
        decay=decay,
    )


def train_room2(output_root: Path, episodes: int, seed: int) -> Path:
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
    stem = output_root / "room2_sarsa" / f"sarsa_local_{_timestamp()}"
    save_model(result, str(stem), slip_config=Room2SARSA().slip_config, map_grid=ROOM2_GRID)
    return stem.with_suffix(".json")


def train_room3(output_root: Path, episodes: int, seed: int) -> Path:
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
    stem = output_root / "room3_q_learning" / f"ql_local_{_timestamp()}"
    save_q_model(result, str(stem), slip_config=Room3QLearning().slip_config, map_grid=ROOM3_GRID)
    return stem.with_suffix(".json")


def train_room4(output_root: Path, episodes: int, seed: int) -> Path:
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
    stem = output_root / "room4_approximate_sarsa" / f"approx_local_{_timestamp()}"
    save_approximate_model(
        result,
        str(stem),
        tile_coding_config=tile_config,
        motion_config=motion,
        reward_config=reward,
    )
    return stem.with_suffix(".json")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local models for Streamlit showcase rooms.")
    parser.add_argument("--output-root", default="storage/models", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--room2-episodes", default=500, type=int)
    parser.add_argument("--room3-episodes", default=500, type=int)
    parser.add_argument("--room4-episodes", default=500, type=int)
    parser.add_argument("--smoke", action="store_true", help="Use tiny episode counts for wiring checks.")
    args = parser.parse_args()

    if args.smoke:
        args.room2_episodes = 2
        args.room3_episodes = 2
        args.room4_episodes = 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = [
        train_room2(args.output_root, args.room2_episodes, args.seed),
        train_room3(args.output_root, args.room3_episodes, args.seed),
        train_room4(args.output_root, args.room4_episodes, args.seed),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
