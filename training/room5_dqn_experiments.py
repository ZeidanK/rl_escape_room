"""Final optional Room 5 DQN experiments.

The counts are intentionally modest so the pipeline remains runnable on a
student laptop, but every number is produced by real training/evaluation.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn import DQNAgent, DQNNetwork, evaluate_dqn_policy, rollout_dqn_policy, save_dqn_model
from core.types import (
    DQNConfig,
    DQNEvaluationSummary,
    DQNTrainingResult,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    Room4MotionConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
)
from environments.room5_obstacles import Room5Obstacles


FINAL_ROOM5_PATH = Path("storage/experiments/final/room5_dqn_confirmation.json")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _candidate_configs() -> list[dict[str, Any]]:
    # Small candidate set keeps optional DQN experiments practical on a student
    # laptop while still comparing meaningful design choices.
    return [
        {
            "name": "baseline_progress",
            "learning_rate": 0.003,
            "epsilon_decay": 0.955,
            "hidden_units": 32,
            "progress_scale": 6.0,
            "observation_distance_m": 3.5,
        },
        {
            "name": "slower_decay",
            "learning_rate": 0.003,
            "epsilon_decay": 0.970,
            "hidden_units": 32,
            "progress_scale": 6.0,
            "observation_distance_m": 3.5,
        },
        {
            "name": "larger_network",
            "learning_rate": 0.003,
            "epsilon_decay": 0.960,
            "hidden_units": 48,
            "progress_scale": 6.0,
            "observation_distance_m": 3.5,
        },
    ]


def _make_components(params: dict[str, Any], *, episodes: int, seed: int):
    # Build all config objects from one parameter dictionary so screening and
    # confirmation use exactly the same environment/model recipe.
    motion = Room4MotionConfig(time_step_s=0.1, exit_radius_m=0.6)
    obstacle = Room5ObstacleConfig(
        min_obstacles=1,
        max_obstacles=3,
        observation_distance_m=float(params["observation_distance_m"]),
        layout_seed=42,
    )
    reward = Room5RewardConfig(
        exit=120.0,
        obstacle_collision=-60.0,
        timeout=-15.0,
        distance_progress_scale=float(params["progress_scale"]),
    )

    def env_factory() -> Room5Obstacles:
        return Room5Obstacles(
            motion_config=motion,
            obstacle_config=obstacle,
            reward_config=reward,
            max_steps=160,
        )

    config = DQNConfig(
        episodes=episodes,
        learning_rate=float(params["learning_rate"]),
        gamma=0.98,
        max_steps=160,
        seed=seed,
        epsilon=EpsilonScheduleConfig(
            kind=EpsilonDecayKind.EXPONENTIAL,
            start=1.0,
            minimum=0.03,
            decay=float(params["epsilon_decay"]),
        ),
        replay_capacity=10_000,
        batch_size=32,
        warmup_steps=32,
        target_update_interval=40,
        hidden_units=int(params["hidden_units"]),
    )
    return motion, obstacle, reward, env_factory, config


def _fixed_factory(params: dict[str, Any]) -> Room5Obstacles:
    motion = Room4MotionConfig(time_step_s=0.1, exit_radius_m=0.6)
    reward = Room5RewardConfig(
        exit=120.0,
        obstacle_collision=-60.0,
        timeout=-15.0,
        distance_progress_scale=float(params["progress_scale"]),
    )
    obstacle = Room5ObstacleConfig(
        min_obstacles=1,
        max_obstacles=3,
        observation_distance_m=float(params["observation_distance_m"]),
        fixed_layout=True,
    )
    return Room5Obstacles(motion_config=motion, obstacle_config=obstacle, reward_config=reward, max_steps=160)


def _training_summary(result: DQNTrainingResult, runtime_s: float) -> dict[str, Any]:
    window = min(30, len(result.metrics))
    recent = result.metrics[-window:] if window else ()
    return {
        "episodes": result.config.episodes,
        "seed": result.training_seed,
        "runtime_seconds": round(float(runtime_s), 6),
        "final_epsilon": float(result.final_epsilon),
        "recent_window": window,
        "recent_success_rate": float(sum(m.success for m in recent) / window) if window else 0.0,
        "recent_mean_return": float(np.mean([m.total_reward for m in recent])) if recent else 0.0,
        "recent_obstacle_collision_rate": float(sum(m.obstacle_collisions for m in recent) / window) if window else 0.0,
        "final_episode_return": float(result.metrics[-1].total_reward) if result.metrics else 0.0,
        "finite_weights": all(bool(np.all(np.isfinite(w))) for w in result.weights.values()),
    }


def _summary_dict(summary: DQNEvaluationSummary) -> dict[str, Any]:
    return {
        "category": summary.category,
        "episodes": summary.n_episodes,
        "successes": summary.successes,
        "success_rate": float(summary.success_rate),
        "mean_return": float(summary.mean_return),
        "std_return": float(summary.std_return),
        "mean_steps": float(summary.mean_steps),
        "mean_successful_steps": None if summary.mean_successful_steps is None else float(summary.mean_successful_steps),
        "truncated_count": summary.truncated_count,
        "obstacle_collision_count": summary.obstacle_collision_count,
        "boundary_collision_count": summary.boundary_collision_count,
        "rollouts": [
            {
                "seed": r.seed,
                "layout_seed": r.layout_seed,
                "success": r.success,
                "steps": r.steps,
                "return": float(r.total_reward),
                "obstacle_collisions": r.obstacle_collisions,
                "boundary_collisions": r.boundary_collisions,
                "visible_obstacle_steps": r.visible_obstacle_steps,
            }
            for r in summary.rollouts
        ],
    }


def _rollout_dict(rollout) -> dict[str, Any]:
    return {
        "seed": rollout.seed,
        "layout_seed": rollout.layout_seed,
        "start_state": [float(v) for v in rollout.start_state],
        "final_state": [float(v) for v in rollout.final_state],
        "success": rollout.success,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "steps": rollout.steps,
        "simulated_time_s": float(rollout.simulated_time_s),
        "total_reward": float(rollout.total_reward),
        "obstacle_collisions": rollout.obstacle_collisions,
        "boundary_collisions": rollout.boundary_collisions,
        "trajectory": [
            {
                "index": step.index,
                "raw_state": [float(v) for v in step.raw_state],
                "action": step.requested_action.name,
                "reward": float(step.reward),
                "next_raw_state": [float(v) for v in step.next_raw_state],
                "collision": step.collision,
                "event": step.event,
                "cumulative_reward": float(step.cumulative_reward),
                "visible_obstacle_count": step.visible_obstacle_count,
                "distance_to_exit_m": float(step.distance_to_exit_m),
            }
            for step in rollout.trajectory
        ],
    }


def _rank_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    # Prefer policies that succeed, then return more reward, hit fewer
    # obstacles, and finish in fewer steps.
    ev = entry["random_layout_evaluation"]
    return (
        float(ev["success_rate"]),
        float(ev["mean_return"]),
        -float(ev["obstacle_collision_count"]),
        -float(ev["mean_steps"]),
    )


def run_room5_experiments(
    *,
    output_path: Path = FINAL_ROOM5_PATH,
    confirmation_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    save_showcase_stem: Path | None = None,
) -> dict[str, Any]:
    # Two-stage optional pipeline: screen a few DQN settings, then confirm the
    # best one across multiple seeds and save a showcase model if requested.
    started = time.perf_counter()
    screening_entries: list[dict[str, Any]] = []

    for idx, params in enumerate(_candidate_configs()):
        _, _, _, env_factory, config = _make_components(params, episodes=60, seed=100 + idx)
        train_start = time.perf_counter()
        result = DQNAgent(env_factory, config).train()
        runtime_s = time.perf_counter() - train_start
        random_eval = evaluate_dqn_policy(
            env_factory,
            result,
            n_episodes=8,
            seeds=range(8),
            layout_seeds=range(20_000, 20_008),
            max_steps=config.max_steps,
            category="screening_random_layouts",
        )
        screening_entries.append(
            {
                "config": params,
                "training": _training_summary(result, runtime_s),
                "random_layout_evaluation": _summary_dict(random_eval),
            }
        )

    best_screen = sorted(screening_entries, key=_rank_key, reverse=True)[0]
    best_params = dict(best_screen["config"])
    confirmation: list[dict[str, Any]] = []
    best_result: DQNTrainingResult | None = None
    best_result_score: tuple[float, float, float, float] | None = None
    best_env_factory = None

    for seed in confirmation_seeds:
        _, _, _, env_factory, config = _make_components(best_params, episodes=180, seed=seed)
        train_start = time.perf_counter()
        result = DQNAgent(env_factory, config).train()
        runtime_s = time.perf_counter() - train_start
        fixed_eval = evaluate_dqn_policy(
            lambda: _fixed_factory(best_params),
            result,
            n_episodes=12,
            seeds=range(12),
            layout_seeds=[42],
            max_steps=config.max_steps,
            category="fixed_validation_layout",
        )
        random_eval = evaluate_dqn_policy(
            env_factory,
            result,
            n_episodes=12,
            seeds=range(12),
            layout_seeds=range(30_000, 30_012),
            max_steps=config.max_steps,
            category="seeded_random_layouts",
        )
        unseen_eval = evaluate_dqn_policy(
            env_factory,
            result,
            n_episodes=12,
            seeds=range(10_000, 10_012),
            layout_seeds=range(50_000, 50_012),
            max_steps=config.max_steps,
            category="unseen_random_layouts",
        )
        entry = {
            "seed": seed,
            "training": _training_summary(result, runtime_s),
            "fixed_layout_evaluation": _summary_dict(fixed_eval),
            "random_layout_evaluation": _summary_dict(random_eval),
            "unseen_layout_evaluation": _summary_dict(unseen_eval),
        }
        confirmation.append(entry)
        score = (
            entry["unseen_layout_evaluation"]["success_rate"],
            entry["random_layout_evaluation"]["success_rate"],
            entry["unseen_layout_evaluation"]["mean_return"],
            entry["random_layout_evaluation"]["mean_return"],
        )
        if best_result_score is None or score > best_result_score:
            best_result = result
            best_result_score = score
            best_env_factory = env_factory

    assert best_result is not None and best_env_factory is not None
    best_network = DQNNetwork.from_weights(dict(best_result.weights))
    replay_rollouts = [
        rollout_dqn_policy(best_env_factory, best_network, seed=900 + i, layout_seed=60_000 + i, max_steps=160)
        for i in range(3)
    ]

    if save_showcase_stem is not None:
        save_dqn_model(best_result, str(save_showcase_stem), environment_factory=best_env_factory)

    def _mean(path: str) -> float:
        return float(np.mean([entry[path]["success_rate"] for entry in confirmation]))

    artifact = {
        "schema_version": 1,
        "room": "Room 5 - Dynamic Obstacles",
        "algorithm": "NumPy DQN",
        "deployment_considered": False,
        "git_commit": _git_commit(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(float(time.perf_counter() - started), 6),
        "ranking_rule": "success_rate, then mean_return, then fewer obstacle collisions, then fewer steps",
        "screening_episodes": 60,
        "confirmation_episodes": 180,
        "confirmation_seeds": list(confirmation_seeds),
        "screening_configs": screening_entries,
        "best_config": best_params,
        "confirmation": {
            "seed_results": confirmation,
            "aggregate": {
                "fixed_success_rate_mean": _mean("fixed_layout_evaluation"),
                "random_success_rate_mean": _mean("random_layout_evaluation"),
                "unseen_success_rate_mean": _mean("unseen_layout_evaluation"),
                "training_runtime_seconds_sum": float(sum(e["training"]["runtime_seconds"] for e in confirmation)),
            },
        },
        "replay_trajectories": [_rollout_dict(r) for r in replay_rollouts],
    }
    _assert_json_finite(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _assert_json_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_finite(item)
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite float in Room 5 artifact")


if __name__ == "__main__":
    artifact = run_room5_experiments()
    print(json.dumps({
        "output": str(FINAL_ROOM5_PATH),
        "runtime_seconds": artifact["runtime_seconds"],
        "best_config": artifact["best_config"],
        "aggregate": artifact["confirmation"]["aggregate"],
    }, indent=2))
