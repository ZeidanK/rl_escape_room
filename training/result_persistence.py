"""Persistence helpers for Streamlit training-result history.

The model weights already live in JSON + NPZ artifacts.  This module stores the
UI-facing history that is needed to redraw graphs after Streamlit session state
is lost.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agents.tabular_utils import map_signature
from core.types import (
    Action,
    ApproximateEpisodeMetrics,
    ApproximateEvaluationSummary,
    ContinuousRolloutResult,
    ContinuousTrajectoryStep,
    DQNEpisodeMetrics,
    DQNEvaluationSummary,
    PolicyEvaluationSummary,
    Position,
    QLearningEpisodeMetrics,
    QLearningEvaluationSummary,
    RolloutResult,
    Room5RolloutResult,
    Room5TrajectoryStep,
    SarsaEvaluationSummary,
    TrajectoryStep,
    TrainingEpisodeMetrics,
    ValueIterationConfig,
    ValueIterationResult,
    VelocityAction,
)

RUN_HISTORY_SCHEMA_VERSION = 1
TRAINING_METRICS_KEY = "training_metrics"
SAVED_OUTPUTS_KEY = "saved_outputs"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metadata_path(filepath_stem: str | Path) -> Path:
    return Path(str(filepath_stem) + ".json")


def serialize_sarsa_metrics(metrics: tuple[TrainingEpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]


def deserialize_sarsa_metrics(rows: Any) -> tuple[TrainingEpisodeMetrics, ...]:
    if not isinstance(rows, list):
        return ()
    return tuple(
        TrainingEpisodeMetrics(
            episode=int(row.get("episode", 0)),
            total_reward=float(row.get("total_reward", 0.0)),
            steps=int(row.get("steps", 0)),
            success=bool(row.get("success", False)),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            epsilon=float(row.get("epsilon", 0.0)),
            collision_count=int(row.get("collision_count", 0)),
            slipped_action_count=int(row.get("slipped_action_count", 0)),
            trap_count=int(row.get("trap_count", 0)),
            mean_abs_td_error=float(row.get("mean_abs_td_error", 0.0)),
            max_abs_td_error=float(row.get("max_abs_td_error", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )


def serialize_q_learning_metrics(metrics: tuple[QLearningEpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]


def deserialize_q_learning_metrics(rows: Any) -> tuple[QLearningEpisodeMetrics, ...]:
    if not isinstance(rows, list):
        return ()
    return tuple(
        QLearningEpisodeMetrics(
            episode=int(row.get("episode", 0)),
            total_reward=float(row.get("total_reward", 0.0)),
            steps=int(row.get("steps", 0)),
            success=bool(row.get("success", False)),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            epsilon=float(row.get("epsilon", 0.0)),
            key_collected=bool(row.get("key_collected", False)),
            key_collection_step=(
                int(row["key_collection_step"])
                if row.get("key_collection_step") is not None
                else None
            ),
            locked_exit_attempts=int(row.get("locked_exit_attempts", 0)),
            collision_count=int(row.get("collision_count", 0)),
            slipped_action_count=int(row.get("slipped_action_count", 0)),
            trap_count=int(row.get("trap_count", 0)),
            mean_abs_td_error=float(row.get("mean_abs_td_error", 0.0)),
            max_abs_td_error=float(row.get("max_abs_td_error", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )


def serialize_approximate_metrics(metrics: tuple[ApproximateEpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]


def deserialize_approximate_metrics(rows: Any) -> tuple[ApproximateEpisodeMetrics, ...]:
    if not isinstance(rows, list):
        return ()
    return tuple(
        ApproximateEpisodeMetrics(
            episode=int(row.get("episode", 0)),
            total_reward=float(row.get("total_reward", 0.0)),
            steps=int(row.get("steps", 0)),
            simulated_time_s=float(row.get("simulated_time_s", 0.0)),
            success=bool(row.get("success", False)),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            epsilon=float(row.get("epsilon", 0.0)),
            collision_count=int(row.get("collision_count", 0)),
            distance_travelled_m=float(row.get("distance_travelled_m", 0.0)),
            final_distance_to_exit_m=float(row.get("final_distance_to_exit_m", 0.0)),
            mean_abs_td_error=float(row.get("mean_abs_td_error", 0.0)),
            max_abs_td_error=float(row.get("max_abs_td_error", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )


def serialize_dqn_metrics(metrics: tuple[DQNEpisodeMetrics, ...]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]


def deserialize_dqn_metrics(rows: Any) -> tuple[DQNEpisodeMetrics, ...]:
    if not isinstance(rows, list):
        return ()
    return tuple(
        DQNEpisodeMetrics(
            episode=int(row.get("episode", 0)),
            total_reward=float(row.get("total_reward", 0.0)),
            steps=int(row.get("steps", 0)),
            success=bool(row.get("success", False)),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            epsilon=float(row.get("epsilon", 0.0)),
            obstacle_collisions=int(row.get("obstacle_collisions", 0)),
            boundary_collisions=int(row.get("boundary_collisions", 0)),
            visible_obstacle_steps=int(row.get("visible_obstacle_steps", 0)),
            mean_loss=float(row.get("mean_loss", 0.0)),
            mean_abs_td_error=float(row.get("mean_abs_td_error", 0.0)),
            max_abs_td_error=float(row.get("max_abs_td_error", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )


def _position(raw: Any) -> Position:
    return int(raw[0]), int(raw[1])


def _continuous_state(raw: Any) -> tuple[float, float, int, int]:
    return float(raw[0]), float(raw[1]), int(raw[2]), int(raw[3])


def _observation(raw: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in raw)


def serialize_grid_step(step: TrajectoryStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "state": list(step.state),
        "requested_action": int(step.requested_action),
        "effective_action": int(step.effective_action),
        "reward": step.reward,
        "next_state": list(step.next_state),
        "slipped": step.slipped,
        "collision": step.collision,
        "event": step.event,
        "terminated": step.terminated,
        "truncated": step.truncated,
    }


def deserialize_grid_step(raw: dict[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        index=int(raw.get("index", 0)),
        state=_position(raw.get("state", (0, 0))),
        requested_action=Action(int(raw.get("requested_action", 0))),
        effective_action=Action(int(raw.get("effective_action", raw.get("requested_action", 0)))),
        reward=float(raw.get("reward", 0.0)),
        next_state=_position(raw.get("next_state", raw.get("state", (0, 0)))),
        slipped=bool(raw.get("slipped", False)),
        collision=raw.get("collision"),
        event=raw.get("event"),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
    )


def serialize_grid_rollout(rollout: RolloutResult | None) -> dict[str, Any] | None:
    if rollout is None:
        return None
    return {
        "steps": [serialize_grid_step(step) for step in rollout.steps],
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "success": rollout.success,
        "total_steps": rollout.total_steps,
        "total_reward": rollout.total_reward,
        "collisions": rollout.collisions,
        "slipped_actions": rollout.slipped_actions,
        "trap_count": rollout.trap_count,
    }


def deserialize_grid_rollout(raw: Any) -> RolloutResult | None:
    if not isinstance(raw, dict):
        return None
    return RolloutResult(
        steps=tuple(
            deserialize_grid_step(step)
            for step in raw.get("steps", [])
            if isinstance(step, dict)
        ),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        success=bool(raw.get("success", False)),
        total_steps=int(raw.get("total_steps", 0)),
        total_reward=float(raw.get("total_reward", 0.0)),
        collisions=int(raw.get("collisions", 0)),
        slipped_actions=int(raw.get("slipped_actions", 0)),
        trap_count=int(raw.get("trap_count", 0)),
    )


def serialize_sarsa_evaluation(summary: SarsaEvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "episodes": summary.episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "truncated_episodes": summary.truncated_episodes,
        "total_collisions": summary.total_collisions,
        "total_slipped_actions": summary.total_slipped_actions,
        "total_traps": summary.total_traps,
        "rollouts": [serialize_grid_rollout(rollout) for rollout in summary.rollouts],
    }


def deserialize_sarsa_evaluation(raw: Any) -> SarsaEvaluationSummary | None:
    if not isinstance(raw, dict):
        return None
    return SarsaEvaluationSummary(
        episodes=int(raw.get("episodes", 0)),
        successes=int(raw.get("successes", 0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        mean_return=float(raw.get("mean_return", 0.0)),
        std_return=float(raw.get("std_return", 0.0)),
        mean_steps=float(raw.get("mean_steps", 0.0)),
        mean_successful_steps=(
            float(raw["mean_successful_steps"])
            if raw.get("mean_successful_steps") is not None
            else None
        ),
        truncated_episodes=int(raw.get("truncated_episodes", 0)),
        total_collisions=int(raw.get("total_collisions", 0)),
        total_slipped_actions=int(raw.get("total_slipped_actions", 0)),
        total_traps=int(raw.get("total_traps", 0)),
        rollouts=tuple(
            rollout
            for rollout in (
                deserialize_grid_rollout(item) for item in raw.get("rollouts", [])
            )
            if rollout is not None
        ),
    )


def serialize_q_learning_evaluation(summary: QLearningEvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "episodes": summary.episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "key_collection_rate": summary.key_collection_rate,
        "mean_key_collection_step": summary.mean_key_collection_step,
        "total_locked_exit_attempts": summary.total_locked_exit_attempts,
        "truncated_episodes": summary.truncated_episodes,
        "total_collisions": summary.total_collisions,
        "total_slipped_actions": summary.total_slipped_actions,
        "total_traps": summary.total_traps,
        "rollouts": [serialize_grid_rollout(rollout) for rollout in summary.rollouts],
    }


def deserialize_q_learning_evaluation(raw: Any) -> QLearningEvaluationSummary | None:
    if not isinstance(raw, dict):
        return None
    return QLearningEvaluationSummary(
        episodes=int(raw.get("episodes", 0)),
        successes=int(raw.get("successes", 0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        mean_return=float(raw.get("mean_return", 0.0)),
        std_return=float(raw.get("std_return", 0.0)),
        mean_steps=float(raw.get("mean_steps", 0.0)),
        mean_successful_steps=(
            float(raw["mean_successful_steps"])
            if raw.get("mean_successful_steps") is not None
            else None
        ),
        key_collection_rate=float(raw.get("key_collection_rate", 0.0)),
        mean_key_collection_step=(
            float(raw["mean_key_collection_step"])
            if raw.get("mean_key_collection_step") is not None
            else None
        ),
        total_locked_exit_attempts=int(raw.get("total_locked_exit_attempts", 0)),
        truncated_episodes=int(raw.get("truncated_episodes", 0)),
        total_collisions=int(raw.get("total_collisions", 0)),
        total_slipped_actions=int(raw.get("total_slipped_actions", 0)),
        total_traps=int(raw.get("total_traps", 0)),
        rollouts=tuple(
            rollout
            for rollout in (
                deserialize_grid_rollout(item) for item in raw.get("rollouts", [])
            )
            if rollout is not None
        ),
    )


def serialize_policy_evaluation(summary: PolicyEvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "episodes": summary.episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "std_steps": summary.std_steps,
        "min_steps": summary.min_steps,
        "max_steps": summary.max_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "total_collisions": summary.total_collisions,
        "total_slipped": summary.total_slipped,
        "trajectories": [[list(pos) for pos in traj] for traj in summary.trajectories],
    }


def deserialize_policy_evaluation(raw: Any) -> PolicyEvaluationSummary | None:
    if not isinstance(raw, dict):
        return None
    return PolicyEvaluationSummary(
        episodes=int(raw.get("episodes", 0)),
        successes=int(raw.get("successes", 0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        mean_return=float(raw.get("mean_return", 0.0)),
        std_return=float(raw.get("std_return", 0.0)),
        mean_steps=float(raw.get("mean_steps", 0.0)),
        std_steps=float(raw.get("std_steps", 0.0)),
        min_steps=int(raw["min_steps"]) if raw.get("min_steps") is not None else None,
        max_steps=int(raw["max_steps"]) if raw.get("max_steps") is not None else None,
        mean_successful_steps=(
            float(raw["mean_successful_steps"])
            if raw.get("mean_successful_steps") is not None
            else None
        ),
        total_collisions=int(raw.get("total_collisions", 0)),
        total_slipped=int(raw.get("total_slipped", 0)),
        trajectories=tuple(
            tuple(_position(pos) for pos in traj)
            for traj in raw.get("trajectories", [])
        ),
    )


def serialize_continuous_step(step: ContinuousTrajectoryStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "state": list(step.state),
        "requested_action": int(step.requested_action),
        "reward": step.reward,
        "next_state": list(step.next_state),
        "collision": step.collision,
        "event": step.event,
        "terminated": step.terminated,
        "truncated": step.truncated,
        "distance_to_exit_m": step.distance_to_exit_m,
    }


def deserialize_continuous_step(raw: dict[str, Any]) -> ContinuousTrajectoryStep:
    return ContinuousTrajectoryStep(
        index=int(raw.get("index", 0)),
        state=_continuous_state(raw.get("state", (0.0, 0.0, 0, 0))),
        requested_action=VelocityAction(int(raw.get("requested_action", 0))),
        reward=float(raw.get("reward", 0.0)),
        next_state=_continuous_state(raw.get("next_state", raw.get("state", (0.0, 0.0, 0, 0)))),
        collision=raw.get("collision"),
        event=raw.get("event"),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        distance_to_exit_m=float(raw.get("distance_to_exit_m", 0.0)),
    )


def serialize_continuous_rollout(rollout: ContinuousRolloutResult | None) -> dict[str, Any] | None:
    if rollout is None:
        return None
    return {
        "seed": rollout.seed,
        "start_state": list(rollout.start_state),
        "final_state": list(rollout.final_state),
        "total_reward": rollout.total_reward,
        "steps": rollout.steps,
        "simulated_time_s": rollout.simulated_time_s,
        "success": rollout.success,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "collision_count": rollout.collision_count,
        "distance_travelled_m": rollout.distance_travelled_m,
        "trajectory": [serialize_continuous_step(step) for step in rollout.trajectory],
    }


def deserialize_continuous_rollout(raw: Any) -> ContinuousRolloutResult | None:
    if not isinstance(raw, dict):
        return None
    return ContinuousRolloutResult(
        seed=int(raw.get("seed", 0)),
        start_state=_continuous_state(raw.get("start_state", (0.0, 0.0, 0, 0))),
        final_state=_continuous_state(raw.get("final_state", raw.get("start_state", (0.0, 0.0, 0, 0)))),
        total_reward=float(raw.get("total_reward", 0.0)),
        steps=int(raw.get("steps", 0)),
        simulated_time_s=float(raw.get("simulated_time_s", 0.0)),
        success=bool(raw.get("success", False)),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        collision_count=int(raw.get("collision_count", 0)),
        distance_travelled_m=float(raw.get("distance_travelled_m", 0.0)),
        trajectory=tuple(
            deserialize_continuous_step(step)
            for step in raw.get("trajectory", [])
            if isinstance(step, dict)
        ),
    )


def serialize_approximate_evaluation(summary: ApproximateEvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "n_episodes": summary.n_episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "truncated_count": summary.truncated_count,
        "total_collisions": summary.total_collisions,
        "mean_distance_travelled_m": summary.mean_distance_travelled_m,
        "total_distance_travelled_m": summary.total_distance_travelled_m,
        "rollouts": [serialize_continuous_rollout(rollout) for rollout in summary.rollouts],
        "start_category": summary.start_category,
    }


def deserialize_approximate_evaluation(raw: Any) -> ApproximateEvaluationSummary | None:
    if not isinstance(raw, dict):
        return None
    return ApproximateEvaluationSummary(
        n_episodes=int(raw.get("n_episodes", 0)),
        successes=int(raw.get("successes", 0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        mean_return=float(raw.get("mean_return", 0.0)),
        std_return=float(raw.get("std_return", 0.0)),
        mean_steps=float(raw.get("mean_steps", 0.0)),
        mean_successful_steps=(
            float(raw["mean_successful_steps"])
            if raw.get("mean_successful_steps") is not None
            else None
        ),
        truncated_count=int(raw.get("truncated_count", 0)),
        total_collisions=int(raw.get("total_collisions", 0)),
        mean_distance_travelled_m=float(raw.get("mean_distance_travelled_m", 0.0)),
        total_distance_travelled_m=float(raw.get("total_distance_travelled_m", 0.0)),
        rollouts=tuple(
            rollout
            for rollout in (
                deserialize_continuous_rollout(item) for item in raw.get("rollouts", [])
            )
            if rollout is not None
        ),
        start_category=str(raw.get("start_category", "")),
    )


def serialize_room5_step(step: Room5TrajectoryStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "observation": list(step.observation),
        "raw_state": list(step.raw_state),
        "requested_action": int(step.requested_action),
        "reward": step.reward,
        "next_observation": list(step.next_observation),
        "next_raw_state": list(step.next_raw_state),
        "collision": step.collision,
        "event": step.event,
        "terminated": step.terminated,
        "truncated": step.truncated,
        "cumulative_reward": step.cumulative_reward,
        "visible_obstacle_count": step.visible_obstacle_count,
        "distance_to_exit_m": step.distance_to_exit_m,
    }


def deserialize_room5_step(raw: dict[str, Any]) -> Room5TrajectoryStep:
    return Room5TrajectoryStep(
        index=int(raw.get("index", 0)),
        observation=_observation(raw.get("observation", ())),
        raw_state=_continuous_state(raw.get("raw_state", (0.0, 0.0, 0, 0))),
        requested_action=VelocityAction(int(raw.get("requested_action", 0))),
        reward=float(raw.get("reward", 0.0)),
        next_observation=_observation(raw.get("next_observation", raw.get("observation", ()))),
        next_raw_state=_continuous_state(raw.get("next_raw_state", raw.get("raw_state", (0.0, 0.0, 0, 0)))),
        collision=raw.get("collision"),
        event=raw.get("event"),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        cumulative_reward=float(raw.get("cumulative_reward", 0.0)),
        visible_obstacle_count=int(raw.get("visible_obstacle_count", 0)),
        distance_to_exit_m=float(raw.get("distance_to_exit_m", 0.0)),
    )


def serialize_room5_rollout(rollout: Room5RolloutResult | None) -> dict[str, Any] | None:
    if rollout is None:
        return None
    return {
        "seed": rollout.seed,
        "layout_seed": rollout.layout_seed,
        "start_state": list(rollout.start_state),
        "final_state": list(rollout.final_state),
        "total_reward": rollout.total_reward,
        "steps": rollout.steps,
        "simulated_time_s": rollout.simulated_time_s,
        "success": rollout.success,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "boundary_collisions": rollout.boundary_collisions,
        "obstacle_collisions": rollout.obstacle_collisions,
        "visible_obstacle_steps": rollout.visible_obstacle_steps,
        "trajectory": [serialize_room5_step(step) for step in rollout.trajectory],
    }


def deserialize_room5_rollout(raw: Any) -> Room5RolloutResult | None:
    if not isinstance(raw, dict):
        return None
    return Room5RolloutResult(
        seed=int(raw.get("seed", 0)),
        layout_seed=int(raw.get("layout_seed", 0)),
        start_state=_continuous_state(raw.get("start_state", (0.0, 0.0, 0, 0))),
        final_state=_continuous_state(raw.get("final_state", raw.get("start_state", (0.0, 0.0, 0, 0)))),
        total_reward=float(raw.get("total_reward", 0.0)),
        steps=int(raw.get("steps", 0)),
        simulated_time_s=float(raw.get("simulated_time_s", 0.0)),
        success=bool(raw.get("success", False)),
        terminated=bool(raw.get("terminated", False)),
        truncated=bool(raw.get("truncated", False)),
        boundary_collisions=int(raw.get("boundary_collisions", 0)),
        obstacle_collisions=int(raw.get("obstacle_collisions", 0)),
        visible_obstacle_steps=int(raw.get("visible_obstacle_steps", 0)),
        trajectory=tuple(
            deserialize_room5_step(step)
            for step in raw.get("trajectory", [])
            if isinstance(step, dict)
        ),
    )


def serialize_dqn_evaluation(summary: DQNEvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "n_episodes": summary.n_episodes,
        "successes": summary.successes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "truncated_count": summary.truncated_count,
        "obstacle_collision_count": summary.obstacle_collision_count,
        "boundary_collision_count": summary.boundary_collision_count,
        "rollouts": [serialize_room5_rollout(rollout) for rollout in summary.rollouts],
        "category": summary.category,
    }


def deserialize_dqn_evaluation(raw: Any) -> DQNEvaluationSummary | None:
    if not isinstance(raw, dict):
        return None
    return DQNEvaluationSummary(
        n_episodes=int(raw.get("n_episodes", 0)),
        successes=int(raw.get("successes", 0)),
        success_rate=float(raw.get("success_rate", 0.0)),
        mean_return=float(raw.get("mean_return", 0.0)),
        std_return=float(raw.get("std_return", 0.0)),
        mean_steps=float(raw.get("mean_steps", 0.0)),
        mean_successful_steps=(
            float(raw["mean_successful_steps"])
            if raw.get("mean_successful_steps") is not None
            else None
        ),
        truncated_count=int(raw.get("truncated_count", 0)),
        obstacle_collision_count=int(raw.get("obstacle_collision_count", 0)),
        boundary_collision_count=int(raw.get("boundary_collision_count", 0)),
        rollouts=tuple(
            rollout
            for rollout in (
                deserialize_room5_rollout(item) for item in raw.get("rollouts", [])
            )
            if rollout is not None
        ),
        category=str(raw.get("category", "")),
    )


def serialize_value_iteration_result(result: ValueIterationResult) -> dict[str, Any]:
    return {
        "values": [
            {"state": list(state), "value": float(value)}
            for state, value in sorted(result.values.items())
        ],
        "policy": [
            {
                "state": list(state),
                "action": int(action) if action is not None else None,
            }
            for state, action in sorted(result.policy.items())
        ],
        "iterations": result.iterations,
        "converged": result.converged,
        "final_delta": result.final_delta,
        "delta_history": list(result.delta_history),
        "start_state_value": result.start_state_value,
    }


def deserialize_value_iteration_result(raw: dict[str, Any]) -> ValueIterationResult:
    values = {
        _position(row["state"]): float(row.get("value", 0.0))
        for row in raw.get("values", [])
        if isinstance(row, dict) and "state" in row
    }
    policy = {
        _position(row["state"]): (
            Action(int(row["action"]))
            if row.get("action") is not None
            else None
        )
        for row in raw.get("policy", [])
        if isinstance(row, dict) and "state" in row
    }
    return ValueIterationResult(
        values=MappingProxyType(values),
        policy=MappingProxyType(policy),
        iterations=int(raw.get("iterations", 0)),
        converged=bool(raw.get("converged", False)),
        final_delta=float(raw.get("final_delta", 0.0)),
        delta_history=tuple(float(v) for v in raw.get("delta_history", [])),
        start_state_value=float(raw.get("start_state_value", 0.0)),
    )


def value_iteration_config_dict(config: ValueIterationConfig) -> dict[str, Any]:
    return {
        "gamma": config.gamma,
        "tolerance": config.tolerance,
        "max_iterations": config.max_iterations,
        "tie_tolerance": config.tie_tolerance,
    }


def update_saved_outputs(filepath_stem: str | Path, outputs: dict[str, Any]) -> None:
    path = metadata_path(filepath_stem)
    metadata = read_json(path)
    saved_outputs = dict(metadata.get(SAVED_OUTPUTS_KEY, {}))
    saved_outputs.update(outputs)
    saved_outputs["schema_version"] = RUN_HISTORY_SCHEMA_VERSION
    metadata[SAVED_OUTPUTS_KEY] = saved_outputs
    atomic_write_json(path, metadata)


def save_room1_run(
    result: ValueIterationResult,
    filepath_stem: str | Path,
    *,
    config: ValueIterationConfig,
    slip_config,
    map_grid,
    rollout: RolloutResult | None = None,
    evaluation: PolicyEvaluationSummary | None = None,
) -> str:
    payload = {
        "schema_version": RUN_HISTORY_SCHEMA_VERSION,
        "algorithm": "Value Iteration",
        "room": "Room1DP",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "map_signature": map_signature(map_grid),
        "grid_shape": list(map_grid.shape),
        "config": value_iteration_config_dict(config),
        "slip_config": {
            "intended_probability": slip_config.intended_probability,
            "left_probability": slip_config.left_probability,
            "right_probability": slip_config.right_probability,
        },
        "result": serialize_value_iteration_result(result),
        SAVED_OUTPUTS_KEY: {
            "schema_version": RUN_HISTORY_SCHEMA_VERSION,
            "rollout": serialize_grid_rollout(rollout),
            "evaluation_summary": serialize_policy_evaluation(evaluation),
        },
    }
    path = metadata_path(filepath_stem)
    atomic_write_json(path, payload)
    return str(path)


def load_room1_run(filepath_stem: str | Path, *, map_grid) -> tuple[ValueIterationResult, dict[str, Any]]:
    metadata = read_json(metadata_path(filepath_stem))
    if metadata.get("schema_version") != RUN_HISTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported Room 1 saved-run schema version")
    if metadata.get("algorithm") != "Value Iteration":
        raise ValueError(f"Unknown Room 1 algorithm: {metadata.get('algorithm')}")
    if metadata.get("room") != "Room1DP":
        raise ValueError(f"Expected Room1DP; got {metadata.get('room')}")
    if metadata.get("map_signature") != map_signature(map_grid):
        raise ValueError("Room 1 map signature mismatch")
    return deserialize_value_iteration_result(metadata.get("result", {})), metadata


def list_saved_runs(
    directory: str | Path,
    *,
    requires_npz: bool,
) -> list[dict[str, Any]]:
    directory = Path(directory)
    if not directory.exists():
        return []
    runs: list[dict[str, Any]] = []
    for json_path in directory.glob("*.json"):
        stem = json_path.with_suffix("")
        if requires_npz and not stem.with_suffix(".npz").exists():
            continue
        try:
            metadata = read_json(json_path)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "stem": str(stem),
                "json_path": str(json_path),
                "metadata": metadata,
                "mtime": json_path.stat().st_mtime,
            }
        )
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs


def format_saved_run_label(run: dict[str, Any]) -> str:
    metadata = run.get("metadata", {})
    stem_name = Path(run["stem"]).name
    saved_at = metadata.get("saved_at")
    if not saved_at:
        saved_at = datetime.fromtimestamp(run["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
    cfg = metadata.get("training_config") or metadata.get("config") or {}
    parts = [stem_name, str(saved_at)]
    if "episodes" in cfg:
        parts.append(f"episodes={cfg['episodes']}")
    if "seed" in cfg:
        parts.append(f"seed={cfg['seed']}")
    if "iterations" in metadata.get("result", {}):
        parts.append(f"iterations={metadata['result']['iterations']}")
    return " | ".join(parts)
