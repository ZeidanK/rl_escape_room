"""Streamlit entry point for the reinforcement-learning escape room app."""

from pathlib import Path
import os

import streamlit as st

import numpy as np

APP_DIR = Path(__file__).resolve().parent

# Main Streamlit entry point.  This file wires together the environments,
# agents, visualizations, saved models, and high-level navigation modes.
from core.types import (
    Action,
    ApproximateSarsaConfig,
    ApproximateSarsaTrainingResult,
    DQNConfig,
    DQNTrainingResult,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    QLearningTrainingResult,
    QLearningConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
    SarsaConfig,
    SarsaTrainingResult,
    SlipConfig,
    StartMode,
    StepResult,
    ValueIterationConfig,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from agents.dynamic_programming import (
    ValueIterationAgent,
    evaluate_policy,
    rollout_policy,
)
from agents.sarsa import (
    SarsaAgent,
    evaluate_sarsa_policy,
    extract_greedy_policy,
    load_model,
    rollout_sarsa_policy,
    save_model,
)
from agents.q_learning import (
    QLearningAgent,
    evaluate_q_learning_policy,
    load_q_model,
    rollout_q_learning_policy,
    save_q_model,
)
from visualization.dp_visualization import (
    build_value_matrix,
)
from visualization.sarsa_visualization import (
    build_q_value_tables,
    build_training_dataframe,
    render_sarsa_trajectory_overlay,
)
from visualization.q_learning_visualization import (
    build_q_learning_training_dataframe,
    build_room3_q_value_table,
    render_q_learning_trajectory_overlay,
)
from training.algorithm_comparison import (
    run_matched_comparison,
    run_tuned_comparison,
    save_comparison,
)
from environments.room4_continuous import ContinuousRewardConfig, Room4Continuous, Room4MotionConfig
from environments.room5_obstacles import Room5Obstacles
from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    evaluate_approximate_policy,
    LinearTileQFunction,
    load_approximate_model,
    rollout_approximate_policy,
    save_approximate_model,
)
from agents.dqn import (
    DQNAgent,
    DQNNetwork,
    evaluate_dqn_policy,
    extract_dqn_action_values,
    load_dqn_model,
    make_dqn_result_session_safe,
    rollout_dqn_policy,
    save_dqn_model,
)
from features.tile_coding import TileCoder, TileCodingConfig
from visualization.approximate_sarsa_visualization import (
    build_action_field as build_approx_action_field,
    build_training_dataframe as build_approx_training_dataframe,
    build_value_surface as build_approx_value_surface,
)
from visualization.room5_visualization import render_room5_svg as render_room5_obstacle_svg
from training.approximate_sarsa_experiments import (
    run_confirmation_experiments as run_approx_confirmation,
    run_screening_stage_a,
)
from training.result_persistence import (
    SAVED_OUTPUTS_KEY,
    deserialize_approximate_evaluation,
    deserialize_approximate_metrics,
    deserialize_continuous_rollout,
    deserialize_dqn_evaluation,
    deserialize_dqn_metrics,
    deserialize_grid_rollout,
    deserialize_policy_evaluation,
    deserialize_q_learning_evaluation,
    deserialize_q_learning_metrics,
    deserialize_room5_rollout,
    deserialize_sarsa_evaluation,
    deserialize_sarsa_metrics,
    format_saved_run_label,
    list_saved_runs,
    load_room1_run,
    save_room1_run,
    serialize_approximate_evaluation,
    serialize_continuous_rollout,
    serialize_dqn_evaluation,
    serialize_grid_rollout,
    serialize_policy_evaluation,
    serialize_q_learning_evaluation,
    serialize_room5_rollout,
    serialize_sarsa_evaluation,
    timestamp_slug,
    update_saved_outputs,
)

ROOM_CLASSES = {
    # Manual-play mode lets the user choose one of the grid rooms directly.
    "Room 1 — Frozen Maze (DP)": Room1DP,
    "Room 2 — Laser Corridor (SARSA)": Room2SARSA,
    "Room 3 — Key Vault (Q-Learning)": Room3QLearning,
}

ACTION_BUTTONS = {
    "UP": Action.UP,
    "RIGHT": Action.RIGHT,
    "DOWN": Action.DOWN,
    "LEFT": Action.LEFT,
}


def _epsilon_config_from_metadata(raw: dict, fallback: EpsilonScheduleConfig) -> EpsilonScheduleConfig:
    # Saved models may have older or partial metadata, so the fallback keeps
    # loading robust while preserving the trained schedule when available.
    try:
        kind = EpsilonDecayKind(raw.get("kind", fallback.kind.value))
    except ValueError:
        kind = fallback.kind
    return EpsilonScheduleConfig(
        kind=kind,
        start=float(raw.get("start", fallback.start)),
        minimum=float(raw.get("minimum", fallback.minimum)),
        decay=float(raw.get("decay", fallback.decay)),
        linear_decay_episodes=int(raw.get("linear_decay_episodes", fallback.linear_decay_episodes)),
    )


def _tile_coding_config_from_metadata(raw: dict, fallback: TileCodingConfig) -> TileCodingConfig:
    # Loaded approximate-SARSA weights must be interpreted with the exact
    # tile-coding shape used during training, independent of sidebar defaults.
    tc = raw.get("tile_coding_config", {})
    return TileCodingConfig(
        num_tilings=int(tc.get("num_tilings", fallback.num_tilings)),
        tiles_x=int(tc.get("tiles_x", fallback.tiles_x)),
        tiles_y=int(tc.get("tiles_y", fallback.tiles_y)),
        include_velocity=bool(tc.get("include_velocity", fallback.include_velocity)),
    )


def _loaded_sarsa_result(
    q_values,
    metadata: dict,
    fallback_config: SarsaConfig,
) -> SarsaTrainingResult:
    # Reconstruct just enough of a training result for the UI from saved
    # Q-values and any persisted training history.
    cfg = metadata.get("training_config") or metadata.get("config", {})
    epsilon = _epsilon_config_from_metadata(cfg.get("epsilon", {}), fallback_config.epsilon)
    config = SarsaConfig(
        episodes=int(cfg.get("episodes", fallback_config.episodes)),
        alpha=float(cfg.get("alpha", fallback_config.alpha)),
        gamma=float(cfg.get("gamma", fallback_config.gamma)),
        max_steps=int(cfg.get("max_steps", fallback_config.max_steps)),
        seed=int(cfg.get("seed", fallback_config.seed)),
        epsilon=epsilon,
    )
    training = metadata.get("training", {})
    return SarsaTrainingResult(
        config=config,
        q_values=q_values,
        metrics=deserialize_sarsa_metrics(metadata.get("training_metrics")),
        snapshots={},
        final_epsilon=float(training.get("final_epsilon", epsilon.minimum)),
        training_seed=config.seed,
    )


def _loaded_q_learning_result(
    q_values,
    metadata: dict,
    fallback_config: QLearningConfig,
) -> QLearningTrainingResult:
    # Same idea as SARSA loading, but the Q-table keys include has_key.
    cfg = metadata.get("training_config", {})
    epsilon = _epsilon_config_from_metadata(cfg.get("epsilon", {}), fallback_config.epsilon)
    config = QLearningConfig(
        episodes=int(cfg.get("episodes", fallback_config.episodes)),
        alpha=float(cfg.get("alpha", fallback_config.alpha)),
        gamma=float(cfg.get("gamma", fallback_config.gamma)),
        max_steps=int(cfg.get("max_steps", fallback_config.max_steps)),
        seed=int(cfg.get("seed", metadata.get("training_seed", fallback_config.seed))),
        epsilon=epsilon,
    )
    return QLearningTrainingResult(
        config=config,
        q_values=q_values,
        metrics=deserialize_q_learning_metrics(metadata.get("training_metrics")),
        snapshots={},
        final_epsilon=float(metadata.get("final_epsilon", epsilon.minimum)),
        training_seed=int(metadata.get("training_seed", config.seed)),
    )


def _loaded_approximate_result(
    weights,
    metadata: dict,
    fallback_config: ApproximateSarsaConfig,
    tile_coding_config: TileCodingConfig,
) -> ApproximateSarsaTrainingResult:
    # Rebuild the result wrapper around loaded linear weights so plotting and
    # evaluation code can use the same interface as freshly trained models.
    cfg = metadata.get("training_config", {})
    epsilon = _epsilon_config_from_metadata(cfg.get("epsilon", {}), fallback_config.epsilon)
    try:
        start_mode = StartMode(cfg.get("start_mode", fallback_config.start_mode.value))
    except ValueError:
        start_mode = fallback_config.start_mode
    config = ApproximateSarsaConfig(
        episodes=int(cfg.get("episodes", fallback_config.episodes)),
        alpha=float(cfg.get("alpha", fallback_config.alpha)),
        gamma=float(cfg.get("gamma", fallback_config.gamma)),
        max_steps=int(cfg.get("max_steps", fallback_config.max_steps)),
        seed=int(cfg.get("seed", metadata.get("training_seed", fallback_config.seed))),
        epsilon=epsilon,
        tile_coding=tile_coding_config,
        start_mode=start_mode,
    )
    return ApproximateSarsaTrainingResult(
        config=config,
        weights=weights,
        metrics=deserialize_approximate_metrics(metadata.get("training_metrics")),
        snapshots={},
        final_epsilon=float(metadata.get("final_epsilon", epsilon.minimum)),
        training_seed=int(metadata.get("training_seed", config.seed)),
    )


def _loaded_dqn_result(
    network: DQNNetwork,
    metadata: dict,
    fallback_config: DQNConfig,
) -> DQNTrainingResult:
    # Wrap a loaded NumPy network in the training-result shape expected by the
    # Room 5 UI.
    cfg = metadata.get("training_config", {})
    epsilon = _epsilon_config_from_metadata(cfg.get("epsilon", {}), fallback_config.epsilon)
    config = DQNConfig(
        episodes=int(cfg.get("episodes", fallback_config.episodes)),
        learning_rate=float(cfg.get("learning_rate", fallback_config.learning_rate)),
        gamma=float(cfg.get("gamma", fallback_config.gamma)),
        max_steps=int(cfg.get("max_steps", fallback_config.max_steps)),
        seed=int(metadata.get("training_seed", fallback_config.seed)),
        epsilon=epsilon,
        replay_capacity=int(cfg.get("replay_capacity", fallback_config.replay_capacity)),
        batch_size=int(cfg.get("batch_size", fallback_config.batch_size)),
        warmup_steps=int(cfg.get("warmup_steps", fallback_config.warmup_steps)),
        target_update_interval=int(cfg.get("target_update_interval", fallback_config.target_update_interval)),
        hidden_units=int(metadata.get("hidden_units", fallback_config.hidden_units)),
    )
    return DQNTrainingResult(
        config=config,
        weights=network.weights,
        metrics=deserialize_dqn_metrics(metadata.get("training_metrics")),
        snapshots={},
        final_epsilon=float(metadata.get("final_epsilon", epsilon.minimum)),
        training_seed=config.seed,
        input_dim=int(metadata.get("input_dim", network.input_dim)),
        action_count=int(metadata.get("action_count", network.action_count)),
    )


def _preferred_model_stem(model_dir: str, showcase_stem: str) -> str | None:
    # Prefer committed showcase artifacts, then fall back to the newest local
    # model that has both metadata (.json) and weights (.npz).
    import glob
    import os

    showcase = os.path.join(model_dir, showcase_stem)
    if os.path.exists(showcase + ".json") and os.path.exists(showcase + ".npz"):
        return showcase
    files = glob.glob(os.path.join(model_dir, "*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return latest.replace(".json", "")


def _preferred_saved_run_stem(model_dir: str, *, requires_npz: bool) -> str | None:
    runs = list_saved_runs(model_dir, requires_npz=requires_npz)
    return runs[0]["stem"] if runs else None


def _auto_run_stem(model_dir: str, prefix: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{prefix}_{timestamp_slug()}")


def _saved_run_selector(room_key: str, model_dir: str, *, requires_npz: bool) -> str | None:
    runs = list_saved_runs(model_dir, requires_npz=requires_npz)
    if not runs:
        st.caption("No saved runs found yet.")
        return None
    labels = [format_saved_run_label(run) for run in runs]
    selected = st.selectbox("Saved Runs", options=labels, key=f"{room_key}_saved_runs")
    return runs[labels.index(selected)]["stem"]


def _is_mutable_saved_run(filepath_stem: str | None) -> bool:
    if not filepath_stem:
        return False
    return not os.path.basename(filepath_stem).startswith("showcase")


def _restore_room1_outputs_from_metadata(metadata: dict) -> None:
    outputs = metadata.get(SAVED_OUTPUTS_KEY, {})
    st.session_state.vi_rollout_result = deserialize_grid_rollout(outputs.get("rollout"))
    st.session_state.vi_eval_summary = deserialize_policy_evaluation(outputs.get("evaluation_summary"))
    st.session_state.vi_rollout_key = ("saved", metadata.get("saved_at"))
    st.session_state.vi_eval_key = ("saved", metadata.get("saved_at"))


def _restore_room2_outputs_from_metadata(metadata: dict) -> None:
    outputs = metadata.get(SAVED_OUTPUTS_KEY, {})
    st.session_state.sarsa_eval_summary = deserialize_sarsa_evaluation(outputs.get("evaluation_summary"))
    st.session_state.sarsa_rollout = deserialize_grid_rollout(outputs.get("rollout"))
    st.session_state.sarsa_eval_key = ("saved", metadata.get("saved_at"))
    st.session_state.sarsa_rollout_key = ("saved", metadata.get("saved_at"))


def _restore_room3_outputs_from_metadata(metadata: dict) -> None:
    outputs = metadata.get(SAVED_OUTPUTS_KEY, {})
    st.session_state.ql_eval_summary = deserialize_q_learning_evaluation(outputs.get("evaluation_summary"))
    st.session_state.ql_rollout = deserialize_grid_rollout(outputs.get("rollout"))
    st.session_state.ql_eval_key = ("saved", metadata.get("saved_at"))
    st.session_state.ql_rollout_key = ("saved", metadata.get("saved_at"))


def _restore_room4_outputs_from_metadata(metadata: dict) -> None:
    outputs = metadata.get(SAVED_OUTPUTS_KEY, {})
    st.session_state.approx_eval_fixed = deserialize_approximate_evaluation(outputs.get("eval_fixed"))
    st.session_state.approx_eval_gen = deserialize_approximate_evaluation(outputs.get("eval_gen"))
    st.session_state.approx_rollout = deserialize_continuous_rollout(outputs.get("rollout"))
    st.session_state.approx_eval_fixed_key = ("saved", metadata.get("saved_at"), "fixed")
    st.session_state.approx_eval_gen_key = ("saved", metadata.get("saved_at"), "gen")
    st.session_state.approx_rollout_key = ("saved", metadata.get("saved_at"))


def _restore_room5_outputs_from_metadata(metadata: dict) -> None:
    outputs = metadata.get(SAVED_OUTPUTS_KEY, {})
    st.session_state.dqn_eval_fixed = deserialize_dqn_evaluation(outputs.get("eval_fixed"))
    st.session_state.dqn_eval_random = deserialize_dqn_evaluation(outputs.get("eval_random"))
    st.session_state.dqn_eval_unseen = deserialize_dqn_evaluation(outputs.get("eval_unseen"))
    st.session_state.dqn_rollout = deserialize_room5_rollout(outputs.get("rollout"))
    st.session_state.dqn_rollout_fixed_layout = outputs.get("rollout_fixed_layout")
    st.session_state.dqn_rollout_key = ("saved", metadata.get("saved_at"))


def _persist_room1_outputs_if_saved() -> None:
    stem = st.session_state.get("vi_model_stem")
    result = st.session_state.get("vi_result")
    env = st.session_state.get("dp_env")
    config = st.session_state.get("vi_config")
    slip_cfg = st.session_state.get("vi_slip_config")
    if not (_is_mutable_saved_run(stem) and result is not None and env is not None and config is not None and slip_cfg is not None):
        return
    save_room1_run(
        result,
        stem,
        config=config,
        slip_config=slip_cfg,
        map_grid=env.grid,
        rollout=st.session_state.get("vi_rollout_result"),
        evaluation=st.session_state.get("vi_eval_summary"),
    )


def _persist_room2_outputs_if_saved() -> None:
    stem = st.session_state.get("sarsa_model_stem")
    if not _is_mutable_saved_run(stem):
        return
    update_saved_outputs(
        stem,
        {
            "evaluation_summary": serialize_sarsa_evaluation(st.session_state.get("sarsa_eval_summary")),
            "rollout": serialize_grid_rollout(st.session_state.get("sarsa_rollout")),
        },
    )


def _persist_room3_outputs_if_saved() -> None:
    stem = st.session_state.get("ql_model_stem")
    if not _is_mutable_saved_run(stem):
        return
    update_saved_outputs(
        stem,
        {
            "evaluation_summary": serialize_q_learning_evaluation(st.session_state.get("ql_eval_summary")),
            "rollout": serialize_grid_rollout(st.session_state.get("ql_rollout")),
        },
    )


def _persist_room4_outputs_if_saved() -> None:
    stem = st.session_state.get("approx_model_stem")
    if not _is_mutable_saved_run(stem):
        return
    update_saved_outputs(
        stem,
        {
            "eval_fixed": serialize_approximate_evaluation(st.session_state.get("approx_eval_fixed")),
            "eval_gen": serialize_approximate_evaluation(st.session_state.get("approx_eval_gen")),
            "rollout": serialize_continuous_rollout(st.session_state.get("approx_rollout")),
        },
    )


def _persist_room5_outputs_if_saved() -> None:
    stem = st.session_state.get("dqn_model_stem")
    if not _is_mutable_saved_run(stem):
        return
    update_saved_outputs(
        stem,
        {
            "eval_fixed": serialize_dqn_evaluation(st.session_state.get("dqn_eval_fixed")),
            "eval_random": serialize_dqn_evaluation(st.session_state.get("dqn_eval_random")),
            "eval_unseen": serialize_dqn_evaluation(st.session_state.get("dqn_eval_unseen")),
            "rollout": serialize_room5_rollout(st.session_state.get("dqn_rollout")),
            "rollout_fixed_layout": st.session_state.get("dqn_rollout_fixed_layout"),
        },
    )


def _room5_training_rows(metrics) -> list[dict]:
    return [
        {
            "episode": m.episode + 1,
            "total_reward": m.total_reward,
            "steps": m.steps,
            "success": m.success,
            "epsilon": m.epsilon,
            "obstacle_collisions": m.obstacle_collisions,
            "boundary_collisions": m.boundary_collisions,
            "visible_obstacle_steps": m.visible_obstacle_steps,
            "mean_loss": m.mean_loss,
            "mean_abs_td_error": m.mean_abs_td_error,
        }
        for m in metrics
    ]


def _clear_sarsa_outputs() -> None:
    for key in ["sarsa_eval_summary", "sarsa_eval_key", "sarsa_rollout", "sarsa_rollout_key"]:
        st.session_state[key] = None


def _clear_room1_outputs() -> None:
    for key in ["vi_rollout_result", "vi_rollout_key", "vi_eval_summary", "vi_eval_key"]:
        st.session_state[key] = None


def _load_room1_run_into_state(filepath_stem: str) -> None:
    result, meta = load_room1_run(filepath_stem, map_grid=Room1DP().grid)
    cfg = meta.get("config", {})
    slip = meta.get("slip_config", {})
    slip_cfg = SlipConfig(
        float(slip.get("intended_probability", 0.8)),
        float(slip.get("left_probability", 0.1)),
        float(slip.get("right_probability", 0.1)),
    )
    config = ValueIterationConfig(
        gamma=float(cfg.get("gamma", 0.95)),
        tolerance=float(cfg.get("tolerance", 1e-6)),
        max_iterations=int(cfg.get("max_iterations", 10_000)),
        tie_tolerance=float(cfg.get("tie_tolerance", 1e-12)),
    )
    st.session_state.dp_env = Room1DP(slip_config=slip_cfg, max_steps=200, seed=42)
    st.session_state.vi_result = result
    st.session_state.vi_config = config
    st.session_state.vi_slip_config = slip_cfg
    st.session_state.vi_solve_key = ("loaded", filepath_stem)
    st.session_state.vi_model_stem = filepath_stem
    st.session_state.vi_autoload_error = None
    st.session_state.vi_autoload_disabled = False
    _clear_room1_outputs()
    _restore_room1_outputs_from_metadata(meta)


def _autoload_room1_saved_run() -> bool:
    if st.session_state.vi_result is not None or st.session_state.get("vi_autoload_disabled"):
        return False
    model_dir = os.path.join("storage", "models", "room1_value_iteration")
    stem = _preferred_saved_run_stem(model_dir, requires_npz=False)
    if stem is None:
        return False
    try:
        _load_room1_run_into_state(stem)
    except ValueError as e:
        st.session_state.vi_autoload_error = str(e)
        return False
    return True


def _load_room2_sarsa_model_into_state(filepath_stem: str, sarsa_config: SarsaConfig) -> None:
    q_vals, meta = load_model(filepath_stem, map_grid=ROOM2_GRID)
    _clear_sarsa_outputs()
    st.session_state.sarsa_result = _loaded_sarsa_result(q_vals, meta, sarsa_config)
    st.session_state.sarsa_train_key = ("loaded", filepath_stem)
    st.session_state.sarsa_model_stem = filepath_stem
    st.session_state.sarsa_autoload_error = None
    st.session_state.sarsa_autoload_disabled = False
    _restore_room2_outputs_from_metadata(meta)


def _autoload_room2_sarsa_showcase(sarsa_config: SarsaConfig) -> bool:
    if st.session_state.sarsa_result is not None or st.session_state.get("sarsa_autoload_disabled"):
        return False

    import os

    model_dir = os.path.join("storage", "models", "room2_sarsa")
    stem = _preferred_saved_run_stem(model_dir, requires_npz=True)
    if stem is None:
        return False

    try:
        _load_room2_sarsa_model_into_state(stem, sarsa_config)
    except ValueError as e:
        st.session_state.sarsa_autoload_error = str(e)
        return False
    return True


def _clear_q_learning_outputs() -> None:
    for key in ["ql_eval_summary", "ql_eval_key", "ql_rollout", "ql_rollout_key"]:
        st.session_state[key] = None


def _load_room3_q_model_into_state(filepath_stem: str, ql_config: QLearningConfig) -> None:
    q_vals, meta = load_q_model(filepath_stem, map_grid=ROOM3_GRID)
    _clear_q_learning_outputs()
    st.session_state.ql_result = _loaded_q_learning_result(q_vals, meta, ql_config)
    st.session_state.ql_train_key = ("loaded", filepath_stem)
    st.session_state.ql_model_stem = filepath_stem
    st.session_state.ql_autoload_error = None
    st.session_state.ql_autoload_disabled = False
    _restore_room3_outputs_from_metadata(meta)


def _autoload_room3_q_showcase(ql_config: QLearningConfig) -> bool:
    if st.session_state.ql_result is not None or st.session_state.get("ql_autoload_disabled"):
        return False

    import os

    model_dir = os.path.join("storage", "models", "room3_q_learning")
    stem = _preferred_saved_run_stem(model_dir, requires_npz=True)
    if stem is None:
        return False

    try:
        _load_room3_q_model_into_state(stem, ql_config)
    except ValueError as e:
        st.session_state.ql_autoload_error = str(e)
        return False
    return True


def _clear_room4_outputs() -> None:
    for key in [
        "approx_eval_fixed",
        "approx_eval_fixed_key",
        "approx_eval_gen",
        "approx_eval_gen_key",
        "approx_rollout",
        "approx_rollout_key",
    ]:
        st.session_state[key] = None


def _load_room4_model_into_state(
    filepath_stem: str,
    approx_config: ApproximateSarsaConfig,
    tile_coding_config: TileCodingConfig,
) -> None:
    weights, meta = load_approximate_model(filepath_stem)
    loaded_tc_cfg = _tile_coding_config_from_metadata(meta, tile_coding_config)
    _clear_room4_outputs()
    st.session_state.approx_result = _loaded_approximate_result(weights, meta, approx_config, loaded_tc_cfg)
    st.session_state.approx_train_key = ("loaded", filepath_stem)
    st.session_state.approx_model_stem = filepath_stem
    st.session_state.approx_autoload_error = None
    st.session_state.approx_autoload_disabled = False
    _restore_room4_outputs_from_metadata(meta)


def _autoload_room4_showcase(
    approx_config: ApproximateSarsaConfig,
    tile_coding_config: TileCodingConfig,
) -> bool:
    if st.session_state.approx_result is not None or st.session_state.get("approx_autoload_disabled"):
        return False

    import os

    model_dir = os.path.join("storage", "models", "room4_approximate_sarsa")
    stem = _preferred_saved_run_stem(model_dir, requires_npz=True)
    if stem is None:
        return False

    try:
        _load_room4_model_into_state(stem, approx_config, tile_coding_config)
    except ValueError as e:
        st.session_state.approx_autoload_error = str(e)
        return False
    return True


def _approx_q_function_from_weights(
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    motion_config: Room4MotionConfig,
) -> LinearTileQFunction:
    tile_coder = TileCoder(
        tile_coding_config,
        room_width=motion_config.room_width_m,
        room_height=motion_config.room_height_m,
    )
    q_func = LinearTileQFunction(tile_coder, n_actions=9)
    q_func._weights = weights.copy()
    return q_func


def _clear_room5_outputs() -> None:
    for key in [
        "dqn_eval_fixed",
        "dqn_eval_random",
        "dqn_eval_unseen",
        "dqn_rollout",
        "dqn_rollout_key",
        "dqn_rollout_fixed_layout",
    ]:
        st.session_state[key] = None


def _load_room5_model_into_state(filepath_stem: str, dqn_config: DQNConfig) -> None:
    network, meta = load_dqn_model(filepath_stem)
    _clear_room5_outputs()
    st.session_state.dqn_network = network
    st.session_state.dqn_meta = meta
    st.session_state.dqn_result = _loaded_dqn_result(network, meta, dqn_config)
    st.session_state.dqn_train_key = ("loaded", filepath_stem)
    st.session_state.dqn_model_stem = filepath_stem
    st.session_state.dqn_result_source = "loaded"
    st.session_state.dqn_autoload_error = None
    st.session_state.dqn_autoload_disabled = False
    _restore_room5_outputs_from_metadata(meta)


def _autoload_room5_showcase(dqn_config: DQNConfig) -> bool:
    if st.session_state.dqn_result is not None or st.session_state.get("dqn_autoload_disabled"):
        return False

    import os

    model_dir = os.path.join("storage", "models", "room5_dqn")
    stem = _preferred_saved_run_stem(model_dir, requires_npz=True)
    if stem is None:
        return False

    try:
        _load_room5_model_into_state(stem, dqn_config)
    except ValueError as e:
        st.session_state.dqn_autoload_error = str(e)
        return False
    return True


def _render_room5_svg(env: Room5Obstacles, rollout=None) -> str:
    # Room 5 uses a custom SVG because its continuous obstacle layout does not
    # fit the grid renderer used by Rooms 1-3.
    return render_room5_obstacle_svg(env, rollout)


def _room5_display_env_for_rollout(make_env, rollout, *, fixed_layout: bool) -> Room5Obstacles:
    # Rebuild the same layout before drawing a rollout recorded during
    # evaluation or greedy replay.
    disp_env = make_env(fixed_layout=bool(fixed_layout), layout_seed=int(rollout.layout_seed))
    disp_env.reset(seed=int(rollout.seed), layout_seed=int(rollout.layout_seed))
    return disp_env


def _load_final_comparison_payload() -> dict | None:
    matched = read_json("storage/experiments/final/sarsa_vs_q_learning_matched.json")
    if not matched:
        return None
    tuned = read_json("storage/experiments/final/sarsa_vs_q_learning_tuned.json")
    if tuned and tuned.get("tuned_comparison"):
        matched["tuned_comparison"] = tuned["tuned_comparison"]
    return matched


def _row_dict(row) -> dict:
    return row if isinstance(row, dict) else vars(row)


def _matched_parts(comp_matched) -> tuple[list[dict], list[dict], list[dict]]:
    if isinstance(comp_matched, dict):
        return (
            [_row_dict(r) for r in comp_matched.get("sarsa", [])],
            [_row_dict(r) for r in comp_matched.get("q_learning", [])],
            [_row_dict(r) for r in comp_matched.get("paired_differences", [])],
        )

    sarsa_rows, q_rows = comp_matched
    paired = []
    for s_row, q_row in zip(sarsa_rows, q_rows):
        s = _row_dict(s_row)
        q = _row_dict(q_row)
        paired.append(
            {
                "seed": s.get("seed"),
                "diff_success_rate": float(q.get("success_rate", 0.0)) - float(s.get("success_rate", 0.0)),
                "diff_mean_return": float(q.get("mean_return", 0.0)) - float(s.get("mean_return", 0.0)),
                "diff_mean_steps": float(q.get("mean_steps", 0.0)) - float(s.get("mean_steps", 0.0)),
            }
        )
    return [_row_dict(r) for r in sarsa_rows], [_row_dict(r) for r in q_rows], paired


def _saved_comparison_into_state() -> bool:
    payload = _load_final_comparison_payload()
    if not payload:
        return False
    st.session_state.comp_matched = payload.get("matched_comparison")
    st.session_state.comp_tuned = payload.get("tuned_comparison")
    st.session_state.comp_metadata = payload.get("_metadata", {})
    st.session_state.comp_source = "Final saved comparison"
    st.session_state.comp_key = ("saved", "storage/experiments/final")
    return True


def _metric_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _manual_agent_comparison(room_name: str, seed: int, max_steps: int) -> dict:
    try:
        if "Room 1" in room_name:
            env = Room1DP(max_steps=max_steps, seed=seed)
            vi = ValueIterationAgent(env, ValueIterationConfig()).solve()
            rollout = rollout_policy(env, vi.policy, seed=seed, max_steps=max_steps)
            return {
                "available": True,
                "agent": "Value Iteration",
                "steps": rollout.total_steps,
                "return": rollout.total_reward,
                "success": rollout.success,
            }

        if "Room 2" in room_name:
            stem = _preferred_model_stem("storage/models/room2_sarsa", "showcase_sarsa")
            if stem is None:
                return {"available": False, "message": "agent comparison unavailable: no saved SARSA model"}
            q_values, _ = load_model(stem, map_grid=ROOM2_GRID)
            rollout = rollout_sarsa_policy(lambda: Room2SARSA(max_steps=max_steps), q_values, seed=seed, max_steps=max_steps)
            return {
                "available": True,
                "agent": "SARSA saved model",
                "steps": rollout.total_steps,
                "return": rollout.total_reward,
                "success": rollout.success,
            }

        if "Room 3" in room_name:
            stem = _preferred_model_stem("storage/models/room3_q_learning", "showcase_ql")
            if stem is None:
                return {"available": False, "message": "agent comparison unavailable: no saved Q-Learning model"}
            q_values, _ = load_q_model(stem, map_grid=ROOM3_GRID)
            rollout = rollout_q_learning_policy(lambda: Room3QLearning(max_steps=max_steps), q_values, seed=seed, max_steps=max_steps)
            return {
                "available": True,
                "agent": "Q-Learning saved model",
                "steps": rollout.total_steps,
                "return": rollout.total_reward,
                "success": rollout.success,
            }
    except Exception as exc:
        return {"available": False, "message": f"agent comparison unavailable: {exc}"}

    return {"available": False, "message": "agent comparison unavailable for this room"}

st.set_page_config(page_title="RL Escape Room", layout="wide", page_icon="🧊")
st.title("RL Escape Room")

# --- Session state ---
# Streamlit reruns the script after each interaction.  Session state preserves
# trained models, selected rooms, replays, and cached evaluation summaries
# between those reruns.
for key in [
    "env", "last_result", "room_key", "manual_total_reward", "manual_agent_comparison", "manual_compare_key",
    "vi_result", "vi_solve_key", "vi_rollout_result", "vi_rollout_key",
    "vi_eval_summary", "vi_eval_key", "vi_model_stem", "vi_autoload_error",
    "vi_config", "vi_slip_config", "dp_env",
    "sarsa_result", "sarsa_train_key", "sarsa_eval_summary", "sarsa_eval_key",
    "sarsa_rollout", "sarsa_rollout_key", "sarsa_env_factory",
    "sarsa_model_stem", "sarsa_autoload_error",
    "ql_result", "ql_train_key", "ql_eval_summary", "ql_eval_key",
    "ql_rollout", "ql_rollout_key", "ql_env_factory",
    "ql_model_stem", "ql_autoload_error",
    "comp_matched", "comp_tuned", "comp_key", "comp_source", "comp_metadata",
    "approx_result", "approx_train_key",
    "approx_eval_fixed", "approx_eval_fixed_key",
    "approx_eval_gen", "approx_eval_gen_key",
    "approx_rollout", "approx_rollout_key",
    "approx_env_factory",
    "approx_model_stem", "approx_autoload_error",
    "dqn_result", "dqn_network", "dqn_meta", "dqn_train_key",
    "dqn_eval_fixed", "dqn_eval_random", "dqn_eval_unseen",
    "dqn_rollout", "dqn_rollout_key", "dqn_rollout_fixed_layout",
    "dqn_model_stem", "dqn_autoload_error", "dqn_result_source",
    "game_mode", "game_room", "show_lab",
]:
    if key not in st.session_state:
        st.session_state[key] = None
if "dqn_autoload_disabled" not in st.session_state:
    st.session_state.dqn_autoload_disabled = False
if "approx_autoload_disabled" not in st.session_state:
    st.session_state.approx_autoload_disabled = False
if "sarsa_autoload_disabled" not in st.session_state:
    st.session_state.sarsa_autoload_disabled = False
if "ql_autoload_disabled" not in st.session_state:
    st.session_state.ql_autoload_disabled = False
if "vi_autoload_disabled" not in st.session_state:
    st.session_state.vi_autoload_disabled = False
if "mode" not in st.session_state:
    st.session_state.mode = "Escape Room Showcase"

# ============================================================
# Game mode imports
# ============================================================
# Imported after page setup because these modules render Streamlit content and
# depend on the app-wide theme/session state.
from game.home_page import render_home_page
from game.room1_game import render_room1_game
from game.room2_game import render_room2_game
from game.room3_game import render_room3_game
from game.room4_game import render_room4_game
from game.room5_game import render_room5_game
from game.theme import render_global_styles
from game.achievements import AchievementTracker
from game.canvas_renderer import (
    render_action_field_canvas,
    render_continuous_trajectory_canvas,
    render_policy_grid_canvas,
)
from game.html_rendering import render_html
from game.constants import (
    ABOUT_MODE,
    COMPARISON_MODE,
    LAB_MODE,
    LEGACY_HOME_MODE,
    MANUAL_MODE_LABEL,
    MODE_SELECTOR_KEY,
    PENDING_MODE_SELECTOR_KEY,
    PENDING_SHOWCASE_ROOM_SELECTOR_KEY,
    ROOM1_LAB_MODE,
    ROOM2_LAB_MODE,
    ROOM3_LAB_MODE,
    ROOM4_LAB_MODE,
    ROOM5_BONUS_MODE,
    SHOWCASE_MODE,
    SHOWCASE_ROOM_SELECTOR_KEY,
)
from game.presentation import (
    apply_query_params_once,
    final_summary_success,
    go_to_showcase_room,
    render_assignment_proof,
    render_model_provenance,
    render_public_project_links,
    read_json,
)

MODE_LABELS = [
    SHOWCASE_MODE,
    LAB_MODE,
    MANUAL_MODE_LABEL,
    COMPARISON_MODE,
    ABOUT_MODE,
]

# --- Mode selector ---
GAME_LABEL = SHOWCASE_MODE
LAB_LABEL = LAB_MODE
ABOUT_LABEL = ABOUT_MODE

# Selectable = game showcase, analysis rooms, manual, about
SELECTABLE_MODES = [
    GAME_LABEL,
    LAB_LABEL,
    MANUAL_MODE_LABEL,
    COMPARISON_MODE,
    ABOUT_LABEL,
]

_MODE_NAME_MAP = {
    GAME_LABEL: GAME_LABEL,
    LAB_LABEL: LAB_LABEL,
    MANUAL_MODE_LABEL: "Manual Environment",
    ABOUT_LABEL: ABOUT_LABEL,
    COMPARISON_MODE: COMPARISON_MODE,
    LEGACY_HOME_MODE: LAB_LABEL,
    ROOM1_LAB_MODE: ROOM1_LAB_MODE,
    ROOM2_LAB_MODE: ROOM2_LAB_MODE,
    ROOM3_LAB_MODE: ROOM3_LAB_MODE,
    ROOM4_LAB_MODE: ROOM4_LAB_MODE,
    ROOM5_BONUS_MODE: ROOM5_BONUS_MODE,
}

LAB_SECTION_MODES = {
    LAB_LABEL,
    ROOM1_LAB_MODE,
    ROOM2_LAB_MODE,
    ROOM3_LAB_MODE,
    ROOM4_LAB_MODE,
    ROOM5_BONUS_MODE,
}

apply_query_params_once()
if st.session_state.mode == LEGACY_HOME_MODE:
    st.session_state.mode = LAB_LABEL

if PENDING_MODE_SELECTOR_KEY in st.session_state:
    pending_selector = st.session_state[PENDING_MODE_SELECTOR_KEY]
    del st.session_state[PENDING_MODE_SELECTOR_KEY]
    if pending_selector in SELECTABLE_MODES:
        st.session_state.mode = _MODE_NAME_MAP.get(pending_selector, pending_selector)
        if MODE_SELECTOR_KEY in st.session_state:
            del st.session_state[MODE_SELECTOR_KEY]

if PENDING_SHOWCASE_ROOM_SELECTOR_KEY in st.session_state:
    del st.session_state[PENDING_SHOWCASE_ROOM_SELECTOR_KEY]
    if SHOWCASE_ROOM_SELECTOR_KEY in st.session_state:
        del st.session_state[SHOWCASE_ROOM_SELECTOR_KEY]

# Custom sidebar with categorized radio buttons
render_html(
    '<div style="font-size:0.75em;color:#616161;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Navigation</div>',
    target=st.sidebar.markdown,
)

# High contrast mode toggle
if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = False

if st.sidebar.checkbox("High Contrast Mode", value=st.session_state.high_contrast, key="high_contrast_toggle"):
    st.session_state.high_contrast = True
else:
    st.session_state.high_contrast = False

# Inject high contrast class
if st.session_state.high_contrast:
    render_html("""
    <style>
    html body .stApp { background-color: #000 !important; color: #fff !important; }
    html body .stSidebar { background-color: #000 !important; }
    html body .stMarkdown,
    html body .stCaption,
    html body label,
    html body p,
    html body span { color: #fff !important; }
    </style>
    """)

# Determine which radio index to show based on current mode
mode = st.session_state.mode
_reverse_map = {v: k for k, v in _MODE_NAME_MAP.items()}
_sidebar_to_show = LAB_LABEL if mode in LAB_SECTION_MODES else _reverse_map.get(mode, GAME_LABEL)
_default_idx = SELECTABLE_MODES.index(_sidebar_to_show) if _sidebar_to_show in SELECTABLE_MODES else 0

sidebar_selection = st.sidebar.radio(
    "Mode",
    SELECTABLE_MODES,
    index=_default_idx,
    key=MODE_SELECTOR_KEY,
    label_visibility="collapsed",
)
sidebar_effective = _MODE_NAME_MAP.get(sidebar_selection, sidebar_selection)

# Sidebar always overrides; clear deep-linked game room on nav change
# If a user navigates away from a room, discard the deep-linked room selection
# so returning to the showcase starts at the room-selection screen.
if sidebar_effective != mode and not (sidebar_effective == LAB_LABEL and mode in LAB_SECTION_MODES):
    st.session_state.game_room = None
    st.session_state.mode = sidebar_effective
    st.rerun()

mode = st.session_state.mode

if mode == GAME_LABEL:
    showcase_labels = [
        "Overview",
        "Room 1 — Frozen Maze",
        "Room 2 — Laser Corridor",
        "Room 3 — Key Vault",
        "Room 4 — Momentum Chamber",
        "Room 5 — Obstacle Lab",
    ]
    showcase_targets = {
        "Overview": None,
        "Room 1 — Frozen Maze": "room1",
        "Room 2 — Laser Corridor": "room2",
        "Room 3 — Key Vault": "room3",
        "Room 4 — Momentum Chamber": "room4",
        "Room 5 — Obstacle Lab": "room5",
    }
    valid_showcase_rooms = {room for room in showcase_targets.values() if room is not None}
    current_game_room = st.session_state.get("game_room")
    if current_game_room not in valid_showcase_rooms:
        st.session_state.game_room = None
        current_game_room = None
    current_showcase_label = next(
        (label for label, target in showcase_targets.items() if target == current_game_room),
        "Overview",
    )
    showcase_selection = st.sidebar.selectbox(
        "Showcase Room",
        showcase_labels,
        index=showcase_labels.index(current_showcase_label),
        key=SHOWCASE_ROOM_SELECTOR_KEY,
    )
    showcase_target = showcase_targets[showcase_selection]
    if showcase_target != current_game_room:
        go_to_showcase_room(showcase_target)
    mode = st.session_state.mode

if mode in LAB_SECTION_MODES:
    lab_labels = [
        "Overview",
        "Room 1 — Frozen Maze",
        "Room 2 — Laser Corridor",
        "Room 3 — Key Vault",
        "Room 4 — Momentum Chamber",
        "Room 5 — Obstacle Lab",
    ]
    lab_targets = {
        "Overview": LAB_LABEL,
        "Room 1 — Frozen Maze": ROOM1_LAB_MODE,
        "Room 2 — Laser Corridor": ROOM2_LAB_MODE,
        "Room 3 — Key Vault": ROOM3_LAB_MODE,
        "Room 4 — Momentum Chamber": ROOM4_LAB_MODE,
        "Room 5 — Obstacle Lab": ROOM5_BONUS_MODE,
    }
    current_lab_label = next(
        (label for label, target in lab_targets.items() if target == mode),
        "Overview",
    )
    lab_selection = st.sidebar.selectbox(
        "Learning Laboratory Room",
        lab_labels,
        index=lab_labels.index(current_lab_label),
        key="lab_room_selector",
    )
    lab_target = lab_targets[lab_selection]
    if lab_target != mode:
        st.session_state.mode = lab_target
        st.session_state.game_room = None
        st.rerun()
    mode = st.session_state.mode

# Ensure achievement tracker exists
AchievementTracker.from_session_state()

# ============================================================
# MODE: Escape Room Showcase
# ============================================================
if st.session_state.mode == GAME_LABEL:
    # Direct showcase routing: render a selected room, or the room-selection
    # screen when no room is active.
    render_html(render_global_styles())
    game_room = st.session_state.get("game_room")
    if game_room == "room1":
        render_room1_game()
    elif game_room == "room2":
        render_room2_game()
    elif game_room == "room3":
        render_room3_game()
    elif game_room == "room4":
        render_room4_game()
    elif game_room == "room5":
        render_room5_game()
    else:
        render_home_page()

# ============================================================
# MODE: About the Project
# ============================================================
elif st.session_state.mode == ABOUT_LABEL:
    render_html(render_global_styles())
    st.markdown("## About RL Escape Room")
    st.markdown("""
    This project applies reinforcement learning algorithms of increasing difficulty to
    navigate a series of escape-room environments. Rooms 1-4 cover the required assignment,
    and Room 5 is an optional bonus extension:

    | Room | Algorithm | Key Concept |
    |------|-----------|-------------|
    | 1 — Frozen Maze | Value Iteration | Dynamic Programming on known MDP |
    | 2 — Laser Corridor | SARSA | On-policy TD learning with risk sensitivity |
    | 3 — Key Vault | Q-Learning | Off-policy TD with augmented state space |
    | 4 — Momentum Chamber | Approximate SARSA | Linear function approximation with tile coding |
    | 5 - Dynamic Obstacles | NumPy DQN | Replay buffer + target network in continuous space |

    The **Escape Room Showcase** presents each room as a direct animated
    replay, while the **Learning Laboratory** provides full analysis tools including training
    curves, policy visualization, Q-value inspection, and algorithm comparison.
    """)

    st.markdown("### Deployment")
    render_public_project_links()

    # Screenshots
    st.markdown("### Screenshots")
    screenshots = [
        ("docs/screenshots/home.png", "Home / Room Selection"),
        ("docs/screenshots/room1_value_policy.png", "Room 1 - Value Iteration Convergence & Policy"),
        ("docs/screenshots/room2_training.png", "Room 2 - SARSA Training Progress"),
        ("docs/screenshots/room3_policy_no_key.png", "Room 3 - Q-Learning Policy (No Key)"),
        ("docs/screenshots/room4_trajectory.png", "Room 4 - Approximate SARSA Continuous Trajectory"),
        ("docs/screenshots/comparison.png", "Algorithm Comparison - SARSA vs Q-Learning"),
    ]
    for i, (path, caption) in enumerate(screenshots):
        if i % 2 == 0:
            cols = st.columns(2)
        image_path = APP_DIR / path
        if image_path.exists():
            cols[i % 2].image(str(image_path), caption=caption, width="stretch")
        else:
            cols[i % 2].warning(f"Screenshot not found: {path}")

    st.markdown("### Technical Stack")
    st.markdown("""
    - **Framework:** Streamlit
    - **Runtime:** Python 3.11+
    - **Numerics:** NumPy
    - **RL Algorithms:** Value Iteration, SARSA, Q-Learning, Semi-Gradient SARSA, NumPy DQN
    - **Function Approximation:** Tile Coding with linear basis functions
    - **Visualization:** SVG via `st.components.v1.html` and inline CSS
    """)

    st.markdown("### Repository")
    st.markdown("[GitHub](https://github.com/ZeidanK/rl_escape_room)")

# ============================================================
# MODE: Learning Laboratory overview
# ============================================================
elif st.session_state.mode == LAB_LABEL:
    st.header("Learning Laboratory")
    st.header("Project Objective")
    st.markdown("""
    Apply reinforcement learning algorithms of increasing difficulty to
    navigate a series of escape-room environments. Rooms 1-4 cover the required
    assignment path; Room 5 adds an optional DQN challenge with dynamic obstacles.
    """)

    st.header("Rooms")
    cols = st.columns(5)
    cols[0].markdown("**Room 1 — Frozen Maze**")
    cols[0].markdown("Value Iteration on known MDP with slippery cells.")
    cols[1].markdown("**Room 2 — Laser Corridor**")
    cols[1].markdown("SARSA learning risk-aware behaviour under slip and traps.")
    cols[2].markdown("**Room 3 — Key Vault**")
    cols[2].markdown("Q-Learning with key-collection and locked-exit states.")
    cols[3].markdown("**Room 4 — Momentum Chamber**")
    cols[3].markdown("Continuous state (x,y,vx,vy) with tile coding + linear approx SARSA.")
    cols[4].markdown("**Room 5 — Obstacle Lab**")
    cols[4].markdown("Continuous 10m room with 0.5m obstacles and NumPy DQN.")

    st.header("Room & Algorithm Summary")
    st.dataframe({
        "Room": ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"],
        "Algorithm": ["Value Iteration", "SARSA", "Q-Learning", "Approximate SARSA", "NumPy DQN"],
        "State Space": ["10x10 grid", "10x10 grid", "92 states (46 non-wall x key)", "Continuous (x,y,vx,vy)", "Continuous 22-feature vector"],
        "On/Off Policy": ["—", "On-policy", "Off-policy", "On-policy"],
        "State Space": [
            "10x10 grid",
            "10x10 grid",
            "92 states (46 non-wall x key)",
            "Continuous (x,y,vx,vy)",
            "Continuous 22-feature vector",
        ],
        "On/Off Policy": ["-", "On-policy", "Off-policy", "On-policy", "Off-policy"],
        "Model Known": ["Yes", "No", "No", "No", "No"],
    }, width="stretch")

    st.header("Instructions")
    st.markdown("""
    1. Use the **sidebar** to select a mode.
    2. For Rooms 1–3, select a room, configure parameters, and run the algorithm.
    3. For Room 4, configure tile-coding and training parameters.
    4. For Room 5, configure DQN replay, target-network, and obstacle settings.
    5. View training curves, policies, and trajectory replays.
    6. The **Algorithm Comparison** mode compares SARSA and Q-Learning; Room 5 remains a separate optional result.
    """)

    st.header("Symbols & Legend")
    st.markdown("""
    - `S` — Start cell
    - `G` — Goal / Exit
    - `W` — Wall (impassable)
    - `T` — Trap (penalty)
    - `K` — Key (Room 3)
    - `L` — Locked exit (Room 3)
    - `•` — Agent position
    - Arrows (↑→↓←) — Policy direction
    """)

    st.header("Final Local Measured Results")
    render_public_project_links()
    import csv, os
    csv_path = "storage/experiments/final/final_summary.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        st.dataframe(rows, width="stretch")
    else:
        st.info("Final summary not yet generated.")

    st.markdown("---")

# ============================================================
# MODE: Manual Environment
# ============================================================
if st.session_state.mode == "Manual Environment":
    from game.canvas_renderer import render_grid_canvas
    from game.theme import get_theme

    with st.sidebar:
        st.header("Controls")
        room_name = st.selectbox("Room", list(ROOM_CLASSES.keys()), key="room_selector",
                                 help="Select which room environment to play manually.")
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                               help="Random seed for environment stochasticity (slip outcomes).")
        if st.button("Reset") or st.session_state.room_key != room_name:
            cls = ROOM_CLASSES[room_name]
            st.session_state.env = cls(seed=seed)
            st.session_state.last_result = st.session_state.env.reset()
            st.session_state.room_key = room_name
            st.session_state.manual_total_reward = 0.0
            st.session_state.manual_agent_comparison = None
            st.session_state.manual_compare_key = None
            st.rerun()
        env = st.session_state.env
        if env is not None:
            st.markdown("---")
            st.markdown("**Actions**")
            disabled = env.is_done
            cols = st.columns(4)
            for i, (label, action) in enumerate(ACTION_BUTTONS.items()):
                if cols[i].button(label, disabled=disabled, key=f"m_btn_{action}",
                                  help=f"Move {label} in the grid."):
                    result: StepResult = env.step(action)
                    st.session_state.last_result = result
                    st.session_state.manual_total_reward = float(st.session_state.manual_total_reward or 0.0) + result.reward
                    st.session_state.manual_agent_comparison = None
                    st.session_state.manual_compare_key = None
                    st.rerun()
            st.markdown("---")
            st.markdown("**Status**")
            st.metric("Step", env.step_count)
            st.metric("Human Return", f"{float(st.session_state.manual_total_reward or 0.0):.1f}")
            if st.session_state.last_result is not None:
                r = st.session_state.last_result
                if isinstance(r, StepResult):
                    st.metric("Last Reward", f"{r.reward:.1f}")
                    if isinstance(r.info, dict):
                        st.markdown(f"**Requested:** {Action(r.info.get('requested_action', '?')).name}")
                        st.markdown(f"**Effective:** {Action(r.info.get('effective_action', '?')).name}")
                        st.markdown(f"**Slipped:** {r.info.get('slipped', False)}")
                        dash = "\u2014"
                        st.markdown(f"**Collision:** {r.info.get('collision', dash)}")
                        st.markdown(f"**Event:** {r.info.get('event', dash)}")
            if env.is_done:
                if env._terminated:
                    st.success("EXIT REACHED")
                elif env._truncated:
                    st.error("TIMEOUT")
                if st.button("Compare With Agent", key="manual_compare_agent"):
                    st.session_state.manual_agent_comparison = _manual_agent_comparison(
                        room_name,
                        int(seed),
                        int(env.max_steps),
                    )
                    st.session_state.manual_compare_key = (room_name, int(seed), int(env.max_steps))
                    st.rerun()

            comparison = st.session_state.manual_agent_comparison
            if comparison is not None:
                if comparison.get("available"):
                    st.markdown("---")
                    st.markdown("**Agent Comparison**")
                    st.metric("Agent", comparison["agent"])
                    st.metric("Agent Steps", comparison["steps"])
                    st.metric("Agent Return", f"{comparison['return']:.1f}")
                    st.metric("Agent Success", "Yes" if comparison["success"] else "No")
                else:
                    st.info(comparison.get("message", "agent comparison unavailable"))
    
    if env is not None:
        # Determine room_id for theme
        room_id_map = {
            "Room 1 — Frozen Maze (DP)": "room1",
            "Room 2 — Laser Corridor (SARSA)": "room2",
            "Room 3 — Key Vault (Q-Learning)": "room3",
        }
        room_id = room_id_map.get(room_name, "room1")
        theme = get_theme(room_id)
        
        # Render SVG grid with error handling
        try:
            svg = render_grid_canvas(
                env.grid,
                agent_pos=env.agent_position,
                room_id=room_id,
                cell_size=48,
                show_policy=False,
                show_values=False,
                show_labels=True,
            )
            render_html(f'<div style="overflow:hidden;">{svg}</div>')
        except Exception as e:
            st.error(f"Failed to render grid: {e}")
            st.code(env.render_ansi())
        
        # Legend
        render_html(f"""
        <div class="game-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_wall};"></span> Wall</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_slippery};"></span> Slippery</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Agent</span>
        </div>
        """)

# ============================================================
# MODE: Room 1 — DP
# ============================================================
elif st.session_state.mode == ROOM1_LAB_MODE:
    with st.sidebar:
        st.header("DP Parameters")
        gamma = st.slider("Discount (\u03b3)", 0.50, 0.99, 0.95, step=0.01,
                          help="How much future rewards are valued vs immediate rewards. Higher = more far-sighted.")
        tolerance = st.select_slider("Tolerance", options=[1e-2, 1e-4, 1e-6], value=1e-6,
                                     help="Stop iterating when max value change per iteration falls below this threshold.")
        max_it = st.number_input("Max Iterations", min_value=100, max_value=50000, value=10000, step=100,
                                 help="Hard cap on iterations. Value Iteration stops when converged or this limit is reached.")
        st.markdown("**Slip Probabilities**")
        p_int = st.slider("Intended", 0.0, 1.0, 0.80, step=0.05,
                          help="Probability the agent moves in the intended direction.")
        p_left = st.slider("Left", 0.0, 1.0, 0.10, step=0.05,
                           help="Probability the agent slips left (counter-clockwise) from intended direction.")
        p_right = st.slider("Right", 0.0, 1.0, 0.10, step=0.05,
                            help="Probability the agent slips right (clockwise) from intended direction.")
        slip_sum = p_int + p_left + p_right
        slip_valid = abs(slip_sum - 1.0) <= 1e-7
        if not slip_valid:
            st.error(f"Slip probabilities must sum to 1.0 (currently {slip_sum:.2f})")
        slip_cfg = SlipConfig(p_int, p_left, p_right) if slip_valid else SlipConfig()
        st.markdown("---")
        rollout_seed = st.number_input("Rollout Seed", min_value=0, max_value=2**31 - 1, value=0, step=1,
                                       help="Random seed for the policy rollout (trajectory simulation).")
        eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=1000, value=100, step=1,
                                  help="Number of episodes to run for policy evaluation.")
        solve_params = (gamma, tolerance, max_it, p_int, p_left, p_right)
        roll_params = solve_params + (rollout_seed,)
        ev_params = solve_params + (eval_ep,)
        st.markdown("**Saved Runs**")
        selected_vi_stem = _saved_run_selector(
            "room1",
            os.path.join("storage", "models", "room1_value_iteration"),
            requires_npz=False,
        )
        load_clicked = st.button("Load Saved Run", disabled=selected_vi_stem is None)
        col1, col2 = st.columns(2)
        solve_clicked = col1.button("Solve", type="primary", disabled=not slip_valid)
        if st.session_state.get("vi_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all DP results.")
            if col2.button("Confirm Reset", key="vi_confirm"):
                st.session_state.vi_result = None
                st.session_state.vi_rollout_result = None
                st.session_state.vi_eval_summary = None
                st.session_state.vi_model_stem = None
                st.session_state.vi_autoload_error = None
                st.session_state.vi_autoload_disabled = True
                st.session_state.vi_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="vi_cancel_reset"):
                st.session_state.vi_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results"):
            st.session_state.vi_confirm_reset = True
            st.rerun()
        rollout_clicked = st.button("Run Rollout", disabled=st.session_state.vi_result is None or not slip_valid)
        eval_clicked = st.button("Evaluate Policy", disabled=st.session_state.vi_result is None or not slip_valid)

    if load_clicked and selected_vi_stem:
        try:
            _load_room1_run_into_state(selected_vi_stem)
            st.success(f"Loaded saved run from {selected_vi_stem}")
            st.rerun()
        except ValueError as e:
            st.error(f"Load failed: {e}")

    if slip_valid:
        _autoload_room1_saved_run()

    auto_solve = (
        slip_valid
        and st.session_state.vi_solve_key != solve_params
        and st.session_state.vi_result is None
    )
    if solve_clicked or auto_solve:
        with st.spinner("Running Value Iteration..."):
            env = Room1DP(slip_config=slip_cfg, max_steps=200, seed=42)
            st.session_state.dp_env = env
            config = ValueIterationConfig(gamma=gamma, tolerance=tolerance, max_iterations=max_it)
            vi_r = ValueIterationAgent(env, config).solve()
            stem = _auto_run_stem(
                os.path.join("storage", "models", "room1_value_iteration"),
                "vi_auto",
            )
            save_room1_run(vi_r, stem, config=config, slip_config=slip_cfg, map_grid=env.grid)
            st.session_state.vi_result = vi_r
            st.session_state.vi_config = config
            st.session_state.vi_slip_config = slip_cfg
            st.session_state.vi_solve_key = solve_params
            st.session_state.vi_model_stem = stem
            st.session_state.vi_autoload_error = None
            st.session_state.vi_autoload_disabled = False
            st.session_state.vi_rollout_result = None
            st.session_state.vi_eval_summary = None
            st.rerun()

    vi_result = st.session_state.vi_result
    if vi_result is not None:
        env = st.session_state.dp_env
        if st.session_state.vi_rollout_result is None and st.session_state.vi_rollout_key != roll_params:
            if not solve_clicked:
                with st.spinner("Running rollout..."):
                    st.session_state.vi_rollout_result = rollout_policy(env, vi_result.policy, seed=rollout_seed)
                    st.session_state.vi_rollout_key = roll_params
                    _persist_room1_outputs_if_saved()
        if rollout_clicked:
            with st.spinner("Running rollout..."):
                st.session_state.vi_rollout_result = rollout_policy(env, vi_result.policy, seed=rollout_seed)
                st.session_state.vi_rollout_key = roll_params
                _persist_room1_outputs_if_saved()
                st.rerun()
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                st.session_state.vi_eval_summary = evaluate_policy(env, vi_result.policy, n_episodes=eval_ep)
                st.session_state.vi_eval_key = ev_params
                _persist_room1_outputs_if_saved()
                st.rerun()

        roll_r = st.session_state.vi_rollout_result
        ev_s = st.session_state.vi_eval_summary
        t1, t2, t3, t4, t5 = st.tabs(["Convergence", "Value Grid", "Policy Grid", "Rollout", "Evaluation"])
        with t1:
            st.metric("Converged", "Yes" if vi_result.converged else "No")
            c1, c2, c3 = st.columns(3)
            c1.metric("Iterations", vi_result.iterations)
            c2.metric("Final Delta", f"{vi_result.final_delta:.2e}")
            c3.metric("Start Value", f"{vi_result.start_state_value:.2f}")
            if len(vi_result.delta_history) > 1:
                st.line_chart({"delta": np.array(vi_result.delta_history)})
        with t2:
            st.dataframe(np.round(build_value_matrix(env, vi_result.values), 2), width="stretch")
        with t3:
            svg = render_policy_grid_canvas(env.grid, vi_result.policy, room_id="room1")
            render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')
        with t4:
            if roll_r:
                st.metric("Success", "Yes" if roll_r.success else "No")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Steps", roll_r.total_steps)
                c2.metric("Reward", f"{roll_r.total_reward:.1f}")
                c3.metric("Collisions", roll_r.collisions)
                c4.metric("Slipped", roll_r.slipped_actions)
        with t5:
            if ev_s:
                st.metric("Success Rate", f"{ev_s.success_rate:.1%}")
                c1, c2 = st.columns(2)
                c1.metric("Mean Return", f"{ev_s.mean_return:.2f}")
                c2.metric("Mean Steps", f"{ev_s.mean_steps:.1f}")

# ============================================================
# MODE: Room 2 — SARSA
# ============================================================
elif st.session_state.mode == ROOM2_LAB_MODE:
    with st.sidebar:
        st.header("SARSA Parameters")
        episodes = st.number_input("Episodes", min_value=100, max_value=50000, value=5000, step=500,
                                   help="Number of training episodes. More episodes = better convergence but slower.")
        alpha = st.slider("Alpha (\u03b1)", 0.01, 1.0, 0.10, step=0.01,
                          help="Learning rate. How much new experience overrides old knowledge.")
        gamma = st.slider("Gamma (\u03b3)", 0.50, 0.99, 0.95, step=0.01,
                          help="Discount factor for future rewards.")
        max_steps = st.number_input("Max Steps", min_value=50, max_value=2000, value=500, step=50,
                                    help="Maximum steps per episode. Episode truncates if exceeded.")

        st.markdown("**Epsilon Schedule**")
        eps_kind = st.selectbox("Decay Kind", ["exponential", "linear", "constant"], index=0,
                                help="How exploration rate decreases over time.")
        eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0, step=0.05,
                              help="Initial exploration rate (1.0 = fully random).")
        eps_min = st.slider("Epsilon Min", 0.0, 1.0, 0.05, step=0.01,
                            help="Minimum exploration rate. Never go below this.")
        eps_decay = st.slider("Decay Rate", 0.9, 1.0, 0.995, step=0.001,
                              help="Exponential decay factor per episode. Closer to 1 = slower decay.")
        linear_decay_ep = st.number_input("Linear Decay Episodes", min_value=1, max_value=50000, value=4000, step=100,
                                          help="Episodes over which to linearly decay epsilon (if linear kind selected).")

        st.markdown("**Slip Probabilities**")
        p_int = st.slider("Intended", 0.0, 1.0, 0.80, step=0.05,
                          help="Probability the agent moves in the intended direction.")
        p_left = st.slider("Left", 0.0, 1.0, 0.10, step=0.05,
                           help="Probability the agent slips left (counter-clockwise) from intended direction.")
        p_right = st.slider("Right", 0.0, 1.0, 0.10, step=0.05,
                            help="Probability the agent slips right (clockwise) from intended direction.")
        slip_sum = p_int + p_left + p_right
        slip_valid = abs(slip_sum - 1.0) <= 1e-7
        if not slip_valid:
            st.error(f"Slip probabilities must sum to 1.0 (currently {slip_sum:.2f})")
        slip_cfg = SlipConfig(p_int, p_left, p_right) if slip_valid else SlipConfig()

        train_seed = st.number_input("Training Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                     help="Random seed for training reproducibility.")
        eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=1000, value=100, step=1,
                                  help="Number of episodes for policy evaluation after training.")
        rw = st.number_input("Rolling Window", min_value=10, max_value=5000, value=100, step=10,
                             help="Window size for rolling averages in training charts.")

        # Build cache keys — includes rewards, slip, map signature
        import hashlib
        map_sig = hashlib.sha256(ROOM2_GRID.tobytes()).hexdigest()[:16]
        eps_cfg_key = (eps_kind, eps_start, eps_min, eps_decay, linear_decay_ep)
        train_key = (episodes, alpha, gamma, max_steps, eps_cfg_key, train_seed,
                     p_int, p_left, p_right, map_sig)
        eval_key = train_key + (eval_ep,)

        sarsa_config = SarsaConfig(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            max_steps=max_steps,
            seed=train_seed,
            epsilon=EpsilonScheduleConfig(
                kind=EpsilonDecayKind(eps_kind),
                start=eps_start,
                minimum=eps_min,
                decay=eps_decay,
                linear_decay_episodes=linear_decay_ep,
            ),
        )
        if slip_valid:
            _autoload_room2_sarsa_showcase(sarsa_config)

        st.markdown("**Saved Runs**")
        selected_sarsa_stem = _saved_run_selector(
            "room2",
            os.path.join("storage", "models", "room2_sarsa"),
            requires_npz=True,
        )
        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train SARSA", type="primary", disabled=not slip_valid)
        if st.session_state.get("sarsa_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all SARSA results.")
            if col2.button("Confirm Reset", key="sarsa_confirm"):
                st.session_state.sarsa_result = None
                st.session_state.sarsa_model_stem = None
                st.session_state.sarsa_autoload_error = None
                st.session_state.sarsa_autoload_disabled = True
                _clear_sarsa_outputs()
                st.session_state.sarsa_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="sarsa_cancel_reset"):
                st.session_state.sarsa_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results"):
            st.session_state.sarsa_confirm_reset = True
            st.rerun()
        eval_clicked = st.button("Evaluate Policy", disabled=st.session_state.sarsa_result is None or not slip_valid)
        save_clicked = st.button("Save Model", disabled=st.session_state.sarsa_result is None)
        load_clicked = st.button("Load Saved Run", disabled=selected_sarsa_stem is None)

    # Factory
    def make_env():
        return Room2SARSA(max_steps=max_steps, slip_config=slip_cfg)

    # --- Train ---
    if train_clicked:
        with st.spinner(f"Training SARSA for {episodes} episodes..."):
            agent = SarsaAgent(make_env, sarsa_config)

            progress_bar = st.progress(0)
            status_text = st.empty()

            def _cb(ep, total, metrics):
                progress_bar.progress((ep + 1) / total)
                if ep % max(1, total // 50) == 0:
                    status_text.text(
                        f"Episode {ep + 1}/{total} | "
                        f"Reward={metrics.total_reward:.1f} | "
                        f"Eps={metrics.epsilon:.3f} | "
                        f"Success={'Yes' if metrics.success else 'No'}"
                    )

            result = agent.train(progress_callback=_cb, progress_every=1)
            stem = _auto_run_stem(os.path.join("storage", "models", "room2_sarsa"), "sarsa_auto")
            save_model(result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM2_GRID)
            st.session_state.sarsa_result = result
            st.session_state.sarsa_train_key = train_key
            st.session_state.sarsa_model_stem = stem
            st.session_state.sarsa_autoload_error = None
            st.session_state.sarsa_autoload_disabled = False
            _clear_sarsa_outputs()
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    sarsa_result = st.session_state.sarsa_result

    # --- Load ---
    if load_clicked:
        if selected_sarsa_stem:
            try:
                _load_room2_sarsa_model_into_state(selected_sarsa_stem, sarsa_config)
                st.success(f"Loaded saved run from {selected_sarsa_stem}")
                st.rerun()
            except ValueError as e:
                st.error(f"Load failed: {e}")
        else:
            st.caption("No saved Room 2 models found.")
    sarsa_result = st.session_state.sarsa_result

    if sarsa_result is not None:
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                summary = evaluate_sarsa_policy(make_env, sarsa_result.q_values, n_episodes=eval_ep)
                st.session_state.sarsa_eval_summary = summary
                st.session_state.sarsa_eval_key = eval_key
                _persist_room2_outputs_if_saved()
                st.rerun()

        # --- Save ---
        if save_clicked:
            import os
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room2_sarsa", f"sarsa_{ts}")
            save_model(sarsa_result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM2_GRID)
            st.session_state.sarsa_model_stem = stem
            _persist_room2_outputs_if_saved()
            st.success(f"Model saved to {stem}")

        if st.session_state.sarsa_eval_summary is None:
            auto_eval_ep = min(int(eval_ep), 100)
            with st.spinner("Preparing Room 2 baseline evaluation..."):
                st.session_state.sarsa_eval_summary = evaluate_sarsa_policy(
                    make_env,
                    sarsa_result.q_values,
                    n_episodes=auto_eval_ep,
                )
                st.session_state.sarsa_eval_key = ("auto", train_key, auto_eval_ep)
                _persist_room2_outputs_if_saved()
        if st.session_state.sarsa_rollout is None:
            with st.spinner("Preparing Room 2 greedy replay..."):
                st.session_state.sarsa_rollout = rollout_sarsa_policy(
                    make_env,
                    sarsa_result.q_values,
                    seed=int(train_seed),
                    max_steps=int(max_steps),
                )
                st.session_state.sarsa_rollout_key = ("auto", train_key, int(train_seed), int(max_steps))
                _persist_room2_outputs_if_saved()

        # --- Greedy policy ---
        greedy_policy = extract_greedy_policy(sarsa_result.q_values)
        env_sample = make_env()

        # --- Tabs ---
        t1, t2, t3, t4, t5 = st.tabs([
            "Training Progress", "Learned Policy", "State Q-Values",
            "Training-Stage Replay", "Final Evaluation",
        ])

        # Tab 1: Training Progress
        with t1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(sarsa_result.metrics))
            c2.metric("Final Epsilon", f"{sarsa_result.final_epsilon:.4f}")
            if sarsa_result.metrics:
                df = build_training_dataframe(sarsa_result.metrics)
                window = min(rw, max(1, len(sarsa_result.metrics)))
                final_100 = sarsa_result.metrics[-window:]
                sr = sum(1 for m in final_100 if m.success) / len(final_100)
                c3.metric(f"Rolling Success ({window})", f"{sr:.1%}")

                st.subheader("Reward per Episode")
                rewards = np.array(df["total_reward"])
                rolling_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
                chart_data = {"reward": rewards, f"rolling ({window})": np.pad(rolling_r, (window - 1, 0))}
                st.line_chart(chart_data)

                st.subheader("Steps per Episode")
                st.line_chart({"steps": np.array(df["steps"])})

                st.subheader("Success Rate")
                successes = np.array(df["success"])
                rolling_s = np.convolve(successes, np.ones(window) / window, mode="valid")
                st.line_chart({"success": successes, f"rolling ({window})": np.pad(rolling_s, (window - 1, 0))})

                st.subheader("Epsilon")
                st.line_chart({"epsilon": np.array(df["epsilon"])})
            else:
                ev = st.session_state.sarsa_eval_summary
                rollout = st.session_state.sarsa_rollout
                if ev is not None:
                    c3.metric("Evaluation Success", f"{ev.success_rate:.1%}")
                if rollout is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                    c2.metric("Greedy Replay Steps", rollout.total_steps)
                    c3.metric("Greedy Replay Return", f"{rollout.total_reward:.1f}")

        # Tab 2: Learned Policy
        with t2:
            svg = render_policy_grid_canvas(env_sample.grid, greedy_policy, room_id="room2")
            render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')
            st.caption(
                "Legend: \u2191\u2192\u2193\u2190 = greedy action | "
                "S = Start | E = Exit | # = Wall | "
                "I = Slippery | T = Trap"
            )

        # Tab 3: State Q-Values
        with t3:
            q_tables = build_q_value_tables(sarsa_result.q_values)
            all_states = sorted(q_tables.keys())
            sel_state = st.selectbox("Select State", options=all_states,
                                     format_func=lambda s: f"({s[0]}, {s[1]})")
            if sel_state:
                vals = q_tables[sel_state]
                st.table({k: [f"{v:.4f}"] for k, v in vals.items()})

        # Tab 4: Training-Stage Replay
        with t4:
            snap_eps = sorted(sarsa_result.snapshots.keys())
            if snap_eps:
                sel_snap = st.selectbox("Snapshot Episode", options=snap_eps,
                                        format_func=lambda x: f"Episode {x}")
                snap = sarsa_result.snapshots[sel_snap]
                if snap.rollout:
                    st.metric("Rollout Success", "Yes" if snap.rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", snap.rollout.total_steps)
                    c2.metric("Reward", f"{snap.rollout.total_reward:.1f}")
                    c3.metric("Epsilon at Snapshot", f"{snap.epsilon:.4f}")
                    traj = tuple(s.state for s in snap.rollout.steps)
                    overlay = render_sarsa_trajectory_overlay(env_sample, traj)
                    st.dataframe(overlay, width="stretch")
            else:
                rollout = st.session_state.sarsa_rollout
                if rollout is not None:
                    st.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", rollout.total_steps)
                    c2.metric("Reward", f"{rollout.total_reward:.1f}")
                    c3.metric("Collisions", rollout.collisions)
                    traj = tuple(s.state for s in rollout.steps)
                    overlay = render_sarsa_trajectory_overlay(env_sample, traj)
                    st.dataframe(overlay, width="stretch")

        # Tab 5: Final Evaluation
        with t5:
            ev = st.session_state.sarsa_eval_summary
            if ev is not None:
                st.metric("Success Rate", f"{ev.success_rate:.1%}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Return", f"{ev.mean_return:.2f}")
                c2.metric("Mean Steps", f"{ev.mean_steps:.1f}")
                c3.metric("Std Return", f"{ev.std_return:.2f}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Truncated", ev.truncated_episodes)
                c2.metric("Collisions", ev.total_collisions)
                c3.metric("Traps", ev.total_traps)

# ============================================================
# MODE: Room 3 — Q-Learning
# ============================================================
elif st.session_state.mode == ROOM3_LAB_MODE:
    with st.sidebar:
        st.header("Q-Learning Parameters")
        episodes = st.number_input("Episodes", min_value=100, max_value=50000, value=5000, step=500,
                                   key="ql_episodes",
                                   help="Number of training episodes. More episodes = better convergence but slower.")
        alpha = st.slider("Alpha (\u03b1)", 0.01, 1.0, 0.10, step=0.01, key="ql_alpha",
                          help="Learning rate. How much new experience overrides old knowledge.")
        gamma_ql = st.slider("Gamma (\u03b3)", 0.50, 0.99, 0.95, step=0.01, key="ql_gamma",
                             help="Discount factor for future rewards.")
        max_steps = st.number_input("Max Steps", min_value=50, max_value=2000, value=500, step=50,
                                     key="ql_max_steps",
                                     help="Maximum steps per episode. Episode truncates if exceeded.")

        st.markdown("**Epsilon Schedule**")
        eps_kind = st.selectbox("Decay Kind", ["exponential", "linear", "constant"], index=0,
                                key="ql_eps_kind",
                                help="How exploration rate decreases over time.")
        eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0, step=0.05, key="ql_eps_start",
                              help="Initial exploration rate (1.0 = fully random).")
        eps_min = st.slider("Epsilon Min", 0.0, 1.0, 0.05, step=0.01, key="ql_eps_min",
                            help="Minimum exploration rate. Never go below this.")
        eps_decay = st.slider("Decay Rate", 0.9, 1.0, 0.995, step=0.001, key="ql_eps_decay",
                              help="Exponential decay factor per episode. Closer to 1 = slower decay.")
        linear_decay_ep = st.number_input("Linear Decay Episodes", min_value=1, max_value=50000, value=4000, step=100,
                                           key="ql_linear_decay",
                                           help="Episodes over which to linearly decay epsilon (if linear kind selected).")

        st.markdown("**Slip Probabilities**")
        p_int = st.slider("Intended", 0.0, 1.0, 0.80, step=0.05, key="ql_p_int",
                          help="Probability the agent moves in the intended direction.")
        p_left = st.slider("Left", 0.0, 1.0, 0.10, step=0.05, key="ql_p_left",
                           help="Probability the agent slips left (counter-clockwise) from intended direction.")
        p_right = st.slider("Right", 0.0, 1.0, 0.10, step=0.05, key="ql_p_right",
                            help="Probability the agent slips right (clockwise) from intended direction.")
        slip_sum = p_int + p_left + p_right
        slip_valid = abs(slip_sum - 1.0) <= 1e-7
        if not slip_valid:
            st.error(f"Slip probabilities must sum to 1.0 (currently {slip_sum:.2f})")
        slip_cfg = SlipConfig(p_int, p_left, p_right) if slip_valid else SlipConfig()

        train_seed = st.number_input("Training Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                      key="ql_seed")
        eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=1000, value=100, step=1,
                                   key="ql_eval_ep")
        rw = st.number_input("Rolling Window", min_value=10, max_value=5000, value=100, step=10,
                              key="ql_rw")

        import hashlib
        map_sig = hashlib.sha256(ROOM3_GRID.tobytes()).hexdigest()[:16]
        eps_cfg_key = (eps_kind, eps_start, eps_min, eps_decay, linear_decay_ep)
        train_key = (episodes, alpha, gamma_ql, max_steps, eps_cfg_key, train_seed,
                     p_int, p_left, p_right, map_sig)
        eval_key = train_key + (eval_ep,)

        ql_config = QLearningConfig(
            episodes=episodes,
            alpha=alpha,
            gamma=gamma_ql,
            max_steps=max_steps,
            seed=train_seed,
            epsilon=EpsilonScheduleConfig(
                kind=EpsilonDecayKind(eps_kind),
                start=eps_start,
                minimum=eps_min,
                decay=eps_decay,
                linear_decay_episodes=linear_decay_ep,
            ),
        )
        if slip_valid:
            _autoload_room3_q_showcase(ql_config)

        st.markdown("**Saved Runs**")
        selected_ql_stem = _saved_run_selector(
            "room3",
            os.path.join("storage", "models", "room3_q_learning"),
            requires_npz=True,
        )
        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train Q-Learning", type="primary", disabled=not slip_valid)
        if st.session_state.get("ql_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all Q-Learning results.")
            if col2.button("Confirm Reset", key="ql_confirm"):
                st.session_state.ql_result = None
                st.session_state.ql_model_stem = None
                st.session_state.ql_autoload_error = None
                st.session_state.ql_autoload_disabled = True
                _clear_q_learning_outputs()
                st.session_state.ql_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="ql_cancel_reset"):
                st.session_state.ql_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results", key="ql_reset"):
            st.session_state.ql_confirm_reset = True
            st.rerun()
        eval_clicked = st.button("Evaluate Policy", key="ql_eval_btn",
                                  disabled=st.session_state.ql_result is None or not slip_valid)
        save_clicked = st.button("Save Model", key="ql_save_btn",
                                  disabled=st.session_state.ql_result is None)
        load_clicked = st.button("Load Saved Run", key="ql_load_btn", disabled=selected_ql_stem is None)

    def make_ql_env():
        return Room3QLearning(max_steps=max_steps, slip_config=slip_cfg)

    if train_clicked:
        with st.spinner(f"Training Q-Learning for {episodes} episodes..."):
            agent = QLearningAgent(make_ql_env, ql_config)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _ql_cb(ep, total, metrics):
                progress_bar.progress((ep + 1) / total)
                if ep % max(1, total // 50) == 0:
                    status_text.text(
                        f"Episode {ep + 1}/{total} | "
                        f"Reward={metrics.total_reward:.1f} | "
                        f"Eps={metrics.epsilon:.3f} | "
                        f"Key={metrics.key_collected} | "
                        f"Success={'Yes' if metrics.success else 'No'}"
                    )

            result = agent.train(progress_callback=_ql_cb, progress_every=1)
            stem = _auto_run_stem(os.path.join("storage", "models", "room3_q_learning"), "ql_auto")
            save_q_model(result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM3_GRID)
            st.session_state.ql_result = result
            st.session_state.ql_train_key = train_key
            st.session_state.ql_model_stem = stem
            st.session_state.ql_autoload_error = None
            st.session_state.ql_autoload_disabled = False
            _clear_q_learning_outputs()
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    ql_result = st.session_state.ql_result

    if load_clicked:
        if selected_ql_stem:
            try:
                _load_room3_q_model_into_state(selected_ql_stem, ql_config)
                st.success(f"Loaded saved run from {selected_ql_stem}")
                st.rerun()
            except ValueError as e:
                st.error(f"Load failed: {e}")
        else:
            st.caption("No saved Room 3 models found.")
    ql_result = st.session_state.ql_result

    if ql_result is not None:
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                summary = evaluate_q_learning_policy(make_ql_env, ql_result.q_values, n_episodes=eval_ep)
                st.session_state.ql_eval_summary = summary
                st.session_state.ql_eval_key = eval_key
                _persist_room3_outputs_if_saved()
                st.rerun()

        if save_clicked:
            import os
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room3_q_learning", f"ql_{ts}")
            save_q_model(ql_result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM3_GRID)
            st.session_state.ql_model_stem = stem
            _persist_room3_outputs_if_saved()
            st.success(f"Model saved to {stem}")

        if st.session_state.ql_eval_summary is None:
            auto_eval_ep = min(int(eval_ep), 100)
            with st.spinner("Preparing Room 3 baseline evaluation..."):
                st.session_state.ql_eval_summary = evaluate_q_learning_policy(
                    make_ql_env,
                    ql_result.q_values,
                    n_episodes=auto_eval_ep,
                )
                st.session_state.ql_eval_key = ("auto", train_key, auto_eval_ep)
                _persist_room3_outputs_if_saved()
        if st.session_state.ql_rollout is None:
            with st.spinner("Preparing Room 3 greedy replay..."):
                st.session_state.ql_rollout = rollout_q_learning_policy(
                    make_ql_env,
                    ql_result.q_values,
                    seed=int(train_seed),
                    max_steps=int(max_steps),
                )
                st.session_state.ql_rollout_key = ("auto", train_key, int(train_seed), int(max_steps))
                _persist_room3_outputs_if_saved()

        from agents.tabular_utils import extract_deterministic_greedy_policy
        policy_no_key = extract_deterministic_greedy_policy(
            {s: v for s, v in ql_result.q_values.items() if not s[2]}
        )
        policy_with_key = extract_deterministic_greedy_policy(
            {s: v for s, v in ql_result.q_values.items() if s[2]}
        )
        env_sample = make_ql_env()

        t1, t2, t3, t4, t5, t6 = st.tabs([
            "Training Progress", "Policy (No Key)", "Policy (With Key)",
            "State Q-Values", "Training-Stage Replay", "Final Evaluation",
        ])

        with t1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(ql_result.metrics))
            c2.metric("Final Epsilon", f"{ql_result.final_epsilon:.4f}")
            if ql_result.metrics:
                df = build_q_learning_training_dataframe(ql_result.metrics)
                window = min(rw, max(1, len(ql_result.metrics)))
                final_100 = ql_result.metrics[-window:]
                sr = sum(1 for m in final_100 if m.success) / len(final_100)
                c3.metric(f"Rolling Success ({window})", f"{sr:.1%}")

                st.subheader("Reward per Episode")
                rewards = np.array(df["total_reward"])
                rolling_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
                chart_data = {"reward": rewards, f"rolling ({window})": np.pad(rolling_r, (window - 1, 0))}
                st.line_chart(chart_data)

                st.subheader("Steps per Episode")
                st.line_chart({"steps": np.array(df["steps"])})

                st.subheader("Key Events")
                key_col = np.array(df["key_collected"], dtype=float)
                locked = np.array(df["locked_exit_attempts"], dtype=float)
                st.line_chart({"key_collected": key_col, "locked_exit_attempts": locked})

                st.subheader("Success Rate")
                successes = np.array(df["success"])
                rolling_s = np.convolve(successes, np.ones(window) / window, mode="valid")
                st.line_chart({"success": successes, f"rolling ({window})": np.pad(rolling_s, (window - 1, 0))})

                st.subheader("Epsilon")
                st.line_chart({"epsilon": np.array(df["epsilon"])})
            else:
                ev = st.session_state.ql_eval_summary
                rollout = st.session_state.ql_rollout
                if ev is not None:
                    c3.metric("Evaluation Success", f"{ev.success_rate:.1%}")
                if rollout is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                    c2.metric("Greedy Replay Steps", rollout.total_steps)
                    c3.metric("Greedy Replay Return", f"{rollout.total_reward:.1f}")

        with t2:
            svg = render_policy_grid_canvas(env_sample.grid, policy_no_key, room_id="room3", has_key=False)
            render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')

        with t3:
            svg = render_policy_grid_canvas(env_sample.grid, policy_with_key, room_id="room3", has_key=True)
            render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')

        with t4:
            all_states = sorted(ql_result.q_values.keys())
            sel_state = st.selectbox(
                "Select State (row, col, has_key)", options=all_states,
                format_func=lambda s: f"({s[0]}, {s[1]}, key={'Y' if s[2] else 'N'})",
            )
            if sel_state:
                info = build_room3_q_value_table(env_sample, ql_result.q_values, sel_state[0], sel_state[1], sel_state[2])
                st.markdown(f"**State:** {info['state']}")
                st.markdown(f"**Terminal:** {'Yes' if info['is_terminal'] else 'No'}")
                st.markdown(f"**Greedy Action:** {info['greedy_action']}")
                st.table({a["action"]: [f"{a['value']:.4f}"] for a in info["actions"]})

        with t5:
            snap_eps = sorted(ql_result.snapshots.keys())
            if snap_eps:
                sel_snap = st.selectbox("Snapshot Episode", options=snap_eps,
                                        format_func=lambda x: f"Episode {x}")
                snap = ql_result.snapshots[sel_snap]
                if snap.rollout:
                    st.metric("Rollout Success", "Yes" if snap.rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", snap.rollout.total_steps)
                    c2.metric("Reward", f"{snap.rollout.total_reward:.1f}")
                    c3.metric("Epsilon", f"{snap.epsilon:.4f}")
                    traj = tuple(s.state for s in snap.rollout.steps)
                    overlay = render_q_learning_trajectory_overlay(env_sample, snap.rollout)
                    st.dataframe(overlay, width="stretch")
            else:
                rollout = st.session_state.ql_rollout
                if rollout is not None:
                    st.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", rollout.total_steps)
                    c2.metric("Reward", f"{rollout.total_reward:.1f}")
                    c3.metric("Collisions", rollout.collisions)
                    overlay = render_q_learning_trajectory_overlay(env_sample, rollout)
                    st.dataframe(overlay, width="stretch")

        with t6:
            ev = st.session_state.ql_eval_summary
            if ev is not None:
                st.metric("Success Rate", f"{ev.success_rate:.1%}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Return", f"{ev.mean_return:.2f}")
                c2.metric("Mean Steps", f"{ev.mean_steps:.1f}")
                c3.metric("Std Return", f"{ev.std_return:.2f}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Key Collection Rate", f"{ev.key_collection_rate:.1%}")
                c2.metric("Mean Key Step", f"{ev.mean_key_collection_step}" if ev.mean_key_collection_step else "N/A")
                c3.metric("Locked Exit Attempts", ev.total_locked_exit_attempts)
                c1, c2, c3 = st.columns(3)
                c1.metric("Truncated", ev.truncated_episodes)
                c2.metric("Collisions", ev.total_collisions)
                c3.metric("Traps", ev.total_traps)

# ============================================================
# MODE: Room 4 — Function Approximation
# ============================================================
elif st.session_state.mode == ROOM4_LAB_MODE:
    with st.sidebar:
        st.header("Approximate SARSA Parameters")
        approx_episodes = st.number_input("Episodes", min_value=50, max_value=50000, value=3000, step=500,
                                          key="approx_episodes",
                                          help="Number of training episodes. More = better convergence but slower.")
        approx_alpha = st.slider("Alpha (\u03b1)", 0.01, 1.0, 0.10, step=0.01, key="approx_alpha",
                                 help="Learning rate. Higher = faster learning but may overshoot. Lower = more stable.")
        approx_gamma = st.slider("Gamma (\u03b3)", 0.50, 0.99, 0.99, step=0.01, key="approx_gamma",
                                 help="Discount factor. Higher = more far-sighted. 0.99 is typical for continuing tasks.")
        approx_max_steps = st.number_input("Max Steps", min_value=50, max_value=2000, value=750, step=50,
                                            key="approx_max_steps",
                                            help="Maximum steps per episode. Episode truncates if exit not reached.")

        st.markdown("**Epsilon Schedule**")
        approx_eps_kind = st.selectbox("Decay Kind", ["exponential", "linear", "constant"], index=0,
                                        key="approx_eps_kind",
                                        help="How exploration rate decreases. Exponential = smooth decay. Linear = steady decrease. Constant = no decay.")
        approx_eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0, step=0.05, key="approx_eps_start",
                                     help="Initial exploration rate. 1.0 = fully random, 0.0 = fully greedy.")
        approx_eps_min = st.slider("Epsilon Min", 0.0, 1.0, 0.02, step=0.01, key="approx_eps_min",
                                   help="Minimum exploration rate. Never decays below this.")
        approx_eps_decay = st.slider("Decay Rate", 0.9, 1.0, 0.997, step=0.001, key="approx_eps_decay",
                                     help="Exponential decay factor per episode. Closer to 1.0 = slower decay.")
        approx_linear_decay = st.number_input("Linear Decay Episodes", min_value=1, max_value=50000, value=2000,
                                               step=100, key="approx_linear_decay",
                                               help="Episodes over which epsilon linearly decays from start to min (only for linear decay).")

        st.markdown("**Tile Coding**")
        approx_tilings = st.number_input("Num Tilings", min_value=1, max_value=64, value=8, step=1,
                                          key="approx_tilings",
                                          help="Number of overlapping tile grids. More tilings = better approximation but more computation. 4-16 typical.")
        approx_tiles_xy = st.selectbox("Tiles per Dim", [4, 8, 10, 16, 20], index=2, key="approx_tiles_xy",
                                       help="Tiles per dimension (x and y). More tiles = finer discretization. 8-16 typical.")

        st.markdown("**Reward**")
        approx_progress_scale = st.slider("Progress Scale", 0.0, 2.0, 1.0, step=0.1, key="approx_progress_scale",
                                          help="Scales the reward for moving toward exit. 0 = no shaping, 1 = standard shaping, >1 = strong guidance.")

        st.markdown("**Start Mode**")
        approx_start_mode = st.selectbox("Training Start", ["fixed", "random_lower_left", "random_room"],
                                         index=1, key="approx_start_mode",
                                         help="Where episodes start. fixed = always same start. random_lower_left = varied starts near origin. random_room = any valid start position.")

        train_seed = st.number_input("Training Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                     key="approx_seed",
                                     help="Random seed for training reproducibility.")
        eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=500, value=50, step=1,
                                  key="approx_eval_ep",
                                  help="Number of episodes for policy evaluation after training.")
        rw = st.number_input("Rolling Window", min_value=10, max_value=5000, value=100, step=10,
                             key="approx_rw",
                             help="Window size for rolling averages in training charts.")

        # Build cache keys
        tc_cfg = TileCodingConfig(num_tilings=approx_tilings, tiles_x=approx_tiles_xy,
                                  tiles_y=approx_tiles_xy, include_velocity=True)
        eps_cfg_key = (approx_eps_kind, approx_eps_start, approx_eps_min, approx_eps_decay, approx_linear_decay)
        train_key = (approx_episodes, approx_alpha, approx_gamma, approx_max_steps,
                     eps_cfg_key, train_seed, approx_tilings, approx_tiles_xy,
                     approx_progress_scale, approx_start_mode)
        eval_key_fixed = train_key + (eval_ep, "fixed")
        eval_key_gen = train_key + (eval_ep, "gen")

        approx_config = ApproximateSarsaConfig(
            episodes=approx_episodes,
            alpha=approx_alpha,
            gamma=approx_gamma,
            max_steps=approx_max_steps,
            seed=train_seed,
            epsilon=EpsilonScheduleConfig(
                kind=EpsilonDecayKind(approx_eps_kind),
                start=approx_eps_start,
                minimum=approx_eps_min,
                decay=approx_eps_decay,
                linear_decay_episodes=approx_linear_decay,
            ),
            tile_coding=tc_cfg,
            start_mode=StartMode(approx_start_mode),
        )
        _autoload_room4_showcase(approx_config, tc_cfg)

        st.markdown("**Saved Runs**")
        selected_approx_stem = _saved_run_selector(
            "room4",
            os.path.join("storage", "models", "room4_approximate_sarsa"),
            requires_npz=True,
        )
        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train Approx SARSA", type="primary")
        if st.session_state.get("approx_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all Approximate SARSA results.")
            if col2.button("Confirm Reset", key="approx_confirm"):
                st.session_state.approx_result = None
                st.session_state.approx_model_stem = None
                st.session_state.approx_autoload_error = None
                st.session_state.approx_autoload_disabled = True
                _clear_room4_outputs()
                st.session_state.approx_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="approx_cancel_reset"):
                st.session_state.approx_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results", key="approx_reset"):
            st.session_state.approx_confirm_reset = True
            st.rerun()

        eval_fixed_clicked = st.button("Evaluate Fixed Start", key="approx_eval_fixed_btn",
                                       disabled=st.session_state.approx_result is None)
        eval_gen_clicked = st.button("Evaluate Generalization", key="approx_eval_gen_btn",
                                     disabled=st.session_state.approx_result is None)
        save_clicked = st.button("Save Model", key="approx_save_btn",
                                 disabled=st.session_state.approx_result is None)
        load_clicked = st.button("Load Saved Run", key="approx_load_btn", disabled=selected_approx_stem is None)

    def make_approx_env(start_mode=None, max_steps=None):
        sm = start_mode if start_mode is not None else StartMode(approx_start_mode)
        return Room4Continuous(
            motion_config=Room4MotionConfig(),
            reward_config=ContinuousRewardConfig(distance_progress_scale=approx_progress_scale),
            max_steps=int(approx_max_steps if max_steps is None else max_steps),
            start_mode=sm,
        )

    # --- Train ---
    if train_clicked:
        with st.spinner(f"Training Approximate SARSA for {approx_episodes} episodes..."):
            factory = lambda: make_approx_env()
            agent = ApproximateSarsaAgent(factory, approx_config)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _approx_cb(ep, total, metrics):
                progress_bar.progress((ep + 1) / total)
                if ep % max(1, total // 50) == 0:
                    status_text.text(
                        f"Episode {ep + 1}/{total} | "
                        f"Reward={metrics.total_reward:.1f} | "
                        f"Eps={metrics.epsilon:.3f} | "
                        f"Dist={metrics.final_distance_to_exit_m:.2f} | "
                        f"Success={'Yes' if metrics.success else 'No'}"
                    )

            result = agent.train(progress_callback=_approx_cb, progress_every=1)
            stem = _auto_run_stem(
                os.path.join("storage", "models", "room4_approximate_sarsa"),
                "approx_auto",
            )
            save_approximate_model(
                result,
                stem,
                tile_coding_config=tc_cfg,
                motion_config=Room4MotionConfig(),
                reward_config=ContinuousRewardConfig(distance_progress_scale=approx_progress_scale),
            )
            st.session_state.approx_result = result
            st.session_state.approx_train_key = train_key
            st.session_state.approx_model_stem = stem
            st.session_state.approx_autoload_error = None
            st.session_state.approx_autoload_disabled = False
            _clear_room4_outputs()
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    approx_result = st.session_state.approx_result

    if load_clicked:
        if selected_approx_stem:
            try:
                _load_room4_model_into_state(selected_approx_stem, approx_config, tc_cfg)
                st.success(f"Loaded saved run from {selected_approx_stem}")
                st.rerun()
            except ValueError as e:
                st.error(f"Load failed: {e}")
        else:
            st.caption("No saved Room 4 models found.")
    approx_result = st.session_state.approx_result

    if approx_result is not None:
        effective_tc_cfg = approx_result.config.tile_coding

        # --- Eval fixed ---
        if eval_fixed_clicked:
            with st.spinner(f"Evaluating fixed start ({eval_ep} episodes)..."):
                factory = lambda: make_approx_env(start_mode=StartMode.FIXED)
                ev = evaluate_approximate_policy(
                    factory, approx_result.weights, effective_tc_cfg, Room4MotionConfig(),
                    n_episodes=eval_ep, start_mode=StartMode.FIXED,
                )
                st.session_state.approx_eval_fixed = ev
                st.session_state.approx_eval_fixed_key = eval_key_fixed
                _persist_room4_outputs_if_saved()
                st.rerun()

        # --- Eval generalization ---
        if eval_gen_clicked:
            with st.spinner(f"Evaluating generalization ({eval_ep} episodes)..."):
                factory = lambda: make_approx_env(start_mode=StartMode.RANDOM_LOWER_LEFT)
                ev_gen = evaluate_approximate_policy(
                    factory, approx_result.weights, effective_tc_cfg, Room4MotionConfig(),
                    n_episodes=eval_ep, start_mode=StartMode.RANDOM_LOWER_LEFT,
                )
                st.session_state.approx_eval_gen = ev_gen
                st.session_state.approx_eval_gen_key = eval_key_gen
                _persist_room4_outputs_if_saved()
                st.rerun()

        # --- Save ---
        if save_clicked:
            import os
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room4_approximate_sarsa", f"approx_{ts}")
            save_approximate_model(approx_result, stem, tile_coding_config=tc_cfg,
                                   motion_config=Room4MotionConfig(),
                                   reward_config=ContinuousRewardConfig(distance_progress_scale=approx_progress_scale))
            st.session_state.approx_model_stem = stem
            _persist_room4_outputs_if_saved()
            st.success(f"Model saved to {stem}")

        if (
            st.session_state.approx_eval_fixed is None
            or st.session_state.approx_eval_gen is None
            or st.session_state.approx_rollout is None
        ):
            auto_eval_ep = min(int(eval_ep), 20)
            auto_max_steps = int(approx_result.config.max_steps)
            auto_motion_cfg = Room4MotionConfig()
            with st.spinner("Preparing Room 4 baseline evaluations and trajectory..."):
                if st.session_state.approx_eval_fixed is None:
                    st.session_state.approx_eval_fixed = evaluate_approximate_policy(
                        lambda: make_approx_env(start_mode=StartMode.FIXED, max_steps=auto_max_steps),
                        approx_result.weights,
                        effective_tc_cfg,
                        auto_motion_cfg,
                        n_episodes=auto_eval_ep,
                        start_mode=StartMode.FIXED,
                        max_steps=auto_max_steps,
                    )
                    st.session_state.approx_eval_fixed_key = ("auto", train_key, auto_eval_ep, "fixed")
                    _persist_room4_outputs_if_saved()
                if st.session_state.approx_eval_gen is None:
                    st.session_state.approx_eval_gen = evaluate_approximate_policy(
                        lambda: make_approx_env(start_mode=StartMode.RANDOM_LOWER_LEFT, max_steps=auto_max_steps),
                        approx_result.weights,
                        effective_tc_cfg,
                        auto_motion_cfg,
                        n_episodes=auto_eval_ep,
                        start_mode=StartMode.RANDOM_LOWER_LEFT,
                        max_steps=auto_max_steps,
                    )
                    st.session_state.approx_eval_gen_key = ("auto", train_key, auto_eval_ep, "gen")
                    _persist_room4_outputs_if_saved()
                if st.session_state.approx_rollout is None:
                    q_func = _approx_q_function_from_weights(approx_result.weights, effective_tc_cfg, auto_motion_cfg)
                    st.session_state.approx_rollout = rollout_approximate_policy(
                        lambda: make_approx_env(start_mode=StartMode.FIXED, max_steps=auto_max_steps),
                        q_func,
                        seed=int(train_seed),
                        max_steps=auto_max_steps,
                    )
                    st.session_state.approx_rollout_key = ("auto", train_key, int(train_seed), auto_max_steps)
                    _persist_room4_outputs_if_saved()

        # --- Tabs ---
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "Training Progress", "Final Trajectory", "Training-Stage Replay",
            "Greedy Action Field", "Value Surface", "Evaluation", "Experiments",
        ])

        # Tab 1: Training Progress
        with t1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(approx_result.metrics))
            c2.metric("Final Epsilon", f"{approx_result.final_epsilon:.4f}")
            if approx_result.metrics:
                df = build_approx_training_dataframe(approx_result.metrics)
                window = min(rw, max(1, len(approx_result.metrics)))
                final_win = approx_result.metrics[-window:]
                sr = sum(1 for m in final_win if m.success) / len(final_win)
                c3.metric(f"Rolling Success ({window})", f"{sr:.1%}")

                st.subheader("Reward per Episode")
                rewards = np.array(df["total_reward"])
                rolling_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
                chart_data = {"reward": rewards, f"rolling ({window})": np.pad(rolling_r, (window - 1, 0))}
                st.line_chart(chart_data)

                st.subheader("Steps per Episode")
                st.line_chart({"steps": np.array(df["steps"])})

                st.subheader("Distance to Exit")
                st.line_chart({"final_distance_to_exit_m": np.array(df["final_distance_to_exit_m"])})

                st.subheader("Success Rate")
                successes = np.array(df["success"])
                rolling_s = np.convolve(successes, np.ones(window) / window, mode="valid")
                st.line_chart({"success": successes, f"rolling ({window})": np.pad(rolling_s, (window - 1, 0))})

                st.subheader("Epsilon")
                st.line_chart({"epsilon": np.array(df["epsilon"])})
            else:
                ev_fixed = st.session_state.approx_eval_fixed
                ev_gen = st.session_state.approx_eval_gen
                rollout = st.session_state.approx_rollout
                if ev_fixed is not None:
                    c3.metric("Fixed Eval Success", f"{ev_fixed.success_rate:.1%}")
                if ev_gen is not None and rollout is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Generalization Success", f"{ev_gen.success_rate:.1%}")
                    c2.metric("Greedy Replay Steps", rollout.steps)
                    c3.metric("Greedy Replay Return", f"{rollout.total_reward:.1f}")

        # Tab 2: Final Trajectory
        with t2:
            last_rollout = st.session_state.approx_rollout
            snap_keys = sorted(approx_result.snapshots.keys())
            if last_rollout is None and snap_keys:
                last_snap = approx_result.snapshots[snap_keys[-1]]
                if last_snap.rollout:
                    last_rollout = last_snap.rollout
            if last_rollout:
                env_disp = make_approx_env(start_mode=StartMode.FIXED, max_steps=approx_result.config.max_steps)
                env_disp.reset(seed=last_rollout.seed)
                svg = render_continuous_trajectory_canvas(env_disp, last_rollout, max_arrows=20)
                render_html(svg)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Success", "Yes" if last_rollout.success else "No")
                c2.metric("Steps", last_rollout.steps)
                c3.metric("Reward", f"{last_rollout.total_reward:.1f}")
                c4.metric("Distance", f"{last_rollout.distance_travelled_m:.1f}m")

        # Tab 3: Training-Stage Replay
        with t3:
            snap_keys = sorted(approx_result.snapshots.keys())
            if snap_keys:
                sel_snap = st.selectbox("Snapshot Episode", options=snap_keys,
                                        format_func=lambda x: f"Episode {x}")
                snap = approx_result.snapshots[sel_snap]
                if snap.rollout:
                    st.metric("Rollout Success", "Yes" if snap.rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", snap.rollout.steps)
                    c2.metric("Reward", f"{snap.rollout.total_reward:.1f}")
                    c3.metric("Epsilon", f"{snap.epsilon:.4f}")
                    env_disp = make_approx_env(start_mode=StartMode.FIXED)
                    env_disp.reset(seed=99)
                    svg = render_continuous_trajectory_canvas(env_disp, snap.rollout, max_arrows=20)
                    render_html(svg)
            else:
                rollout = st.session_state.approx_rollout
                if rollout is not None:
                    st.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Steps", rollout.steps)
                    c2.metric("Reward", f"{rollout.total_reward:.1f}")
                    c3.metric("Collisions", rollout.collision_count)
                    env_disp = make_approx_env(start_mode=StartMode.FIXED, max_steps=approx_result.config.max_steps)
                    env_disp.reset(seed=rollout.seed)
                    svg = render_continuous_trajectory_canvas(env_disp, rollout, max_arrows=20)
                    render_html(svg)

        # Tab 4: Greedy Action Field
        with t4:
            vx_choice = st.selectbox("Vx", [-1, 0, 1], index=1, key="af_vx")
            vy_choice = st.selectbox("Vy", [-1, 0, 1], index=1, key="af_vy")
            af_size = st.slider("Grid Resolution", 5, 30, 10, key="af_size")
            env_disp = make_approx_env(start_mode=StartMode.FIXED)
            field = build_approx_action_field(env_disp, approx_result.weights, effective_tc_cfg,
                                              fixed_vx=vx_choice, fixed_vy=vy_choice, grid_size=af_size)
            svg = render_action_field_canvas(env_disp, field, fixed_velocity=(vx_choice, vy_choice))
            render_html(svg)

        # Tab 5: Value Surface
        with t5:
            vs_vx = st.selectbox("Vx", [-1, 0, 1], index=1, key="vs_vx")
            vs_vy = st.selectbox("Vy", [-1, 0, 1], index=1, key="vs_vy")
            vs_size = st.slider("Grid Resolution", 5, 40, 20, key="vs_size")
            env_disp = make_approx_env(start_mode=StartMode.FIXED)
            surface = build_approx_value_surface(env_disp, approx_result.weights, effective_tc_cfg,
                                                 fixed_vx=vs_vx, fixed_vy=vs_vy, grid_size=vs_size)
            st.dataframe(np.round(surface, 2), width="stretch")

        # Tab 6: Evaluation
        with t6:
            ev_fixed = st.session_state.approx_eval_fixed
            ev_gen = st.session_state.approx_eval_gen

            if ev_fixed is not None:
                st.subheader("Fixed Start")
                c1, c2, c3 = st.columns(3)
                c1.metric("Success Rate", f"{ev_fixed.success_rate:.1%}")
                c2.metric("Mean Return", f"{ev_fixed.mean_return:.2f}")
                c3.metric("Std Return", f"{ev_fixed.std_return:.2f}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Steps", f"{ev_fixed.mean_steps:.1f}")
                c2.metric("Truncated", ev_fixed.truncated_count)
                c3.metric("Collisions", ev_fixed.total_collisions)

            if ev_gen is not None:
                st.subheader("Generalization (Random Lower-Left)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Success Rate", f"{ev_gen.success_rate:.1%}")
                c2.metric("Mean Return", f"{ev_gen.mean_return:.2f}")
                c3.metric("Std Return", f"{ev_gen.std_return:.2f}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Steps", f"{ev_gen.mean_steps:.1f}")
                c2.metric("Truncated", ev_gen.truncated_count)
                c3.metric("Collisions", ev_gen.total_collisions)

            if ev_fixed is None and ev_gen is None:
                pass

        # Tab 7: Experiments
        with t7:
            st.subheader("Hyperparameter Experiments")
            st.markdown("**Stage A — One Factor at a Time**")
            if st.button("Run Stage A Screening", key="approx_stage_a"):
                with st.spinner("Running Stage A screening..."):
                    stage_a = run_screening_stage_a(n_episodes=200, eval_episodes=20, seed=train_seed)
                    st.dataframe(stage_a, width="stretch")
                    import json, os
                    from datetime import datetime
                    path = os.path.join("storage", "experiments", "room4_approximate_sarsa",
                                        f"stage_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(stage_a, f, indent=2, default=str)
                    st.success(f"Saved to {path}")

            st.markdown("**Confirmation**")
            st.markdown("Top configs (num_tilings=8, tiles_xy=10, alpha=0.10, progress=1.0, decay=0.997)")
            if st.button("Run Confirmation", key="approx_confirmation"):
                with st.spinner("Running confirmation..."):
                    configs = [{"num_tilings": 8, "tiles_xy": 10, "alpha": 0.10,
                                "progress_scale": 1.0, "epsilon_decay": 0.997}]
                    conf = run_approx_confirmation(configs, n_episodes=500, eval_episodes=30, seeds=(42, 43, 44))
                    st.dataframe(conf, width="stretch")

# ============================================================
# MODE: Room 5 — Dynamic Obstacles
# ============================================================
elif st.session_state.mode == ROOM5_BONUS_MODE:
    st.header("Room 5 — Obstacle Lab")
    st.caption(
        "Continuous 10x10m escape room with seeded 0.5m square obstacles, "
        "local observation records, replay buffer DQN updates, and separate fixed/random/unseen layout evaluation."
    )
    render_assignment_proof("room5")

    with st.sidebar:
        st.header("DQN Parameters")
        dqn_episodes = st.number_input("Episodes", min_value=10, max_value=20000, value=600, step=50,
                                       key="dqn_episodes")
        dqn_lr = st.slider("Learning Rate", 0.0001, 0.05, 0.001, step=0.0001,
                           format="%.4f", key="dqn_lr")
        dqn_gamma = st.slider("Gamma", 0.50, 0.99, 0.99, step=0.01, key="dqn_gamma")
        dqn_max_steps = st.number_input("Max Steps", min_value=50, max_value=1500, value=260, step=10,
                                        key="dqn_max_steps")

        st.markdown("**Epsilon Schedule**")
        dqn_eps_kind = st.selectbox("Decay Kind", ["exponential", "linear", "constant"], index=0,
                                    key="dqn_eps_kind")
        dqn_eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0, step=0.05, key="dqn_eps_start")
        dqn_eps_min = st.slider("Epsilon Min", 0.0, 1.0, 0.05, step=0.01, key="dqn_eps_min")
        dqn_eps_decay = st.slider("Decay Rate", 0.90, 1.0, 0.995, step=0.001, key="dqn_eps_decay")
        dqn_linear_decay = st.number_input("Linear Decay Episodes", min_value=1, max_value=20000, value=500,
                                           step=50, key="dqn_linear_decay")

        st.markdown("**Replay and Network**")
        dqn_replay_capacity = st.number_input("Replay Capacity", min_value=100, max_value=200000, value=20000,
                                              step=1000, key="dqn_replay_capacity")
        dqn_batch_size = st.number_input("Batch Size", min_value=4, max_value=512, value=64, step=4,
                                         key="dqn_batch_size")
        dqn_warmup = st.number_input("Warmup Steps", min_value=0, max_value=10000, value=128, step=16,
                                     key="dqn_warmup")
        dqn_target_update = st.number_input("Target Update Interval", min_value=1, max_value=5000, value=100,
                                            step=10, key="dqn_target_update")
        dqn_hidden_units = st.number_input("Hidden Units", min_value=8, max_value=512, value=64, step=8,
                                           key="dqn_hidden_units")

        st.markdown("**Obstacle Layout**")
        dqn_min_obs = st.number_input("Min Obstacles", min_value=0, max_value=12, value=3, step=1,
                                      key="dqn_min_obs")
        dqn_max_obs = st.number_input("Max Obstacles", min_value=int(dqn_min_obs), max_value=12, value=max(5, int(dqn_min_obs)),
                                      step=1, key="dqn_max_obs")
        dqn_obs_dist = st.slider("Observation Distance X (m)", 0.5, 8.0, 2.5, step=0.25,
                                 key="dqn_obs_dist")
        dqn_layout_seed = st.number_input("Layout Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                          key="dqn_layout_seed")
        dqn_fixed_layout = st.checkbox("Train on Fixed Layout", value=False, key="dqn_fixed_layout")
        dqn_progress_scale = st.slider("Progress Reward Scale", 0.0, 5.0, 2.0, step=0.25,
                                       key="dqn_progress_scale")

        dqn_train_seed = st.number_input("Training Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                         key="dqn_train_seed")
        dqn_eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=500, value=25, step=1,
                                      key="dqn_eval_ep")
        dqn_rollout_seed = st.number_input("Replay Seed", min_value=0, max_value=2**31 - 1, value=7, step=1,
                                           key="dqn_rollout_seed")
        dqn_rollout_layout_seed = st.number_input("Replay Layout Seed", min_value=0, max_value=2**31 - 1,
                                                  value=1007, step=1, key="dqn_rollout_layout_seed")

        dqn_epsilon_cfg = EpsilonScheduleConfig(
            kind=EpsilonDecayKind(dqn_eps_kind),
            start=float(dqn_eps_start),
            minimum=float(dqn_eps_min),
            decay=float(dqn_eps_decay),
            linear_decay_episodes=int(dqn_linear_decay),
        )
        dqn_config = DQNConfig(
            episodes=int(dqn_episodes),
            learning_rate=float(dqn_lr),
            gamma=float(dqn_gamma),
            max_steps=int(dqn_max_steps),
            seed=int(dqn_train_seed),
            epsilon=dqn_epsilon_cfg,
            replay_capacity=int(dqn_replay_capacity),
            batch_size=int(dqn_batch_size),
            warmup_steps=int(dqn_warmup),
            target_update_interval=int(dqn_target_update),
            hidden_units=int(dqn_hidden_units),
        )
        _autoload_room5_showcase(dqn_config)

        st.markdown("**Saved Runs**")
        selected_dqn_stem = _saved_run_selector(
            "room5",
            os.path.join("storage", "models", "room5_dqn"),
            requires_npz=True,
        )
        col1, col2 = st.columns(2)
        dqn_train_clicked = col1.button("Train DQN", type="primary", key="dqn_train_btn")
        if st.session_state.get("dqn_confirm_reset"):
            st.warning("Click again to confirm reset - this will clear Bonus Room results.")
            if col2.button("Confirm Reset", key="dqn_confirm"):
                for key in [
                    "dqn_result", "dqn_network", "dqn_meta", "dqn_eval_fixed",
                    "dqn_eval_random", "dqn_eval_unseen", "dqn_rollout",
                    "dqn_rollout_key", "dqn_rollout_fixed_layout", "dqn_model_stem",
                    "dqn_autoload_error", "dqn_result_source",
                ]:
                    st.session_state[key] = None
                st.session_state.dqn_autoload_disabled = True
                st.session_state.dqn_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="dqn_cancel_reset"):
                st.session_state.dqn_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results", key="dqn_reset"):
            st.session_state.dqn_confirm_reset = True
            st.rerun()

        dqn_eval_fixed_clicked = st.button("Evaluate Fixed Layout", key="dqn_eval_fixed_btn",
                                           disabled=st.session_state.dqn_result is None)
        dqn_eval_random_clicked = st.button("Evaluate Random Layouts", key="dqn_eval_random_btn",
                                            disabled=st.session_state.dqn_result is None)
        dqn_eval_unseen_clicked = st.button("Evaluate Unseen Layouts", key="dqn_eval_unseen_btn",
                                            disabled=st.session_state.dqn_result is None)
        dqn_rollout_clicked = st.button("Generate Greedy Replay", key="dqn_rollout_btn",
                                        disabled=st.session_state.dqn_result is None)
        dqn_save_clicked = st.button("Save Model", key="dqn_save_btn",
                                     disabled=st.session_state.dqn_result is None)
        dqn_load_clicked = st.button("Load Saved Run", key="dqn_load_btn", disabled=selected_dqn_stem is None)

    obstacle_max = max(int(dqn_min_obs), int(dqn_max_obs))
    dqn_reward_cfg = Room5RewardConfig(distance_progress_scale=float(dqn_progress_scale))

    def make_room5_env(*, fixed_layout: bool | None = None, layout_seed: int | None = None) -> Room5Obstacles:
        use_fixed = dqn_fixed_layout if fixed_layout is None else fixed_layout
        obs_cfg = Room5ObstacleConfig(
            min_obstacles=int(dqn_min_obs),
            max_obstacles=obstacle_max,
            observation_distance_m=float(dqn_obs_dist),
            layout_seed=int(dqn_layout_seed if layout_seed is None else layout_seed),
            fixed_layout=bool(use_fixed),
        )
        return Room5Obstacles(
            motion_config=Room4MotionConfig(time_step_s=0.05),
            obstacle_config=obs_cfg,
            reward_config=dqn_reward_cfg,
            max_steps=int(dqn_max_steps),
        )

    if dqn_train_clicked:
        with st.spinner(f"Training NumPy DQN for {dqn_episodes} episodes..."):
            agent = DQNAgent(lambda: make_room5_env(), dqn_config)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _dqn_cb(ep, total, metrics):
                progress_bar.progress((ep + 1) / total)
                if ep % max(1, total // 50) == 0 or ep == total - 1:
                    status_text.text(
                        f"Episode {ep + 1}/{total} | "
                        f"Reward={metrics.total_reward:.1f} | "
                        f"Eps={metrics.epsilon:.3f} | "
                        f"Success={'Yes' if metrics.success else 'No'} | "
                        f"Obstacle hits={metrics.obstacle_collisions}"
                    )

            result = agent.train(progress_callback=_dqn_cb, progress_every=1)
            safe_result = make_dqn_result_session_safe(result)
            stem = _auto_run_stem(os.path.join("storage", "models", "room5_dqn"), "dqn_auto")
            save_dqn_model(safe_result, stem, environment_factory=lambda: make_room5_env())
            st.session_state.dqn_result = safe_result
            st.session_state.dqn_network = DQNNetwork.from_weights(dict(safe_result.weights))
            st.session_state.dqn_meta = None
            st.session_state.dqn_train_key = (
                int(dqn_episodes),
                float(dqn_lr),
                float(dqn_gamma),
                int(dqn_max_steps),
                (
                    dqn_eps_kind,
                    float(dqn_eps_start),
                    float(dqn_eps_min),
                    float(dqn_eps_decay),
                    int(dqn_linear_decay),
                ),
                int(dqn_replay_capacity),
                int(dqn_batch_size),
                int(dqn_warmup),
                int(dqn_target_update),
                int(dqn_hidden_units),
                int(dqn_train_seed),
                int(dqn_min_obs),
                int(obstacle_max),
                float(dqn_obs_dist),
                int(dqn_layout_seed),
                bool(dqn_fixed_layout),
                float(dqn_progress_scale),
            )
            st.session_state.dqn_model_stem = stem
            st.session_state.dqn_result_source = "live"
            st.session_state.dqn_autoload_disabled = False
            _clear_room5_outputs()
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    if dqn_load_clicked:
        if selected_dqn_stem is None:
            st.info("No Bonus Room model found. Run training here or use tools/generate_local_models.py --showcase.")
        else:
            try:
                _load_room5_model_into_state(selected_dqn_stem, dqn_config)
                st.success(f"Loaded saved run from {selected_dqn_stem}")
                st.rerun()
            except ValueError as e:
                st.error(f"Load failed: {e}")

    dqn_result = st.session_state.dqn_result
    dqn_network = st.session_state.dqn_network

    if dqn_result is not None:
        if dqn_network is None:
            dqn_network = DQNNetwork.from_weights(dict(dqn_result.weights))
            st.session_state.dqn_network = dqn_network

        if dqn_eval_fixed_clicked:
            with st.spinner(f"Evaluating fixed validation layout ({dqn_eval_ep} episodes)..."):
                st.session_state.dqn_eval_fixed = evaluate_dqn_policy(
                    lambda: make_room5_env(fixed_layout=True),
                    dqn_network,
                    n_episodes=int(dqn_eval_ep),
                    seeds=range(int(dqn_eval_ep)),
                    layout_seeds=[int(dqn_layout_seed)],
                    max_steps=int(dqn_max_steps),
                    category="fixed_validation_layout",
                )
                _persist_room5_outputs_if_saved()
                st.rerun()

        if dqn_eval_random_clicked:
            with st.spinner(f"Evaluating seeded random layouts ({dqn_eval_ep} episodes)..."):
                st.session_state.dqn_eval_random = evaluate_dqn_policy(
                    lambda: make_room5_env(fixed_layout=False),
                    dqn_network,
                    n_episodes=int(dqn_eval_ep),
                    seeds=range(int(dqn_eval_ep)),
                    layout_seeds=[int(dqn_layout_seed) + i for i in range(int(dqn_eval_ep))],
                    max_steps=int(dqn_max_steps),
                    category="seeded_random_layouts",
                )
                _persist_room5_outputs_if_saved()
                st.rerun()

        if dqn_eval_unseen_clicked:
            with st.spinner(f"Evaluating unseen layouts ({dqn_eval_ep} episodes)..."):
                st.session_state.dqn_eval_unseen = evaluate_dqn_policy(
                    lambda: make_room5_env(fixed_layout=False),
                    dqn_network,
                    n_episodes=int(dqn_eval_ep),
                    seeds=range(10_000, 10_000 + int(dqn_eval_ep)),
                    layout_seeds=range(50_000, 50_000 + int(dqn_eval_ep)),
                    max_steps=int(dqn_max_steps),
                    category="unseen_random_layouts",
                )
                _persist_room5_outputs_if_saved()
                st.rerun()

        if dqn_rollout_clicked:
            with st.spinner("Generating greedy Bonus Room replay..."):
                st.session_state.dqn_rollout = rollout_dqn_policy(
                    lambda: make_room5_env(fixed_layout=dqn_fixed_layout),
                    dqn_network,
                    seed=int(dqn_rollout_seed),
                    layout_seed=int(dqn_rollout_layout_seed),
                    max_steps=int(dqn_max_steps),
                )
                st.session_state.dqn_rollout_fixed_layout = bool(dqn_fixed_layout)
                st.session_state.dqn_rollout_key = (
                    int(dqn_rollout_seed), int(dqn_rollout_layout_seed), dqn_fixed_layout,
                )
                _persist_room5_outputs_if_saved()
                st.rerun()

        if dqn_save_clicked:
            import os
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room5_dqn", f"dqn_{ts}")
            save_dqn_model(dqn_result, stem, environment_factory=lambda: make_room5_env())
            st.session_state.dqn_model_stem = stem
            _persist_room5_outputs_if_saved()
            st.success(f"Model saved to {stem}")

        if (
            st.session_state.dqn_eval_fixed is None
            or st.session_state.dqn_eval_random is None
            or st.session_state.dqn_eval_unseen is None
            or st.session_state.dqn_rollout is None
        ):
            auto_eval_ep = min(int(dqn_eval_ep), 25)
            with st.spinner("Preparing Bonus Room baseline evaluations and replay..."):
                if st.session_state.dqn_eval_fixed is None:
                    st.session_state.dqn_eval_fixed = evaluate_dqn_policy(
                        lambda: make_room5_env(fixed_layout=True),
                        dqn_network,
                        n_episodes=auto_eval_ep,
                        seeds=range(auto_eval_ep),
                        layout_seeds=[int(dqn_layout_seed)],
                        max_steps=int(dqn_max_steps),
                        category="fixed_validation_layout",
                    )
                    _persist_room5_outputs_if_saved()
                if st.session_state.dqn_eval_random is None:
                    st.session_state.dqn_eval_random = evaluate_dqn_policy(
                        lambda: make_room5_env(fixed_layout=False),
                        dqn_network,
                        n_episodes=auto_eval_ep,
                        seeds=range(auto_eval_ep),
                        layout_seeds=[int(dqn_layout_seed) + i for i in range(auto_eval_ep)],
                        max_steps=int(dqn_max_steps),
                        category="seeded_random_layouts",
                    )
                    _persist_room5_outputs_if_saved()
                if st.session_state.dqn_eval_unseen is None:
                    st.session_state.dqn_eval_unseen = evaluate_dqn_policy(
                        lambda: make_room5_env(fixed_layout=False),
                        dqn_network,
                        n_episodes=auto_eval_ep,
                        seeds=range(10_000, 10_000 + auto_eval_ep),
                        layout_seeds=range(50_000, 50_000 + auto_eval_ep),
                        max_steps=int(dqn_max_steps),
                        category="unseen_random_layouts",
                    )
                    _persist_room5_outputs_if_saved()
                if st.session_state.dqn_rollout is None:
                    st.session_state.dqn_rollout = rollout_dqn_policy(
                        lambda: make_room5_env(fixed_layout=dqn_fixed_layout),
                        dqn_network,
                        seed=int(dqn_rollout_seed),
                        layout_seed=int(dqn_rollout_layout_seed),
                        max_steps=int(dqn_max_steps),
                    )
                    st.session_state.dqn_rollout_fixed_layout = bool(dqn_fixed_layout)
                    st.session_state.dqn_rollout_key = (
                        int(dqn_rollout_seed), int(dqn_rollout_layout_seed), dqn_fixed_layout,
                    )
                    _persist_room5_outputs_if_saved()

    dqn_meta = st.session_state.get("dqn_meta")
    if dqn_meta:
        render_model_provenance(
            title="Bonus Room - Dynamic Obstacles",
            model_stem=st.session_state.dqn_model_stem,
            metadata=dqn_meta,
            evaluation_success=final_summary_success("Room 5"),
        )

    env_meta = dqn_meta.get("environment_config", {}) if isinstance(dqn_meta, dict) else {}
    observation_size = (
        dqn_meta.get("input_dim")
        if isinstance(dqn_meta, dict) and dqn_meta.get("input_dim") is not None
        else (dqn_network.input_dim if dqn_network is not None else 22)
    )
    with st.container(border=True):
        st.markdown("#### Bonus Proof")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observation Vector", f"{observation_size} features")
        c2.metric("Obstacle Width", f"{float(env_meta.get('obstacle_width_m', 0.5)):.1f}m")
        c3.metric("Observation Range", f"{float(env_meta.get('observation_distance_m', dqn_obs_dist)):.2f}m")
        c4.metric("Layout Evaluations", "Fixed / Random / Unseen")

        eval_rows = []
        for label, summary in [
            ("Fixed layout", st.session_state.dqn_eval_fixed),
            ("Random layouts", st.session_state.dqn_eval_random),
            ("Unseen layouts", st.session_state.dqn_eval_unseen),
        ]:
            if summary is None:
                continue
            eval_source = (
                "live evaluation"
                if st.session_state.get("dqn_result_source") == "live"
                else "saved model evaluation"
            )
            eval_rows.append(
                {
                    "Evaluation": label,
                    "Success Rate": f"{summary.success_rate:.1%}",
                    "Mean Return": f"{summary.mean_return:.2f}",
                    "Mean Steps": f"{summary.mean_steps:.1f}",
                    "Source": eval_source,
                }
            )
        if eval_rows:
            st.dataframe(eval_rows, width="stretch", hide_index=True)

    t1, t2, t3, t4 = st.tabs(["Training Progress", "Evaluation", "Greedy Replay", "Action Values"])

    with t1:
        preview_env = make_room5_env()
        preview_env.reset(seed=int(dqn_train_seed), layout_seed=int(dqn_layout_seed))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observation Size", len(preview_env.state))
        c2.metric("Obstacle Width", f"{preview_env.obstacle_width_m:.1f}m")
        c3.metric("Obstacle Count", len(preview_env.obstacles))
        c4.metric("Layout Signature", preview_env.layout_signature())
        render_html(_render_room5_svg(preview_env))

        if dqn_result is None:
            autoload_error = st.session_state.get("dqn_autoload_error")
            if autoload_error:
                st.error(f"Bonus Room model auto-load failed: {autoload_error}")
        elif dqn_result.metrics:
            rows = _room5_training_rows(dqn_result.metrics)
            rewards = np.array([row["total_reward"] for row in rows], dtype=float)
            successes = np.array([1.0 if row["success"] else 0.0 for row in rows], dtype=float)
            collisions = np.array([row["obstacle_collisions"] for row in rows], dtype=float)
            epsilons = np.array([row["epsilon"] for row in rows], dtype=float)
            losses = np.array([row["mean_loss"] for row in rows], dtype=float)
            steps = np.array([row["steps"] for row in rows], dtype=float)
            window = min(50, max(1, len(rows)))
            recent = rows[-window:]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Episodes", len(rows))
            c2.metric(f"Recent Success ({window})", f"{sum(r['success'] for r in recent) / window:.1%}")
            c3.metric("Final Epsilon", f"{dqn_result.final_epsilon:.4f}")
            c4.metric("Recent Obstacle Rate", f"{sum(r['obstacle_collisions'] for r in recent) / window:.1%}")
            st.subheader("Reward per Episode")
            st.line_chart({"total_reward": rewards})
            st.subheader("Steps per Episode")
            st.line_chart({"steps": steps})
            st.subheader("Success and Obstacle Collisions")
            st.line_chart({"success": successes, "obstacle_collision": collisions})
            st.subheader("Epsilon and Loss")
            st.line_chart({"epsilon": epsilons, "mean_loss": losses})
            st.dataframe(rows[-min(20, len(rows)):], width="stretch")
        else:
            ev_fixed = st.session_state.dqn_eval_fixed
            ev_random = st.session_state.dqn_eval_random
            ev_unseen = st.session_state.dqn_eval_unseen
            rollout = st.session_state.dqn_rollout
            if ev_fixed is not None and ev_random is not None and ev_unseen is not None:
                st.subheader("Loaded Showcase Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Fixed Layout", f"{ev_fixed.success_rate:.1%}")
                c2.metric("Random Layouts", f"{ev_random.success_rate:.1%}")
                c3.metric("Unseen Layouts", f"{ev_unseen.success_rate:.1%}")
            if rollout is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric("Greedy Replay Success", "Yes" if rollout.success else "No")
                c2.metric("Greedy Replay Steps", rollout.steps)
                c3.metric("Greedy Replay Return", f"{rollout.total_reward:.1f}")

    with t2:
        summaries = [
            ("Fixed Validation Layout", "fixed", True, st.session_state.dqn_eval_fixed),
            ("Seeded Random Layouts", "random", False, st.session_state.dqn_eval_random),
            ("Unseen Random Layouts", "unseen", False, st.session_state.dqn_eval_unseen),
        ]
        shown = False
        for label, key, fixed_layout, summary in summaries:
            if summary is None:
                continue
            shown = True
            st.subheader(label)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success Rate", f"{summary.success_rate:.1%}")
            c2.metric("Mean Return", f"{summary.mean_return:.2f}")
            c3.metric("Mean Steps", f"{summary.mean_steps:.1f}")
            c4.metric("Obstacle Hits", summary.obstacle_collision_count)
            rollout_rows = [
                {
                    "episode": i + 1,
                    "seed": r.seed,
                    "layout_seed": r.layout_seed,
                    "success": r.success,
                    "steps": r.steps,
                    "return": r.total_reward,
                    "obstacle_collisions": r.obstacle_collisions,
                    "boundary_collisions": r.boundary_collisions,
                }
                for i, r in enumerate(summary.rollouts)
            ]
            st.dataframe(
                rollout_rows[:10],
                width="stretch",
            )
            if summary.rollouts:
                rollout_idx = st.selectbox(
                    "Rendered rollout",
                    options=list(range(len(summary.rollouts))),
                    key=f"dqn_eval_rollout_{key}",
                    format_func=lambda i, rollouts=summary.rollouts: (
                        f"Episode {i + 1} | seed {rollouts[i].seed} | layout {rollouts[i].layout_seed}"
                    ),
                )
                selected_rollout = summary.rollouts[int(rollout_idx)]
                disp_env = _room5_display_env_for_rollout(
                    make_room5_env,
                    selected_rollout,
                    fixed_layout=fixed_layout,
                )
                render_html(_render_room5_svg(disp_env, selected_rollout))
            else:
                st.info("No rollout records were saved for this evaluation.")
        if not shown:
            if dqn_result is None:
                st.info("Train or load a DQN model to show Room 5 evaluations.")
            else:
                st.info("Preparing Room 5 evaluation outputs.")

    with t3:
        rollout = st.session_state.dqn_rollout
        if rollout is not None:
            rollout_fixed_layout = st.session_state.get("dqn_rollout_fixed_layout")
            if rollout_fixed_layout is None:
                rollout_fixed_layout = bool(dqn_fixed_layout)
            disp_env = _room5_display_env_for_rollout(
                make_room5_env,
                rollout,
                fixed_layout=bool(rollout_fixed_layout),
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Success", "Yes" if rollout.success else "No")
            c2.metric("Steps", rollout.steps)
            c3.metric("Return", f"{rollout.total_reward:.2f}")
            c4.metric("Obstacle Hits", rollout.obstacle_collisions)
            render_html(_render_room5_svg(disp_env, rollout))
            st.dataframe(
                [
                    {
                        "step": step.index,
                        "action": step.requested_action.name,
                        "reward": step.reward,
                        "cumulative": step.cumulative_reward,
                        "visible_obstacles": step.visible_obstacle_count,
                        "event": step.event,
                        "distance_to_exit_m": step.distance_to_exit_m,
                    }
                    for step in rollout.trajectory[:50]
                ],
                width="stretch",
            )
        elif dqn_result is None:
            st.info("Train or load a DQN model to show a greedy replay.")
        else:
            st.info("Preparing Room 5 greedy replay output.")

    with t4:
        if dqn_network is not None:
            value_env = make_room5_env()
            obs = value_env.reset(seed=int(dqn_train_seed), layout_seed=int(dqn_layout_seed))
            q_vals = extract_dqn_action_values(dqn_network, obs)
            st.dataframe(
                [{"action": action, "q_value": value} for action, value in q_vals.items()],
                width="stretch",
            )

# ============================================================
# MODE: Algorithm Comparison
# ============================================================
elif st.session_state.mode == "Algorithm Comparison":
    st.header("SARSA vs Q-Learning — Controlled Comparison")
    st.caption(
        "These results are for this benchmark map only. They compare update rules and tuned settings here; "
        "they do not prove universal algorithm superiority."
    )

    with st.sidebar:
        st.markdown("**Comparison Settings**")
        comp_episodes = st.number_input("Episodes", min_value=100, max_value=10000, value=2000, step=500,
                                         key="comp_episodes",
                                         help="Training episodes per algorithm per seed.")
        comp_alpha = st.slider("Alpha", 0.01, 1.0, 0.10, step=0.01, key="comp_alpha",
                               help="Learning rate for both algorithms.")
        comp_gamma = st.slider("Gamma", 0.50, 0.99, 0.95, step=0.01, key="comp_gamma",
                               help="Discount factor for both algorithms.")
        comp_decay = st.slider("Epsilon Decay", 0.9, 1.0, 0.995, step=0.001, key="comp_decay",
                               help="Exponential epsilon decay rate per episode.")
        comp_seeds = st.number_input("Training Seeds", min_value=1, max_value=10, value=5, step=1,
                                      key="comp_seeds",
                                      help="Number of random seeds to average over. More = more reliable but slower.")
        comp_eval_ep = st.number_input("Eval Episodes per Model", min_value=10, max_value=500, value=100, step=10,
                                        key="comp_eval_ep",
                                        help="Evaluation episodes per trained model per seed.")

        load_saved_clicked = st.button("Load Final Saved Comparison", type="primary", key="comp_load_saved")
        short_clicked = st.button("Run Short Demonstration", key="comp_run_short")
        full_clicked = st.button("Run Full Comparison", key="comp_run_full")

    if st.session_state.comp_matched is None and st.session_state.comp_tuned is None:
        _saved_comparison_into_state()

    if load_saved_clicked:
        if _saved_comparison_into_state():
            st.success("Loaded final saved comparison from storage/experiments/final.")
            st.rerun()
        st.error("Final saved comparison artifacts were not found.")

    if short_clicked:
        with st.spinner("Running short matched comparison..."):
            matched = run_matched_comparison(
                alpha=0.10,
                gamma=0.95,
                episodes=300,
                epsilon_decay=0.995,
                training_seeds=[0, 1],
                eval_seeds=range(20),
            )
        with st.spinner("Running short tuned comparison..."):
            tuned = run_tuned_comparison(
                sarsa_configs=[{"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.99}],
                q_configs=[{"alpha": 0.50, "gamma": 0.99, "epsilon_decay": 0.999}],
                training_seeds=[0],
                eval_seeds=range(20),
                episodes=300,
            )
        st.session_state.comp_matched = matched
        st.session_state.comp_tuned = tuned
        st.session_state.comp_metadata = {
            "map_signature": "live demonstration",
            "match_config": {
                "episodes": 300,
                "max_steps": 200,
                "training_seeds": [0, 1],
                "eval_seeds": list(range(20)),
                "alpha": 0.10,
                "gamma": 0.95,
                "epsilon_decay": 0.995,
            },
        }
        st.session_state.comp_source = "Short live demonstration"
        st.session_state.comp_key = ("short", 300, 2, 20)
        st.rerun()

    if full_clicked:
        comp_key = (comp_episodes, comp_alpha, comp_gamma, comp_decay, comp_seeds, comp_eval_ep)
        with st.spinner("Running matched comparison..."):
            matched = run_matched_comparison(
                alpha=comp_alpha, gamma=comp_gamma,
                episodes=comp_episodes, epsilon_decay=comp_decay,
                training_seeds=list(range(comp_seeds)),
                eval_seeds=range(comp_eval_ep),
            )

        with st.spinner("Running tuned comparison..."):
            tuned = run_tuned_comparison(
                sarsa_configs=[
                    {"alpha": comp_alpha, "gamma": comp_gamma, "epsilon_decay": comp_decay},
                    {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
                ],
                q_configs=[
                    {"alpha": comp_alpha, "gamma": comp_gamma, "epsilon_decay": comp_decay},
                    {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
                ],
                training_seeds=list(range(min(3, comp_seeds))),
                eval_seeds=range(comp_eval_ep),
                episodes=comp_episodes,
            )

        save_comparison(matched, tuned)
        st.session_state.comp_matched = matched
        st.session_state.comp_tuned = tuned
        st.session_state.comp_metadata = {
            "map_signature": "live full comparison",
            "match_config": {
                "episodes": int(comp_episodes),
                "max_steps": 200,
                "training_seeds": list(range(int(comp_seeds))),
                "eval_seeds": list(range(int(comp_eval_ep))),
                "alpha": float(comp_alpha),
                "gamma": float(comp_gamma),
                "epsilon_decay": float(comp_decay),
            },
        }
        st.session_state.comp_source = "Full live comparison"
        st.session_state.comp_key = comp_key
        st.rerun()

    comp_matched = st.session_state.comp_matched
    comp_tuned = st.session_state.comp_tuned
    comp_meta = st.session_state.comp_metadata or {}

    if comp_matched is not None and comp_tuned is not None:
        match_config = comp_meta.get("match_config", {})
        st.markdown(f"**Source:** {st.session_state.comp_source or 'Current session'}")
        with st.container(border=True):
            st.markdown("#### Fairness Proof")
            st.dataframe(
                [
                    {"Check": "Same map", "Evidence": str(comp_meta.get("map_signature", "Room 2 benchmark map"))},
                    {"Check": "Same rewards", "Evidence": "Both use the Room 2 default reward configuration."},
                    {"Check": "Same slip configuration", "Evidence": "Both use intended=0.8, left=0.1, right=0.1."},
                    {"Check": "Same training seeds", "Evidence": str(match_config.get("training_seeds", "same list per algorithm"))},
                    {"Check": "Same evaluation seeds", "Evidence": str(match_config.get("eval_seeds", "same list per algorithm"))},
                ],
                width="stretch",
                hide_index=True,
            )

        # --- Matched comparison summary ---
        st.subheader("Matched Comparison - Same Hyperparameters, Update Rule Comparison")
        sarsa_m, q_m, paired = _matched_parts(comp_matched)

        data = []
        for s, q in zip(sarsa_m, q_m):
            data.append({
                "Seed": s.get("seed"),
                "SARSA SR": f"{_metric_float(s, 'success_rate'):.1%}",
                "Q-Learn SR": f"{_metric_float(q, 'success_rate'):.1%}",
                "SARSA Return": f"{_metric_float(s, 'mean_return'):.1f}",
                "Q-Learn Return": f"{_metric_float(q, 'mean_return'):.1f}",
                "SARSA Steps": f"{_metric_float(s, 'mean_steps'):.1f}",
                "Q-Learn Steps": f"{_metric_float(q, 'mean_steps'):.1f}",
                "SARSA Traps": int(_metric_float(s, "total_traps")),
                "Q-Learn Traps": int(_metric_float(q, "total_traps")),
            })
        st.dataframe(data, width="stretch", hide_index=True)

        s_sr = [_metric_float(r, "success_rate") for r in sarsa_m]
        q_sr = [_metric_float(r, "success_rate") for r in q_m]
        s_ret = [_metric_float(r, "mean_return") for r in sarsa_m]
        q_ret = [_metric_float(r, "mean_return") for r in q_m]
        paired_sr = [_metric_float(r, "diff_success_rate") for r in paired]
        paired_ret = [_metric_float(r, "diff_mean_return") for r in paired]

        c1, c2, c3 = st.columns(3)
        c1.metric("SARSA Mean SR", f"{np.mean(s_sr):.1%}", delta=None)
        c2.metric("Q-Learn Mean SR", f"{np.mean(q_sr):.1%}", delta=None)
        c3.metric("Mean Paired SR Diff", f"{np.mean(paired_sr):.1%}" if paired_sr else "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("SARSA Mean Return", f"{np.mean(s_ret):.1f}")
        c2.metric("Q-Learn Mean Return", f"{np.mean(q_ret):.1f}")
        c3.metric("Mean Paired Return Diff", f"{np.mean(paired_ret):.1f}" if paired_ret else "N/A")

        # Charts
        st.subheader("Per-Seed Success Rate")
        chart = {"SARSA": s_sr, "Q-Learning": q_sr}
        st.bar_chart(chart)

        st.subheader("Per-Seed Mean Return")
        st.bar_chart({"SARSA": s_ret, "Q-Learning": q_ret})

        # --- Tuned comparison ---
        st.subheader("Tuned Comparison - Each Algorithm Uses Its Best Confirmed Parameters")
        tuned_data = []
        for row in comp_tuned:
            r = _row_dict(row)
            tuned_data.append({
                "Algorithm": r.get("algorithm"),
                "Config": str(r.get("config")),
                "Mean SR": f"{_metric_float(r, 'success_rate_mean'):.1%}",
                "SR Std": f"{_metric_float(r, 'success_rate_std'):.2%}",
                "Mean Return": f"{_metric_float(r, 'mean_return_mean'):.1f}",
                "Mean Steps": f"{_metric_float(r, 'mean_steps_mean'):.1f}",
                "Traps": int(_metric_float(r, "total_traps")),
            })
        st.dataframe(tuned_data, width="stretch", hide_index=True)
    else:
        st.info("Load final saved results or run a demonstration from the sidebar.")
