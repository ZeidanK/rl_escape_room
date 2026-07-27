"""Room 4 — Momentum Chamber game view. Continuous state space with trajectory visualization."""

import time

import streamlit as st
import numpy as np

from game.html_rendering import render_html
from game.theme import get_theme
from game.game_view_common import (
    render_back_button,
    check_and_unlock_achievements,
    render_room_transition,
)
from game.canvas_renderer import render_continuous_trajectory_canvas

from core.types import (
    ContinuousRewardConfig,
    ContinuousRolloutResult,
    Room4MotionConfig,
    StartMode,
    VelocityAction,
)
from environments.room4_continuous import Room4Continuous
from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    LinearTileQFunction,
    load_approximate_model,
    rollout_approximate_policy,
)
from features.tile_coding import TileCoder, TileCodingConfig
from game.presentation import (
    final_summary_success,
    read_json,
    render_assignment_proof,
    render_model_provenance,
    render_open_lab_button,
    stage_options,
)


def _training_config(meta: dict) -> dict:
    return meta.get("training_config", {})


def _tile_config_from_meta(meta: dict) -> TileCodingConfig:
    # Older saved artifacts may store tiles as one "tiles_xy" value; newer
    # artifacts store separate x/y counts.
    tc = meta.get("tile_coding_config", {})
    tiles_xy = meta.get("tiles_xy", 10)
    return TileCodingConfig(
        num_tilings=int(tc.get("num_tilings", meta.get("num_tilings", 8))),
        tiles_x=int(tc.get("tiles_x", tiles_xy)),
        tiles_y=int(tc.get("tiles_y", tiles_xy)),
        include_velocity=bool(tc.get("include_velocity", True)),
    )


def _motion_config_from_meta(meta: dict) -> Room4MotionConfig:
    motion = meta.get("motion_config", {})
    return Room4MotionConfig(
        room_width_m=float(motion.get("room_width_m", 10.0)),
        room_height_m=float(motion.get("room_height_m", 10.0)),
        time_step_s=float(motion.get("time_step_s", 0.02)),
        exit_center=tuple(motion.get("exit_center", (9.5, 9.5))),
        exit_radius_m=float(motion.get("exit_radius_m", 0.35)),
    )


def _reward_config_from_meta(meta: dict) -> ContinuousRewardConfig:
    reward = meta.get("reward_config", {})
    return ContinuousRewardConfig(
        step=float(reward.get("step", -0.01)),
        exit=float(reward.get("exit", 100.0)),
        boundary_collision=float(reward.get("boundary_collision", -1.0)),
        timeout=float(reward.get("timeout", -25.0)),
        distance_progress_scale=float(reward.get("distance_progress_scale", meta.get("progress_scale", 1.0))),
    )


def _seed_from_meta(meta: dict, default: int = 42) -> int:
    cfg = _training_config(meta)
    return int(meta.get("training_seed", cfg.get("seed", meta.get("seed", default))))


def _max_steps_from_meta(meta: dict, default: int = 750) -> int:
    return int(_training_config(meta).get("max_steps", default))


def _start_mode_from_meta(meta: dict) -> StartMode:
    cfg = _training_config(meta)
    try:
        return StartMode(cfg.get("start_mode", meta.get("start_mode", StartMode.FIXED.value)))
    except ValueError:
        return StartMode.FIXED


def _q_function_from_weights(weights: np.ndarray, tc_cfg: TileCodingConfig, motion_cfg: Room4MotionConfig) -> LinearTileQFunction:
    # Recreate the linear Q-function wrapper around saved weights so rollout
    # code can ask for action values exactly like after training.
    tile_coder = TileCoder(tc_cfg, room_width=motion_cfg.room_width_m, room_height=motion_cfg.room_height_m)
    q_func = LinearTileQFunction(tile_coder, n_actions=len(VelocityAction))
    q_func._weights = weights.copy()
    return q_func


def _preferred_room4_model_stem() -> str | None:
    import glob
    import os

    model_dir = os.path.join("storage", "models", "room4_approximate_sarsa")
    showcase = os.path.join(model_dir, "showcase_approx")
    if os.path.exists(showcase + ".json") and os.path.exists(showcase + ".npz"):
        return showcase

    files = glob.glob(os.path.join(model_dir, "*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    if not files:
        return None
    return max(files, key=os.path.getmtime).replace(".json", "")


def _load_room4_game_model(filepath_stem: str) -> None:
    weights, meta = load_approximate_model(filepath_stem)
    st.session_state.r4g_weights = weights
    st.session_state.r4g_meta = meta
    st.session_state.r4g_rollout = None
    st.session_state.r4g_loaded = True
    st.session_state.r4g_model_stem = filepath_stem
    st.session_state.r4g_load_error = None
    st.session_state.r4g_autoload_disabled = False


def _final_distance_to_exit_m(rollout, motion_cfg: Room4MotionConfig) -> float:
    x, y = rollout.final_state[:2]
    ex, ey = motion_cfg.exit_center
    return float(np.hypot(x - ex, y - ey))


def _distance_for_prefix(rollout: ContinuousRolloutResult, frame_index: int) -> float:
    points = [rollout.start_state]
    points.extend(step.next_state for step in rollout.trajectory[:frame_index])
    distance = 0.0
    for previous, current in zip(points, points[1:]):
        distance += float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
    return distance


def _rollout_prefix(
    rollout: ContinuousRolloutResult,
    frame_index: int,
    motion_cfg: Room4MotionConfig,
) -> ContinuousRolloutResult:
    frame_index = max(0, min(int(frame_index), len(rollout.trajectory)))
    trajectory = tuple(rollout.trajectory[:frame_index])
    final_state = rollout.start_state if not trajectory else trajectory[-1].next_state
    terminal_step = trajectory[-1] if trajectory else None
    return ContinuousRolloutResult(
        seed=rollout.seed,
        start_state=rollout.start_state,
        final_state=final_state,
        total_reward=float(sum(step.reward for step in trajectory)),
        steps=frame_index,
        simulated_time_s=frame_index * motion_cfg.time_step_s,
        success=bool(frame_index == len(rollout.trajectory) and rollout.success),
        terminated=bool(terminal_step.terminated) if terminal_step else False,
        truncated=bool(terminal_step.truncated) if terminal_step else False,
        collision_count=sum(1 for step in trajectory if step.collision),
        distance_travelled_m=_distance_for_prefix(rollout, frame_index),
        trajectory=trajectory,
    )


def _current_frame_state(
    rollout: ContinuousRolloutResult,
    frame_index: int,
    motion_cfg: Room4MotionConfig,
) -> dict[str, object]:
    frame_index = max(0, min(int(frame_index), len(rollout.trajectory)))
    if frame_index == 0:
        return {
            "state": rollout.start_state,
            "action": "Not chosen yet",
            "reward": 0.0,
            "cumulative": 0.0,
            "event": "Start",
            "collision": "None",
            "simulated_time_s": 0.0,
        }

    prefix = rollout.trajectory[:frame_index]
    step = prefix[-1]
    return {
        "state": step.next_state,
        "action": step.requested_action.name,
        "reward": step.reward,
        "cumulative": float(sum(s.reward for s in prefix)),
        "event": step.event or step.collision or "None",
        "collision": step.collision or "None",
        "simulated_time_s": frame_index * motion_cfg.time_step_s,
    }


def _rollout_from_weights(
    weights: np.ndarray,
    meta: dict,
    *,
    seed: int,
) -> tuple[ContinuousRolloutResult, Room4MotionConfig, ContinuousRewardConfig, TileCodingConfig, StartMode, int]:
    tc_cfg = _tile_config_from_meta(meta)
    motion_cfg = _motion_config_from_meta(meta)
    reward_cfg = _reward_config_from_meta(meta)
    start_mode = _start_mode_from_meta(meta)
    max_steps = _max_steps_from_meta(meta)
    q_func = _q_function_from_weights(weights, tc_cfg, motion_cfg)
    make_env = lambda: Room4Continuous(
        motion_config=motion_cfg,
        reward_config=reward_cfg,
        max_steps=max_steps,
        start_mode=start_mode,
        seed=seed,
    )
    rollout = rollout_approximate_policy(make_env, q_func, seed=seed, max_steps=max_steps)
    return rollout, motion_cfg, reward_cfg, tc_cfg, start_mode, max_steps


def _room4_generalization_rows() -> list[dict[str, str]]:
    confirmation = read_json("storage/experiments/final/room4_approximate_sarsa_confirmation.json")
    if not confirmation or not confirmation.get("confirmation_results"):
        return []

    best = confirmation["confirmation_results"][0]
    return [
        {
            "Evaluation": "Training/fixed start",
            "Success Rate": f"{float(best.get('fixed_training_start_success_rate', 0.0)):.1%}",
            "Source": "room4_approximate_sarsa_confirmation.json",
        },
        {
            "Evaluation": "Fixed unseen starts",
            "Success Rate": f"{float(best.get('fixed_unseen_starts_success_rate', 0.0)):.1%}",
            "Source": "room4_approximate_sarsa_confirmation.json",
        },
        {
            "Evaluation": "Random lower-left",
            "Success Rate": f"{float(best.get('random_lower_left_success_rate', 0.0)):.1%}",
            "Source": "room4_approximate_sarsa_confirmation.json",
        },
        {
            "Evaluation": "Random room",
            "Success Rate": f"{float(best.get('random_room_success_rate', 0.0)):.1%}",
            "Source": "room4_approximate_sarsa_confirmation.json",
        },
    ]


def _render_stage_pair_comparison(stage_entries: list[tuple[str, str]], seed: int) -> None:
    by_label = dict(stage_entries)
    if "Beginning" not in by_label or "Final" not in by_label:
        return

    with st.container(border=True):
        st.markdown("#### Early vs Final Trajectory")
        columns = st.columns(2)
        for column, label in zip(columns, ("Beginning", "Final")):
            with column:
                try:
                    weights, meta = load_approximate_model(by_label[label])
                    rollout, motion_cfg, _, _, _, _ = _rollout_from_weights(weights, meta, seed=seed)
                    env = Room4Continuous(
                        motion_config=motion_cfg,
                        reward_config=_reward_config_from_meta(meta),
                        max_steps=_max_steps_from_meta(meta),
                        start_mode=_start_mode_from_meta(meta),
                        seed=seed,
                    )
                except Exception as exc:
                    st.caption(f"{label} stage could not load: {exc}")
                    continue
                st.markdown(f"**{label}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Success", "Yes" if rollout.success else "No")
                c2.metric("Steps", rollout.steps)
                c3.metric("Return", f"{rollout.total_reward:.1f}")
                render_html(
                    render_continuous_trajectory_canvas(
                        env,
                        rollout,
                        max_arrows=12,
                        grid_subdivisions=12,
                    )
                )


def render_room4_game():
    # Showcase view for a trained approximate SARSA model in continuous space.
    # It loads weights, rebuilds the tile coder, then draws the trajectory.
    theme = get_theme("room4")

    render_html(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        f'The discrete grid disappears. The agent must control velocity in continuous space '
        f'and generalize from overlapping tile-coded features.</div>'
    )
    render_assignment_proof("room4")
    render_open_lab_button("room4", key="r4g_open_lab")

    render_back_button("r4g_back")

    # Initialize session state before controls so first entry can auto-load
    # bundled showcase artifacts without waiting for a manual sidebar click.
    for key in [
        "r4g_weights",
        "r4g_meta",
        "r4g_rollout",
        "r4g_rollout_key",
        "r4g_loaded",
        "r4g_model_stem",
        "r4g_load_error",
        "r4g_playing",
        "r4g_play_index",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "r4g_autoload_disabled" not in st.session_state:
        st.session_state.r4g_autoload_disabled = False

    if (
        st.session_state.r4g_weights is None
        and st.session_state.r4g_meta is None
        and not st.session_state.r4g_autoload_disabled
    ):
        latest = _preferred_room4_model_stem()
        if latest is not None:
            try:
                _load_room4_game_model(latest)
            except ValueError as e:
                st.session_state.r4g_load_error = str(e)

    # Sidebar controls
    with st.sidebar:
        st.header("Room 4 Controls")
        
        # Model loading section
        st.markdown("**Model**")
        load_col, reset_col = st.columns(2)
        if load_col.button("Load Latest Model", key="r4g_load"):
            try:
                latest = _preferred_room4_model_stem()
                if latest:
                    _load_room4_game_model(latest)
                    st.success(f"Loaded model from {latest}")
                else:
                    st.caption("No saved Room 4 models found.")
            except Exception as e:
                st.error(f"Load failed: {e}")
            st.rerun()
        
        if reset_col.button("Reset", key="r4g_reset"):
            for key in [
                "r4g_weights",
                "r4g_meta",
                "r4g_rollout",
                "r4g_rollout_key",
                "r4g_loaded",
                "r4g_model_stem",
                "r4g_load_error",
                "r4g_playing",
                "r4g_play_index",
            ]:
                st.session_state[key] = None
            st.session_state.r4g_autoload_disabled = True
            st.rerun()

        st.markdown("---")
        
        # Display toggles
        st.markdown("**Display**")
        show_trajectory = st.checkbox("Show Trajectory", value=True, key="r4g_traj")
        show_velocity = st.checkbox("Show Velocity Arrows", value=True, key="r4g_vel")
        grid_res = st.slider("Grid Resolution", 10, 50, 30, key="r4g_grid",
                             help="Resolution for trajectory discretization display.")

    weights = st.session_state.r4g_weights
    meta = st.session_state.r4g_meta

    if weights is None or meta is None:
        load_error = st.session_state.get("r4g_load_error")
        if load_error:
            st.caption(f"Room 4 model auto-load skipped: {load_error}")
        else:
            st.caption("No Room 4 model is loaded.")
        return

    stage_entries = stage_options("room4", st.session_state.r4g_model_stem)
    selected_stage = "Final"
    selected_stem = st.session_state.r4g_model_stem
    if stage_entries:
        stage_labels = [label for label, _ in stage_entries]
        default_index = stage_labels.index("Final") if "Final" in stage_labels else len(stage_labels) - 1
        selected_stage = st.selectbox(
            "Policy Stage",
            stage_labels,
            index=default_index,
            key="r4g_stage_selector",
        )
        selected_stem = dict(stage_entries)[selected_stage]

    display_weights = weights
    display_meta = meta
    if selected_stem and selected_stem != st.session_state.r4g_model_stem:
        try:
            display_weights, display_meta = load_approximate_model(selected_stem)
        except Exception as exc:
            st.warning(f"Selected stage could not be loaded: {exc}")
            selected_stage = "Final"
            selected_stem = st.session_state.r4g_model_stem
            display_weights = weights
            display_meta = meta

    # Build environment from saved metadata so the replay uses the same motion
    # and reward settings as training.
    tc_cfg = _tile_config_from_meta(display_meta)
    motion_cfg = _motion_config_from_meta(display_meta)
    reward_cfg = _reward_config_from_meta(display_meta)
    start_mode = _start_mode_from_meta(display_meta)
    max_steps = _max_steps_from_meta(display_meta)
    seed = _seed_from_meta(display_meta)
    
    env = Room4Continuous(
        motion_config=motion_cfg,
        reward_config=reward_cfg,
        max_steps=max_steps,
        start_mode=start_mode,
        seed=seed,
    )

    # Build rollout once and cache it; replay/render controls should not change
    # the evaluated trajectory unless the model is reset/reloaded.
    rollout_key = (selected_stem, seed, max_steps)
    rollout = st.session_state.r4g_rollout
    if rollout is None or st.session_state.r4g_rollout_key != rollout_key:
        q_func = _q_function_from_weights(display_weights, tc_cfg, motion_cfg)
        make_env = lambda: Room4Continuous(
            motion_config=motion_cfg,
            reward_config=reward_cfg,
            max_steps=max_steps,
            start_mode=start_mode,
            seed=seed,
        )
        rollout = rollout_approximate_policy(make_env, q_func, seed=seed, max_steps=max_steps)
        st.session_state.r4g_rollout = rollout
        st.session_state.r4g_rollout_key = rollout_key
        st.session_state.r4g_play_index = len(rollout.trajectory)
        st.session_state.r4g_playing = False
        st.rerun()

    # HUD
    status_badges = []
    if rollout.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif not rollout.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    cfg = _training_config(display_meta)
    final_distance = _final_distance_to_exit_m(rollout, motion_cfg)
    render_model_provenance(
        title=f"Room 4 {selected_stage}",
        model_stem=selected_stem,
        metadata=display_meta,
        evaluation_success=final_summary_success("Room 4") if selected_stage == "Final" else None,
    )

    from game.hud import render_hud
    render_html(render_hud(
        room_name="\U0001f300 Room 4: The Momentum Chamber",
        algorithm=f"Approximate SARSA (Tile Coding) | \u03b1={float(cfg.get('alpha', display_meta.get('alpha', 0.1))):.2f} \u03b3={float(cfg.get('gamma', display_meta.get('gamma', 0.99))):.2f} | Tilings={tc_cfg.num_tilings}",
        state_str=f"({rollout.start_state[0]:.2f}, {rollout.start_state[1]:.2f}, v={rollout.start_state[2]}, {rollout.start_state[3]})",
        action=None,
        reward=None,
        total_reward=rollout.total_reward,
        epsilon=None,
        status_badges=status_badges,
        custom_items=[
            ("Steps", str(rollout.steps)),
            ("Distance", f"{rollout.distance_travelled_m:.2f}m"),
            ("Final Dist.", f"{final_distance:.2f}m"),
        ],
    ))

    # Continuous trajectory visualization
    st.markdown("### Continuous Trajectory Playback")

    max_frame = len(rollout.trajectory)
    if st.session_state.r4g_play_index is None:
        st.session_state.r4g_play_index = max_frame
    if int(st.session_state.r4g_play_index) > max_frame:
        st.session_state.r4g_play_index = max_frame

    b1, b2, b3, b4, b5, b6 = st.columns([1, 1, 1.2, 1, 1, 1.4])
    if b1.button("Beginning", key="r4g_beginning"):
        st.session_state.r4g_play_index = 0
        st.session_state.r4g_playing = False
        st.rerun()
    if b2.button("Previous", key="r4g_previous"):
        st.session_state.r4g_play_index = max(0, int(st.session_state.r4g_play_index) - 1)
        st.session_state.r4g_playing = False
        st.rerun()
    play_label = "Pause" if st.session_state.r4g_playing else "Play"
    if b3.button(play_label, key="r4g_play_pause", type="primary"):
        st.session_state.r4g_playing = not bool(st.session_state.r4g_playing)
        st.rerun()
    if b4.button("Next", key="r4g_next"):
        st.session_state.r4g_play_index = min(max_frame, int(st.session_state.r4g_play_index) + 1)
        st.session_state.r4g_playing = False
        st.rerun()
    if b5.button("End", key="r4g_end"):
        st.session_state.r4g_play_index = max_frame
        st.session_state.r4g_playing = False
        st.rerun()
    speed_label = b6.radio(
        "Speed",
        ["0.5x", "1x", "2x", "4x"],
        index=1,
        horizontal=True,
        key="r4g_speed",
    )

    frame_index = st.slider("Trajectory Step", 0, max_frame, key="r4g_play_index")
    visible_rollout = _rollout_prefix(rollout, frame_index, motion_cfg)
    frame_state = _current_frame_state(rollout, frame_index, motion_cfg)
    render_html(
        render_continuous_trajectory_canvas(
            env,
            visible_rollout,
            max_arrows=20,
            grid_subdivisions=grid_res,
            show_path=show_trajectory,
            show_arrows=show_velocity,
        )
    )

    x, y, vx, vy = frame_state["state"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("X", f"{float(x):.2f}")
    c2.metric("Y", f"{float(y):.2f}")
    c3.metric("Vx", str(int(vx)))
    c4.metric("Vy", str(int(vy)))
    c5.metric("Simulation Step", frame_index)
    c6.metric("Simulated Time", f"{float(frame_state['simulated_time_s']):.2f}s")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Decision Interval", f"{motion_cfg.time_step_s:.2f} seconds")
    c2.metric("Current Action", str(frame_state["action"]))
    c3.metric("Reward", f"{float(frame_state['reward']):.2f}")
    c4.metric("Cumulative Return", f"{float(frame_state['cumulative']):.2f}")
    c5.metric("Collision/Event", str(frame_state["event"]))
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Steps", rollout.steps)
    col2.metric("Reward", f"{rollout.total_reward:.1f}")
    col3.metric("Distance", f"{rollout.distance_travelled_m:.2f}m")
    col4.metric("Final Dist.", f"{final_distance:.2f}m")
    
    if rollout.collision_count > 0:
        st.warning(f"Collisions: {rollout.collision_count}")

    gen_rows = _room4_generalization_rows()
    if gen_rows:
        st.markdown("### Saved Generalization Results")
        st.dataframe(gen_rows, width="stretch", hide_index=True)

    _render_stage_pair_comparison(stage_entries, seed)
    
    # Room transition
    achievements = check_and_unlock_achievements("room4", rollout)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")
    
    render_room_transition("room4", rollout, achievements)
    
    # Legend
    render_html(f"""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Trajectory</span>
        <span class="legend-item">X Collision</span>
        <span class="legend-item">↑→↓← Velocity</span>
    </div>
    """)

    if st.session_state.r4g_playing:
        speed = {"0.5x": 1, "1x": 2, "2x": 4, "4x": 8}.get(speed_label, 2)
        if int(st.session_state.r4g_play_index) >= max_frame:
            st.session_state.r4g_playing = False
        else:
            st.session_state.r4g_play_index = min(max_frame, int(st.session_state.r4g_play_index) + speed)
            time.sleep(0.08)
            st.rerun()
