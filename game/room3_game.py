"""Room 3 — Key Vault game view. Full Q-Learning replay implementation."""

from dataclasses import replace

import streamlit as st
import numpy as np

from game.html_rendering import render_html
from game.theme import get_theme
from game.game_view_common import (
    render_back_button,
    render_parameter_sliders,
    render_game_grid,
    render_step_info,
    render_hud_panel,
    render_vi_animation,
    render_room_transition,
    check_and_unlock_achievements,
)

from core.types import (
    Action, CellType, SlipConfig, QLearningConfig, Position,
)
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from agents.q_learning import QLearningAgent, load_q_model, rollout_q_learning_policy
from agents.tabular_utils import extract_deterministic_greedy_policy
from game.canvas_renderer import render_grid_canvas
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.models import ReplayState
from game.achievements import AchievementTracker
from game.room_transitions import render_transition_content
from game.home_page import ROOM_DEFS
from game.canvas_renderer import render_policy_grid_canvas
from game.presentation import (
    final_summary_success,
    render_assignment_proof,
    render_grid_stage_summary,
    render_model_provenance,
    render_open_lab_button,
    stage_options,
)


def _training_config(meta: dict) -> dict:
    # Saved model metadata may nest the training config under this key.
    return meta.get("training_config", {})


def _slip_config_from_meta(meta: dict) -> SlipConfig:
    slip = meta.get("slip_config", {})
    return SlipConfig(
        intended_probability=float(slip.get("intended_probability", meta.get("p_int", 0.8))),
        left_probability=float(slip.get("left_probability", meta.get("p_left", 0.1))),
        right_probability=float(slip.get("right_probability", meta.get("p_right", 0.1))),
    )


def _seed_from_meta(meta: dict, default: int = 42) -> int:
    cfg = _training_config(meta)
    return int(meta.get("training_seed", cfg.get("seed", meta.get("seed", default))))


def _max_steps_from_meta(meta: dict, default: int = 300) -> int:
    return int(_training_config(meta).get("max_steps", default))


def _preferred_room3_model_stem() -> str | None:
    import glob
    import os

    model_dir = os.path.join("storage", "models", "room3_q_learning")
    showcase = os.path.join(model_dir, "showcase_ql")
    if os.path.exists(showcase + ".json") and os.path.exists(showcase + ".npz"):
        return showcase

    files = glob.glob(os.path.join(model_dir, "*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    if not files:
        return None
    return max(files, key=os.path.getmtime).replace(".json", "")


def _load_room3_game_model(filepath_stem: str) -> None:
    q_vals, meta = load_q_model(filepath_stem, map_grid=ROOM3_GRID)
    st.session_state.r3g_q_vals = q_vals
    st.session_state.r3g_meta = meta
    st.session_state.r3g_replay = None
    st.session_state.r3g_loaded = True
    st.session_state.r3g_model_stem = filepath_stem
    st.session_state.r3g_load_error = None
    st.session_state.r3g_autoload_disabled = False


def _has_key_before_current_step(replay) -> bool:
    # Used to decide which policy slice to explain/render before the current
    # replay event has been applied.
    if replay is None:
        return False
    return any(
        step.event in ("key", "key_collected")
        for step in replay.steps[:replay.current_index]
    )


def _has_key_after_current_step(replay) -> bool:
    if replay is None:
        return False
    return any(
        step.event in ("key", "key_collected")
        for step in replay.steps[:replay.current_index + 1]
    )


def render_room3_game():
    # Showcase view for a trained Q-Learning model.  The key flag matters for
    # rendering because one physical cell has two possible Q-table states.
    theme = get_theme("room3")

    render_html(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        f'The exit is locked. The agent must remember whether it has collected the key because '
        f'the same location has different meaning before and after collection.</div>'
    )
    render_assignment_proof("room3")
    render_open_lab_button("room3", key="r3g_open_lab")

    render_back_button("r3g_back")

    for key in [
        "r3g_q_vals", "r3g_meta", "r3g_replay", "r3g_loaded",
        "r3g_model_stem", "r3g_load_error", "r3g_replay_key",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "r3g_autoload_disabled" not in st.session_state:
        st.session_state.r3g_autoload_disabled = False

    if (
        st.session_state.r3g_q_vals is None
        and st.session_state.r3g_meta is None
        and not st.session_state.r3g_autoload_disabled
    ):
        latest = _preferred_room3_model_stem()
        if latest is not None:
            try:
                _load_room3_game_model(latest)
            except ValueError as e:
                st.session_state.r3g_load_error = str(e)

    # Sidebar controls
    with st.sidebar:
        st.header("Room 3 Controls")
        
        # Model loading section
        st.markdown("**Model**")
        load_col, reset_col = st.columns(2)
        if load_col.button("Load Latest Model", key="r3g_load"):
            try:
                latest = _preferred_room3_model_stem()
                if latest:
                    _load_room3_game_model(latest)
                    st.success(f"Loaded model from {latest}")
                else:
                    st.caption("No saved Room 3 models found.")
            except Exception as e:
                st.error(f"Load failed: {e}")
            st.rerun()
        
        if reset_col.button("Reset", key="r3g_reset"):
            for key in ["r3g_q_vals", "r3g_meta", "r3g_replay", "r3g_loaded", "r3g_model_stem", "r3g_load_error", "r3g_replay_key"]:
                st.session_state[key] = None
            st.session_state.r3g_autoload_disabled = True
            st.rerun()

        st.markdown("---")
        
        # Display toggles
        st.markdown("**Display**")
        show_policy = st.checkbox("Policy Arrows", value=True, key="r3g_pol")
        show_values = st.checkbox("State Values", value=False, key="r3g_val")
        show_labels = st.checkbox("Cell Labels", value=False, key="r3g_lbl")

    q_vals = st.session_state.r3g_q_vals
    meta = st.session_state.r3g_meta
    replay = st.session_state.r3g_replay

    if q_vals is None or meta is None:
        load_error = st.session_state.get("r3g_load_error")
        if load_error:
            st.caption(f"Room 3 model auto-load skipped: {load_error}")
        else:
            st.caption("No Room 3 model is loaded.")
        return

    options = stage_options("room3", st.session_state.r3g_model_stem)
    stage_labels = [label for label, _ in options] or ["Final"]
    default_index = stage_labels.index("Final") if "Final" in stage_labels else len(stage_labels) - 1
    selected_stage = st.selectbox(
        "Policy Stage",
        stage_labels,
        index=default_index,
        key="r3g_policy_stage",
    )
    selected_stem = dict(options).get(selected_stage, st.session_state.r3g_model_stem)
    if selected_stem and selected_stem != st.session_state.r3g_model_stem:
        try:
            q_vals, meta = load_q_model(selected_stem, map_grid=ROOM3_GRID)
        except ValueError as e:
            st.error(f"Stage load failed: {e}")
            q_vals = st.session_state.r3g_q_vals
            meta = st.session_state.r3g_meta
            selected_stage = "Final"
            selected_stem = st.session_state.r3g_model_stem

    render_model_provenance(
        title="Q-Learning",
        model_stem=selected_stem,
        metadata=meta,
        evaluation_success=final_summary_success("Room 3"),
    )

    with st.container(border=True):
        st.markdown("#### Markov State Lesson")
        st.markdown(
            "Position alone is not enough in this room. The same physical cell has different "
            "meaning before and after the key, so the state must include `has_key`."
        )

    # Current step data
    current_step_data = get_current_step(replay) if replay else None
    
    # Slip display
    slip_info = None
    slip_effect = False
    if current_step_data and current_step_data.slipped:
        slip_info = {
            "intended": current_step_data.action,
            "actual": current_step_data.effective_action,
        }
        slip_effect = True

    # Build environment for rendering.  The rollout itself is generated below
    # from the loaded model and matching metadata.
    env = Room3QLearning(max_steps=_max_steps_from_meta(meta),
                         slip_config=_slip_config_from_meta(meta),
                         seed=_seed_from_meta(meta))
    
    # Build replay if needed and cache it across Streamlit reruns.
    replay_key = (selected_stem, _seed_from_meta(meta), _max_steps_from_meta(meta))
    if replay is None or st.session_state.r3g_replay_key != replay_key:
        seed = _seed_from_meta(meta)
        slip_config = _slip_config_from_meta(meta)
        max_steps = _max_steps_from_meta(meta)
        make_env = lambda: Room3QLearning(max_steps=max_steps, slip_config=slip_config, seed=seed)
        roll = rollout_q_learning_policy(make_env, q_vals, seed=seed)
        replay = build_replay_from_rollout(roll, "room3", stage_label=selected_stage)
        st.session_state.r3g_replay = replay
        st.session_state.r3g_replay_key = replay_key
        st.rerun()
    render_grid_stage_summary(selected_stage, replay)

    # HUD
    status_badges = []
    if replay and replay.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif replay and not replay.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    cfg = _training_config(meta)
    has_key_before = _has_key_before_current_step(replay)
    has_key_after = _has_key_after_current_step(replay)

    render_html(render_hud(
        room_name="\U0001f511 Room 3: The Key Vault",
        algorithm=f"Q-Learning (Off-Policy TD) | \u03b1={float(cfg.get('alpha', meta.get('alpha', 0.1))):.2f} \u03b3={float(cfg.get('gamma', meta.get('gamma', 0.95))):.2f}",
        state_str=str(env.agent_position) if current_step_data else None,
        action=current_step_data.action if current_step_data else None,
        reward=current_step_data.reward if current_step_data else None,
        total_reward=replay.total_reward if replay else None,
        epsilon=current_step_data.epsilon_at_time if current_step_data else None,
        status_badges=status_badges,
        slip_info=slip_info,
        inventory="KEY" if has_key_after else None,
    ))

    # Main grid area
    col_grid, col_info = st.columns([3, 1])

    with col_grid:
        current_pos = None
        trajectory = None
        if current_step_data:
            current_pos = current_step_data.state[:2]  # (row, col) without key flag
            if replay:
                trajectory = [s.state[:2] for s in replay.steps[:replay.current_index + 1]]

        # The canvas uses the current key flag to select the matching policy slice.
        greedy_policy = extract_deterministic_greedy_policy(q_vals)
        
        render_game_grid(
            env=env,
            agent_pos=current_pos,
            room_id="room3",
            policy=greedy_policy,
            values=None,
            show_policy=show_policy,
            show_values=show_values,
            show_labels=show_labels,
            slip_effect=slip_effect,
            trajectory=trajectory,
            cell_size=48,
            has_key=has_key_before,
        )

        st.markdown("### Policy Before And After Key")
        c_before, c_after = st.columns(2)
        with c_before:
            st.caption("Policy before collecting key")
            render_html(render_policy_grid_canvas(env.grid, greedy_policy, room_id="room3", has_key=False))
        with c_after:
            st.caption("Policy after collecting key")
            render_html(render_policy_grid_canvas(env.grid, greedy_policy, room_id="room3", has_key=True))

    with col_info:
        render_step_info(current_step_data, replay, "room3")
        
        st.markdown("---")
        st.markdown("### Explain Action")
        q_vals_state = None
        if q_vals and current_step_data:
            row, col = current_step_data.state[:2]
            state_key = (row, col, has_key_before)
            q_vals_state = {a.name: q_vals.get(state_key, (0,0,0,0))[i]
                           for i, a in enumerate(Action)}
        sel_action = current_step_data.action if current_step_data else None
        render_html(render_explain_panel(
            q_vals_state,
            selected_action=sel_action,
            algorithm="Q-Learning",
            explanation=get_algorithm_explanation("q_learning"),
        ))

        st.markdown("---")
        st.markdown("### Same Cell, Different State")
        if current_step_data:
            row, col = current_step_data.state[:2]
        else:
            row, col = env.agent_position
        rows = []
        for has_key_flag in (False, True):
            state_key = (row, col, has_key_flag)
            vals = q_vals.get(state_key, (0, 0, 0, 0))
            best_idx = int(np.argmax(np.array(vals, dtype=float)))
            rows.append({
                "state": f"({row}, {col}, {has_key_flag})",
                "greedy_action": Action(best_idx).name,
                "UP": f"{vals[0]:.2f}",
                "RIGHT": f"{vals[1]:.2f}",
                "DOWN": f"{vals[2]:.2f}",
                "LEFT": f"{vals[3]:.2f}",
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    # Replay controls
    if replay:
        from game.episode_replay import render_replay_bar
        render_html(render_replay_bar(replay, replay_key="r3g"))

        rk = "r3g"
        cur = replay.current_index
        total = len(replay.steps)
        rb_cols = st.columns([1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
        with rb_cols[0]:
            if st.button("\u23ee", key=f"{rk}_begin", disabled=cur == 0):
                st.session_state.r3g_replay = replace(replay, current_index=0, playing=False)
                st.rerun()
        with rb_cols[1]:
            if st.button("\u23f4", key=f"{rk}_prev", disabled=cur == 0):
                st.session_state.r3g_replay = replace(replay, current_index=cur - 1, playing=False)
                st.rerun()
        with rb_cols[2]:
            btn_label = "\u23f8" if replay.playing else "\u25b6"
            if st.button(btn_label, key=f"{rk}_play"):
                st.session_state.r3g_replay = replace(replay, playing=not replay.playing)
                st.rerun()
        with rb_cols[3]:
            if st.button("\u23f5", key=f"{rk}_next", disabled=cur >= total - 1):
                st.session_state.r3g_replay = replace(replay, current_index=cur + 1, playing=False)
                st.rerun()
        with rb_cols[4]:
            if st.button("\u23ed", key=f"{rk}_end", disabled=cur >= total - 1):
                st.session_state.r3g_replay = replace(replay, current_index=total - 1, playing=False)
                st.rerun()

        with rb_cols[5]:
            st.markdown(f"Speed: {replay.speed}x")
        with rb_cols[6]:
            if st.button("0.5x", key=f"{rk}_sp05"):
                st.session_state.r3g_replay = replace(replay, speed=0.5)
                st.rerun()
        with rb_cols[7]:
            if st.button("1x", key=f"{rk}_sp1"):
                st.session_state.r3g_replay = replace(replay, speed=1.0)
                st.rerun()
        with rb_cols[8]:
            if st.button("2x", key=f"{rk}_sp2"):
                st.session_state.r3g_replay = replace(replay, speed=2.0)
                st.rerun()
        with rb_cols[9]:
            if st.button("4x", key=f"{rk}_sp4"):
                st.session_state.r3g_replay = replace(replay, speed=4.0)
                st.rerun()

    # Room transition
    achievements = check_and_unlock_achievements("room3", replay)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")

    if replay and replay.success and replay.current_index >= len(replay.steps) - 1:
        key_steps = [s.step_index + 1 for s in replay.steps if s.event in ("key", "key_collected")]
        exit_steps = [s.step_index + 1 for s in replay.steps if s.event == "exit"]
        with st.container(border=True):
            st.markdown("#### Key Vault Completion")
            c1, c2, c3 = st.columns(3)
            c1.metric("Key Collected At", key_steps[0] if key_steps else "N/A")
            c2.metric("Exit Reached At", exit_steps[-1] if exit_steps else replay.total_steps)
            c3.metric("Total Return", f"{replay.total_reward:.1f}")
    
    render_room_transition("room3", replay, achievements)

    # Legend
    from game.game_view_common import render_game_legend
    render_game_legend("room3")
