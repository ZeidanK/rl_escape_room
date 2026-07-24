"""Room 3 — Key Vault game view. Full Q-Learning replay implementation."""

import streamlit as st
import numpy as np

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
from game.canvas_renderer import render_grid_canvas
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.models import ReplayState
from game.achievements import AchievementTracker
from game.room_transitions import render_transition_content
from game.home_page import ROOM_DEFS


def render_room3_game():
    theme = get_theme("room3")

    st.markdown(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        f'The exit is locked. The agent must remember whether it has collected the key because '
        f'the same location has different meaning before and after collection.</div>',
        unsafe_allow_html=True,
    )

    render_back_button("r3g_back")

    # Sidebar controls
    with st.sidebar:
        st.header("Room 3 Controls")
        
        # Model loading section
        st.markdown("**Model**")
        load_col, reset_col = st.columns(2)
        if load_col.button("Load Latest Model", key="r3g_load"):
            try:
                import glob, os
                model_dir = os.path.join("storage", "models", "room3_qlearning")
                pattern = os.path.join(model_dir, "*.json")
                files = glob.glob(pattern)
                if files:
                    latest = max(files).replace(".json", "")
                    q_vals, meta = load_q_model(latest, map_grid=ROOM3_GRID)
                    st.session_state.r3g_q_vals = q_vals
                    st.session_state.r3g_meta = meta
                    st.session_state.r3g_loaded = True
                    st.success(f"Loaded model from {latest}")
                else:
                    st.info("No saved models found. Train in Learning Laboratory first.")
            except Exception as e:
                st.error(f"Load failed: {e}")
            st.rerun()
        
        if reset_col.button("Reset", key="r3g_reset"):
            for key in ["r3g_q_vals", "r3g_meta", "r3g_replay", "r3g_loaded"]:
                st.session_state[key] = None
            st.rerun()

        st.markdown("---")
        
        # Display toggles
        st.markdown("**Display**")
        show_policy = st.checkbox("Policy Arrows", value=True, key="r3g_pol")
        show_values = st.checkbox("State Values", value=False, key="r3g_val")
        show_labels = st.checkbox("Cell Labels", value=False, key="r3g_lbl")

    # Initialize session state
    for key in ["r3g_q_vals", "r3g_meta", "r3g_replay", "r3g_loaded"]:
        if key not in st.session_state:
            st.session_state[key] = None

    q_vals = st.session_state.r3g_q_vals
    meta = st.session_state.r3g_meta
    replay = st.session_state.r3g_replay

    if q_vals is None or meta is None:
        st.info("Press **Load Latest Model** to view a trained policy.")
        return

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

    # Build environment for rendering
    env = Room3QLearning(max_steps=300,
                         slip_config=SlipConfig(meta.get("p_int", 0.8), 
                                                meta.get("p_left", 0.1), 
                                                meta.get("p_right", 0.1)),
                         seed=meta.get("seed", 42))
    
    # Build replay if needed
    if replay is None:
        roll = rollout_q_learning_policy(env, q_vals, seed=meta.get("seed", 42))
        replay = build_replay_from_rollout(roll, "room3", stage_label="Final")
        st.session_state.r3g_replay = replay
        st.rerun()

    # HUD
    status_badges = []
    if replay and replay.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif replay and not replay.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    st.markdown(render_hud(
        room_name="\U0001f511 Room 3: The Key Vault",
        algorithm=f"Q-Learning (Off-Policy TD) | \u03b1={meta.get('alpha', 0.1):.2f} \u03b3={meta.get('gamma', 0.95):.2f}",
        state_str=str(env.agent_position) if current_step_data else None,
        action=current_step_data.action if current_step_data else None,
        reward=current_step_data.reward if current_step_data else None,
        total_reward=replay.total_reward if replay else None,
        epsilon=current_step_data.epsilon_at_time if current_step_data else None,
        status_badges=status_badges,
        slip_info=slip_info,
        inventory="KEY" if current_step_data and current_step_data.event == "key_collected" else None,
    ), unsafe_allow_html=True)

    # Main grid area
    col_grid, col_info = st.columns([3, 1])

    with col_grid:
        current_pos = None
        trajectory = None
        if current_step_data:
            current_pos = current_step_data.state[:2]  # (row, col) without key flag
            if replay:
                trajectory = [s.state[:2] for s in replay.steps[:replay.current_index + 1]]

        # For Room 3, we need to show policy based on key state
        # The policy is different for has_key=True vs False
        # We'll show the greedy policy for the current key state
        from agents.q_learning import extract_greedy_policy_q
        greedy_policy = extract_greedy_policy_q(q_vals)
        
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
            has_key=current_step_data is not None and current_step_data.event == "key_collected",
        )

    with col_info:
        render_step_info(current_step_data, replay, "room3")
        
        st.markdown("---")
        st.markdown("### Explain Action")
        q_vals_state = None
        if q_vals and current_step_data:
            state_key = current_step_data.state
            if isinstance(state_key, tuple) and len(state_key) == 3:
                q_vals_state = {a.name: q_vals.get(state_key, (0,0,0,0))[i] 
                               for i, a in enumerate(Action)}
        sel_action = current_step_data.action if current_step_data else None
        st.markdown(render_explain_panel(
            q_vals_state,
            selected_action=sel_action,
            algorithm="Q-Learning",
            explanation=get_algorithm_explanation("q_learning"),
        ), unsafe_allow_html=True)

    # Replay controls
    if replay:
        from game.episode_replay import render_replay_bar
        st.markdown(render_replay_bar(replay, replay_key="r3g"), unsafe_allow_html=True)

        rk = "r3g"
        cur = replay.current_index
        total = len(replay.steps)
        rb_cols = st.columns([1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
        with rb_cols[0]:
            if st.button("\u23ee", key=f"{rk}_begin", disabled=cur == 0):
                st.session_state.r3g_replay = replay._replace(current_index=0, playing=False)
                st.rerun()
        with rb_cols[1]:
            if st.button("\u23f4", key=f"{rk}_prev", disabled=cur == 0):
                st.session_state.r3g_replay = replay._replace(current_index=cur - 1, playing=False)
                st.rerun()
        with rb_cols[2]:
            btn_label = "\u23f8" if replay.playing else "\u25b6"
            if st.button(btn_label, key=f"{rk}_play"):
                st.session_state.r3g_replay = replay._replace(playing=not replay.playing)
                st.rerun()
        with rb_cols[3]:
            if st.button("\u23f5", key=f"{rk}_next", disabled=cur >= total - 1):
                st.session_state.r3g_replay = replay._replace(current_index=cur + 1, playing=False)
                st.rerun()
        with rb_cols[4]:
            if st.button("\u23ed", key=f"{rk}_end", disabled=cur >= total - 1):
                st.session_state.r3g_replay = replay._replace(current_index=total - 1, playing=False)
                st.rerun()

        with rb_cols[5]:
            st.markdown(f"Speed: {replay.speed}x")
        with rb_cols[6]:
            if st.button("0.5x", key=f"{rk}_sp05"):
                st.session_state.r3g_replay = replay._replace(speed=0.5)
                st.rerun()
        with rb_cols[7]:
            if st.button("1x", key=f"{rk}_sp1"):
                st.session_state.r3g_replay = replay._replace(speed=1.0)
                st.rerun()
        with rb_cols[8]:
            if st.button("2x", key=f"{rk}_sp2"):
                st.session_state.r3g_replay = replay._replace(speed=2.0)
                st.rerun()
        with rb_cols[9]:
            if st.button("4x", key=f"{rk}_sp4"):
                st.session_state.r3g_replay = replay._replace(speed=4.0)
                st.rerun()

    # Room transition
    achievements = check_and_unlock_achievements("room3", replay)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")
    
    render_room_transition("room3", replay, achievements)

    # Legend
    from game.game_view_common import render_game_legend
    render_game_legend("room3")