"""Room 2 — Laser Corridor game view. Full SARSA replay implementation."""

from dataclasses import replace

import streamlit as st

from game.html_rendering import render_html
from game.theme import get_theme
from game.game_view_common import (
    render_back_button,
    render_game_grid,
    render_step_info,
    render_room_transition,
    check_and_unlock_achievements,
    render_game_legend,
)

from core.types import (
    Action, SlipConfig,
)
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from agents.sarsa import extract_greedy_policy, load_model, rollout_sarsa_policy
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.constants import COMPARISON_MODE
from game.presentation import (
    final_summary_success,
    go_to_mode,
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


def _max_steps_from_meta(meta: dict, default: int = 200) -> int:
    return int(_training_config(meta).get("max_steps", default))


def _preferred_room2_model_stem() -> str | None:
    import glob
    import os

    model_dir = os.path.join("storage", "models", "room2_sarsa")
    showcase = os.path.join(model_dir, "showcase_sarsa")
    if os.path.exists(showcase + ".json") and os.path.exists(showcase + ".npz"):
        return showcase

    files = glob.glob(os.path.join(model_dir, "*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    if not files:
        return None
    return max(files, key=os.path.getmtime).replace(".json", "")


def _load_room2_game_model(filepath_stem: str) -> None:
    q_vals, meta = load_model(filepath_stem, map_grid=ROOM2_GRID)
    st.session_state.r2g_q_vals = q_vals
    st.session_state.r2g_meta = meta
    st.session_state.r2g_replay = None
    st.session_state.r2g_loaded = True
    st.session_state.r2g_model_stem = filepath_stem
    st.session_state.r2g_load_error = None
    st.session_state.r2g_autoload_disabled = False


def render_room2_game():
    # Showcase view for a trained SARSA model.  It loads Q-values from storage,
    # builds a greedy rollout, and lets the user replay the learned policy.
    theme = get_theme("room2")

    render_html(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        f'The map is unknown. The agent must learn from experience while deciding whether to risk '
        f'a short path through laser traps or take a safer route.</div>'
    )
    render_assignment_proof("room2")
    render_open_lab_button("room2", key="r2g_open_lab")

    render_back_button("r2g_back")

    for key in [
        "r2g_q_vals", "r2g_meta", "r2g_replay", "r2g_loaded",
        "r2g_model_stem", "r2g_load_error", "r2g_replay_key",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "r2g_autoload_disabled" not in st.session_state:
        st.session_state.r2g_autoload_disabled = False

    if (
        st.session_state.r2g_q_vals is None
        and st.session_state.r2g_meta is None
        and not st.session_state.r2g_autoload_disabled
    ):
        latest = _preferred_room2_model_stem()
        if latest is not None:
            try:
                _load_room2_game_model(latest)
            except ValueError as e:
                st.session_state.r2g_load_error = str(e)

    # Sidebar controls
    with st.sidebar:
        st.header("Room 2 Controls")
        
        # Model loading section
        st.markdown("**Model**")
        load_col, reset_col = st.columns(2)
        if load_col.button("Load Latest Model", key="r2g_load"):
            try:
                latest = _preferred_room2_model_stem()
                if latest:
                    _load_room2_game_model(latest)
                    st.success(f"Loaded model from {latest}")
                else:
                    st.caption("No saved Room 2 models found.")
            except Exception as e:
                st.error(f"Load failed: {e}")
            st.rerun()
        
        if reset_col.button("Reset", key="r2g_reset"):
            for key in ["r2g_q_vals", "r2g_meta", "r2g_replay", "r2g_loaded", "r2g_model_stem", "r2g_load_error", "r2g_replay_key"]:
                st.session_state[key] = None
            st.session_state.r2g_autoload_disabled = True
            st.rerun()

        st.markdown("---")
        
        # Display toggles
        st.markdown("**Display**")
        show_policy = st.checkbox("Policy Arrows", value=True, key="r2g_pol")
        show_values = st.checkbox("State Values", value=False, key="r2g_val")
        show_labels = st.checkbox("Cell Labels", value=False, key="r2g_lbl")

    q_vals = st.session_state.r2g_q_vals
    meta = st.session_state.r2g_meta
    replay = st.session_state.r2g_replay

    if q_vals is None or meta is None:
        load_error = st.session_state.get("r2g_load_error")
        if load_error:
            st.caption(f"Room 2 model auto-load skipped: {load_error}")
        else:
            st.caption("No Room 2 model is loaded.")
        return

    options = stage_options("room2", st.session_state.r2g_model_stem)
    stage_labels = [label for label, _ in options] or ["Final"]
    default_index = stage_labels.index("Final") if "Final" in stage_labels else len(stage_labels) - 1
    selected_stage = st.selectbox(
        "Policy Stage",
        stage_labels,
        index=default_index,
        key="r2g_policy_stage",
    )
    selected_stem = dict(options).get(selected_stage, st.session_state.r2g_model_stem)
    if selected_stem and selected_stem != st.session_state.r2g_model_stem:
        try:
            q_vals, meta = load_model(selected_stem, map_grid=ROOM2_GRID)
        except ValueError as e:
            st.error(f"Stage load failed: {e}")
            q_vals = st.session_state.r2g_q_vals
            meta = st.session_state.r2g_meta
            selected_stage = "Final"
            selected_stem = st.session_state.r2g_model_stem

    render_model_provenance(
        title="SARSA",
        model_stem=selected_stem,
        metadata=meta,
        evaluation_success=final_summary_success("Room 2"),
    )

    with st.container(border=True):
        st.markdown("#### SARSA Lesson")
        st.markdown(
            "SARSA is on-policy: its update uses the next action actually selected by the "
            "epsilon-greedy behavior policy. In a trap corridor, this makes exploration risk "
            "part of what the learned policy evaluates."
        )

    if st.button("📊 Compare SARSA vs Q-Learning →", key="r2g_compare_link"):
        go_to_mode(COMPARISON_MODE, view="comparison")

    # Extract the greedy policy from loaded Q-values for arrows on the grid.
    greedy_policy = extract_greedy_policy(q_vals)
    
    # Build replay if not already built.  The replay is cached in session state
    # so moving the slider/buttons does not rerun the rollout from scratch.
    replay_key = (selected_stem, _seed_from_meta(meta), _max_steps_from_meta(meta))
    if replay is None or st.session_state.r2g_replay_key != replay_key:
        seed = _seed_from_meta(meta)
        slip_config = _slip_config_from_meta(meta)
        max_steps = _max_steps_from_meta(meta)
        make_env = lambda: Room2SARSA(max_steps=max_steps, slip_config=slip_config, seed=seed)
        roll = rollout_sarsa_policy(make_env, q_vals, seed=seed)
        replay = build_replay_from_rollout(roll, "room2", stage_label=selected_stage)
        st.session_state.r2g_replay = replay
        st.session_state.r2g_replay_key = replay_key
        st.rerun()
    render_grid_stage_summary(selected_stage, replay)

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

    # HUD
    env = Room2SARSA(max_steps=_max_steps_from_meta(meta),
                     slip_config=_slip_config_from_meta(meta),
                     seed=_seed_from_meta(meta))
    cfg = _training_config(meta)
    
    status_badges = []
    if replay and replay.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif replay and not replay.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    render_html(render_hud(
        room_name="\u26a1 Room 2: The Laser Corridor",
        algorithm=f"SARSA (On-Policy TD) | \u03b1={float(cfg.get('alpha', meta.get('alpha', 0.1))):.2f} \u03b3={float(cfg.get('gamma', meta.get('gamma', 0.95))):.2f}",
        state_str=str(env.agent_position) if current_step_data else None,
        action=current_step_data.action if current_step_data else None,
        reward=current_step_data.reward if current_step_data else None,
        total_reward=replay.total_reward if replay else None,
        epsilon=current_step_data.epsilon_at_time if current_step_data else None,
        status_badges=status_badges,
        slip_info=slip_info,
    ))

    # Main grid area
    col_grid, col_info = st.columns([3, 1])

    with col_grid:
        current_pos = None
        trajectory = None
        if current_step_data:
            current_pos = current_step_data.state
            if replay:
                trajectory = [s.state for s in replay.steps[:replay.current_index + 1]]

        render_game_grid(
            env=env,
            agent_pos=current_pos,
            room_id="room2",
            policy=greedy_policy,
            values=None,  # SARSA doesn't have state values easily
            show_policy=show_policy,
            show_values=show_values,
            show_labels=show_labels,
            slip_effect=slip_effect,
            trajectory=trajectory,
            cell_size=48,
        )

    with col_info:
        render_step_info(current_step_data, replay, "room2")
        
        st.markdown("---")
        st.markdown("### Explain Action")
        q_vals_state = None
        if q_vals and current_step_data:
            q_vals_state = {a.name: q_vals.get(current_step_data.state, (0,0,0,0))[i] 
                           for i, a in enumerate(Action)}
        sel_action = current_step_data.action if current_step_data else None
        render_html(render_explain_panel(
            q_vals_state,
            selected_action=sel_action,
            algorithm="SARSA",
            explanation=get_algorithm_explanation("sarsa"),
        ))

    # Replay controls
    if replay:
        from game.episode_replay import render_replay_bar
        render_html(render_replay_bar(replay, replay_key="r2g"))

        rk = "r2g"
        cur = replay.current_index
        total = len(replay.steps)
        rb_cols = st.columns([1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
        with rb_cols[0]:
            if st.button("\u23ee", key=f"{rk}_begin", disabled=cur == 0):
                st.session_state.r2g_replay = replace(replay, current_index=0, playing=False)
                st.rerun()
        with rb_cols[1]:
            if st.button("\u23f4", key=f"{rk}_prev", disabled=cur == 0):
                st.session_state.r2g_replay = replace(replay, current_index=cur - 1, playing=False)
                st.rerun()
        with rb_cols[2]:
            btn_label = "\u23f8" if replay.playing else "\u25b6"
            if st.button(btn_label, key=f"{rk}_play"):
                st.session_state.r2g_replay = replace(replay, playing=not replay.playing)
                st.rerun()
        with rb_cols[3]:
            if st.button("\u23f5", key=f"{rk}_next", disabled=cur >= total - 1):
                st.session_state.r2g_replay = replace(replay, current_index=cur + 1, playing=False)
                st.rerun()
        with rb_cols[4]:
            if st.button("\u23ed", key=f"{rk}_end", disabled=cur >= total - 1):
                st.session_state.r2g_replay = replace(replay, current_index=total - 1, playing=False)
                st.rerun()

        with rb_cols[5]:
            st.markdown(f"Speed: {replay.speed}x")
        with rb_cols[6]:
            if st.button("0.5x", key=f"{rk}_sp05"):
                st.session_state.r2g_replay = replace(replay, speed=0.5)
                st.rerun()
        with rb_cols[7]:
            if st.button("1x", key=f"{rk}_sp1"):
                st.session_state.r2g_replay = replace(replay, speed=1.0)
                st.rerun()
        with rb_cols[8]:
            if st.button("2x", key=f"{rk}_sp2"):
                st.session_state.r2g_replay = replace(replay, speed=2.0)
                st.rerun()
        with rb_cols[9]:
            if st.button("4x", key=f"{rk}_sp4"):
                st.session_state.r2g_replay = replace(replay, speed=4.0)
                st.rerun()

    # Room transition
    achievements = check_and_unlock_achievements("room2", replay)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")
    
    render_room_transition("room2", replay, achievements)

    # Legend
    render_game_legend("room2")
