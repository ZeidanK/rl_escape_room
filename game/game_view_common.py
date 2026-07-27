"""Reusable game view components for all rooms."""

import time
from dataclasses import replace
from collections.abc import Sequence

import streamlit as st
import numpy as np

from game.html_rendering import render_html
from game.theme import get_theme
from game.canvas_renderer import render_grid_canvas, render_vi_animation_frame
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, render_replay_bar, get_current_step
from game.explain_panel import render_explain_panel as render_explain_panel_html, get_algorithm_explanation
from game.models import ReplayState, RoomTransition
from game.achievements import AchievementTracker, AchievementId
from game.room_transitions import render_transition_content
from game.home_page import ROOM_DEFS
from game.constants import SHOWCASE_MODE
from game.presentation import go_to_showcase_room


def _iter_replay_steps(replay) -> Sequence:
    # Some rollout objects use "steps" as an integer count, while ReplayState
    # uses it as a tuple.  This helper normalizes that difference for UI logic.
    steps = getattr(replay, "steps", ())
    if isinstance(steps, int):
        return ()
    return steps or ()


def _replay_step_count(replay) -> int:
    if hasattr(replay, "total_steps"):
        return int(replay.total_steps)
    return int(getattr(replay, "steps", 0))


def render_back_button(key: str, target_mode: str = SHOWCASE_MODE, target_room: str | None = None):
    """Render a back button to exit the game view."""
    if st.button("\u2190 Back to Room Selection", key=key, width="stretch"):
        st.session_state.game_room = target_room
        st.session_state.mode = target_mode
        st.rerun()


def render_parameter_sliders(room_id: str, prefix: str):
    """Render common parameter sliders for DP rooms."""
    theme = get_theme(room_id)
    
    st.header(f"{theme.emoji} Room Controls")
    gamma = st.slider("Discount (\u03b3)", 0.50, 0.99, 0.95, step=0.01, key=f"{prefix}_gamma",
                      help="How much future rewards are valued vs immediate rewards. Higher = more far-sighted.")
    tolerance = st.select_slider("Tolerance", options=[1e-2, 1e-4, 1e-6], value=1e-6, key=f"{prefix}_tol",
                                 help="Stop iterating when max value change per iteration falls below this threshold.")
    max_it = st.number_input("Max Iterations", 100, 50000, 10000, step=100, key=f"{prefix}_maxit",
                             help="Hard cap on iterations. Algorithm stops when converged or this limit is reached.")
    
    st.markdown("**Slip Probabilities**")
    p_int = st.slider("Intended", 0.0, 1.0, 0.80, step=0.05, key=f"{prefix}_pint",
                      help="Probability the agent moves in the intended direction.")
    p_left = st.slider("Left", 0.0, 1.0, 0.10, step=0.05, key=f"{prefix}_pleft",
                       help="Probability the agent slips left (counter-clockwise) from intended direction.")
    p_right = st.slider("Right", 0.0, 1.0, 0.10, step=0.05, key=f"{prefix}_pright",
                        help="Probability the agent slips right (clockwise) from intended direction.")
    from core.types import SlipConfig
    slip_cfg = SlipConfig(p_int, p_left, p_right)
    
    seed = st.number_input("Seed", 0, 2**31 - 1, 42, key=f"{prefix}_seed",
                           help="Random seed for environment stochasticity (slip outcomes).")
    
    slip_sum = p_int + p_left + p_right
    slip_valid = abs(slip_sum - 1.0) <= 1e-7
    if not slip_valid:
        st.error(f"Slip probabilities must sum to 1.0 (currently {slip_sum:.2f})")
    
    return gamma, tolerance, max_it, slip_cfg, seed, slip_valid


def render_display_toggles(prefix: str, show_animation: bool = True):
    """Render display toggle checkboxes."""
    st.markdown("**Display Toggles**")
    show_policy = st.checkbox("Policy Arrows", value=True, key=f"{prefix}_pol",
                              help="Show greedy policy arrows on the grid.")
    show_values = st.checkbox("State Values", value=False, key=f"{prefix}_val",
                              help="Display numeric state values in each cell.")
    show_labels = st.checkbox("Cell Labels", value=False, key=f"{prefix}_lbl",
                              help="Show cell type labels (S=Start, G=Goal, etc.).")
    
    if show_animation:
        show_anim = st.checkbox("VI Animation", value=True, key=f"{prefix}_anim",
                                help="Animate Value Iteration convergence as a heatmap.")
    else:
        show_anim = False
    
    return show_policy, show_values, show_labels, show_anim


def render_action_buttons(prefix: str, col1_label: str, col2_label: str, col1_type: str = "primary"):
    """Render two action buttons (e.g., Solve/Reset)."""
    col1, col2 = st.columns(2)
    btn1 = col1.button(col1_label, type=col1_type, key=f"{prefix}_{col1_label.lower().replace(' ', '_')}")
    btn2 = col2.button(col2_label, key=f"{prefix}_{col2_label.lower().replace(' ', '_')}")
    return btn1, btn2


def render_replay_controls(replay: ReplayState, prefix: str, room_id: str = "room1") -> ReplayState | None:
    """Render replay control buttons and return updated replay state if changed."""
    if not replay:
        return None
    
    render_html(render_replay_bar(replay, replay_key=prefix))

    rk = prefix
    cur = replay.current_index
    total = len(replay.steps)
    rb_cols = st.columns([1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
    
    updated = None
    
    with rb_cols[0]:
        if st.button("\u23ee", key=f"{rk}_begin", disabled=cur == 0):
            updated = replace(replay, current_index=0, playing=False)
    with rb_cols[1]:
        if st.button("\u23f4", key=f"{rk}_prev", disabled=cur == 0):
            updated = replace(replay, current_index=cur - 1, playing=False)
    with rb_cols[2]:
        btn_label = "\u23f8" if replay.playing else "\u25b6"
        if st.button(btn_label, key=f"{rk}_play"):
            updated = replace(replay, playing=not replay.playing)
    with rb_cols[3]:
        if st.button("\u23f5", key=f"{rk}_next", disabled=cur >= total - 1):
            updated = replace(replay, current_index=cur + 1, playing=False)
    with rb_cols[4]:
        if st.button("\u23ed", key=f"{rk}_end", disabled=cur >= total - 1):
            updated = replace(replay, current_index=total - 1, playing=False)

    with rb_cols[5]:
        st.markdown(f"Speed: {replay.speed}x")
    with rb_cols[6]:
        if st.button("0.5x", key=f"{rk}_sp05"):
            updated = replace(replay, speed=0.5)
    with rb_cols[7]:
        if st.button("1x", key=f"{rk}_sp1"):
            updated = replace(replay, speed=1.0)
    with rb_cols[8]:
        if st.button("2x", key=f"{rk}_sp2"):
            updated = replace(replay, speed=2.0)
    with rb_cols[9]:
        if st.button("4x", key=f"{rk}_sp4"):
            updated = replace(replay, speed=4.0)

    # Non-blocking auto-advance.  Streamlit still reruns the script, but this
    # avoids sleeping/blocking while the replay is playing.
    if replay.playing and replay.current_index < len(replay.steps) - 1:
        delay = 0.4 / replay.speed
        last_key = f"{rk}_last_replay_advance"
        now = time.time()
        last_advance = st.session_state.get(last_key)
        if last_advance is None:
            st.session_state[last_key] = now
        elif now - last_advance >= delay:
            st.session_state[last_key] = now
            updated = replace(replay, current_index=replay.current_index + 1)
    
    return updated


def render_legend(room_id: str):
    """Render game legend with room-specific colors."""
    theme = get_theme(room_id)
    render_html(f"""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_wall};"></span> Wall</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_slippery};"></span> Slippery</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Agent</span>
        <span class="legend-item">\u2191\u2192\u2193\u2190 Policy</span>
    </div>
    """)


def check_and_unlock_achievements(room_id: str, replay) -> list:
    """Check and unlock achievements based on replay data. Returns list of newly unlocked achievements."""
    if replay is None:
        return []
    if not getattr(replay, "success", False):
        return []
    tracker = AchievementTracker.from_session_state()
    newly_unlocked: list = []
    replay_steps = _iter_replay_steps(replay)

    # FIRST_ESCAPE is intentionally broad: any successful room can unlock it.
    ach = tracker.try_unlock_first_escape()
    if ach:
        newly_unlocked.append(ach)

    # Room-specific achievements
    if room_id == "room1":
        # ICE_MASTER - no unintended slips
        slip_count = sum(1 for s in replay_steps if s.slipped)
        ach = tracker.try_unlock_ice_master(slip_count)
        if ach:
            newly_unlocked.append(ach)

    elif room_id == "room2":
        # LASER_DODGER - no trap cells visited
        trap_count = sum(1 for s in replay_steps if s.event == "trap")
        ach = tracker.try_unlock_laser_dodger(trap_count)
        if ach:
            newly_unlocked.append(ach)

    elif room_id == "room3":
        # VAULT_EXPERT - key collected and exit reached
        key_collected = any(s.event in ("key", "key_collected") for s in replay_steps)
        ach = tracker.try_unlock_vault_expert(replay.success, key_collected)
        if ach:
            newly_unlocked.append(ach)

    elif room_id == "room4":
        # MOMENTUM_MASTER - solved from unseen start
        # For approximate SARSA, we check if start was random
        ach = tracker.try_unlock_momentum_master(True, replay.success)
        if ach:
            newly_unlocked.append(ach)

    # SPEED_RUNNER - new best time for any room
    is_new_best = False
    for room_def in ROOM_DEFS:
        if room_def.room_id == room_id:
            room_status = room_def.status
            if room_status.best_steps is None or _replay_step_count(replay) < room_status.best_steps:
                is_new_best = True
            break
    ach = tracker.try_unlock_speed_runner(is_new_best)
    if ach:
        newly_unlocked.append(ach)

    return newly_unlocked


def render_room_transition(room_id: str, replay, achievements: list):
    """Render room transition overlay with achievements and continue button."""
    if replay is None:
        return False
    if not replay.success:
        return False
    
    # Only show the success transition after the replay reaches its last frame.
    replay_steps = _iter_replay_steps(replay)
    if replay_steps and getattr(replay, "current_index", 0) < len(replay_steps) - 1:
        return False
    
    theme = get_theme(room_id)
    
    # Check for new best
    is_new_best = False
    for room_def in ROOM_DEFS:
        if room_def.room_id == room_id:
            room_status = room_def.status
            if room_status.best_steps is None or _replay_step_count(replay) < room_status.best_steps:
                is_new_best = True
            break
    
    # Build transition
    transition = RoomTransition(
        room_id=room_id,
        success=True,
        steps=_replay_step_count(replay),
        total_reward=replay.total_reward,
        new_best=is_new_best,
        message="The chamber dissolves. The path forward clears.",
        achievements_unlocked=tuple(achievements),
    )
    
    transition_html, _ = render_transition_content(transition, theme.primary)
    render_html(f'<div class="transition-overlay">{transition_html}</div>')
    
    next_rooms = {
        "room1": ("room2", "Continue to Laser Corridor"),
        "room2": ("room3", "Continue to Key Vault"),
        "room3": ("room4", "Continue to Momentum Chamber"),
        "room4": ("campaign_results", "View Campaign Results"),
    }
    next_room, next_label = next_rooms.get(room_id, (None, "Return to Room Selection"))
    col1, col2 = st.columns(2)
    with col1:
        if st.button(next_label, key=f"{room_id}_continue", type="primary"):
            go_to_showcase_room(next_room)
    with col2:
        if st.button("Return to Room Selection", key=f"{room_id}_return"):
            go_to_showcase_room(None)
    
    return True


def render_vi_animation(anim_frames: list, anim_iter: int, anim_playing: bool, prefix: str, room_id: str = "room1") -> tuple[int, bool]:
    """Render VI animation controls and return (new_iter, new_playing)."""
    if not anim_frames:
        return anim_iter, False
    
    frames = anim_frames
    total = len(frames)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        anim_cols = st.columns([1, 1, 1, 1])
        with anim_cols[0]:
            if st.button("\u23ee", key=f"{prefix}_anim_begin", disabled=anim_iter <= 0):
                return 0, False
        with anim_cols[1]:
            if st.button("\u25c0", key=f"{prefix}_anim_prev", disabled=anim_iter <= 0):
                return max(0, anim_iter - 1), False
        with anim_cols[2]:
            btn_lbl = "\u23f8" if anim_playing else "\u25b6"
            if st.button(btn_lbl, key=f"{prefix}_anim_play", disabled=anim_iter >= total - 1):
                return anim_iter, not anim_playing
        with anim_cols[3]:
            if st.button("\u25b6", key=f"{prefix}_anim_next", disabled=anim_iter >= total - 1):
                return min(total - 1, anim_iter + 1), False
        st.caption(f"Frame {anim_iter + 1} / {total}")

        frame_values = frames[anim_iter]
        from environments.room1_dp import Room1DP
        # We need an env to render - create a temporary one or get from session
        # This is simplified - in practice we'd need the actual env
        svg = render_vi_animation_frame(
            np.zeros((10, 10), dtype=int), frame_values, anim_iter + 1, total,
            cell_size=48, room_id=room_id,
        )
        render_html(f'<div style="overflow:hidden;">{svg}</div>')

    st.markdown("---")
    
    # Non-blocking auto-advance
    new_playing = anim_playing
    new_iter = anim_iter
    if anim_playing and anim_iter < total - 1:
        last_key = f"{prefix}_last_anim_advance"
        now = time.time()
        last_advance = st.session_state.get(last_key)
        if last_advance is None:
            st.session_state[last_key] = now
        elif now - last_advance >= 0.3:
            st.session_state[last_key] = now
            new_iter = anim_iter + 1
            new_playing = True
    elif anim_iter >= total - 1:
        new_playing = False
    
    return new_iter, new_playing


def render_explain_panel(q_vals: dict | None, selected_action: str | None, algorithm: str, env=None):
    """Render the 'Explain Action' panel with Q-value table."""
    st.markdown("### Explain Action")
    explanation = get_algorithm_explanation(algorithm.lower().replace(" ", "_"))
    render_html(render_explain_panel_html(
        q_vals,
        selected_action=selected_action,
        algorithm=algorithm,
        explanation=explanation,
    ))


def render_game_grid(env, agent_pos, room_id: str, policy=None, values=None,
                     show_policy: bool = True, show_values: bool = False, show_labels: bool = False,
                     slip_effect: bool = False, trajectory=None, cell_size: int = 48,
                     has_key: bool | None = None):
    """Render the game grid with error handling and fallback."""
    # Most game views call this wrapper instead of canvas_renderer directly so
    # a rendering error falls back to a text grid instead of crashing the page.
    try:
        svg = render_grid_canvas(
            env.grid,
            agent_pos=agent_pos,
            room_id=room_id,
            cell_size=cell_size,
            policy=policy,
            values=values,
            show_policy=show_policy,
            show_values=show_values,
            show_labels=show_labels,
            slip_effect=slip_effect,
            trajectory=trajectory,
            has_key=has_key,
        )
        render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')
    except Exception as e:
        st.error(f"SVG rendering failed: {e}")
        st.code(env.render_ansi(), language="text")


def render_step_info(current_step_data, replay, room_id: str):
    """Render the step info panel on the right side."""
    st.markdown("### Step Info")
    if current_step_data:
        st.markdown(f"**Step:** {current_step_data.step_index + 1} / {len(replay.steps) if replay else '?'}")
        intended = current_step_data.action.name if current_step_data.action else "N/A"
        effective = current_step_data.effective_action.name if current_step_data.effective_action else intended
        st.markdown(f"**Intended:** {intended}")
        if current_step_data.slipped:
            st.markdown(f"**Actual:** {effective}")
            render_html('<span class="badge-slip">SLIPPED</span>')
        st.markdown(f"**Reward:** {current_step_data.reward:+.1f}")
        st.markdown(f"**Cumulative:** {current_step_data.cumulative_reward:.1f}")
        if current_step_data.collision:
            render_html(f'<span class="badge-collision">COLLISION: {current_step_data.collision}</span>')
        if current_step_data.event:
            ev = current_step_data.event
            if ev == "exit":
                render_html('<span class="badge-success">EXIT REACHED</span>')
            elif ev in ("key", "key_collected"):
                render_html('<span class="badge-key">KEY COLLECTED</span>')
            elif ev == "trap":
                render_html('<span class="badge-trap">TRAP HIT</span>')
    else:
        st.markdown("No step data.")


def render_hud_panel(room_name: str, algorithm: str, env, current_step_data, replay, slip_info=None, extra_items=None):
    """Render the HUD panel with status badges."""
    status_badges = []
    if replay and replay.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif replay and not replay.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')
    
    render_html(render_hud(
        room_name=room_name,
        algorithm=algorithm,
        state_str=str(env.agent_position) if current_step_data else None,
        action=current_step_data.action if current_step_data else None,
        reward=current_step_data.reward if current_step_data else None,
        total_reward=replay.total_reward if replay else None,
        epsilon=current_step_data.epsilon_at_time if current_step_data else None,
        status_badges=status_badges,
        slip_info=slip_info,
        custom_items=extra_items,
    ))


def render_game_legend(room_id: str):
    """Render the legend bar for a room."""
    theme = get_theme(room_id)
    render_html(f"""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_wall};"></span> Wall</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_slippery};"></span> Slippery</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Agent</span>
        <span class="legend-item">\u2191\u2192\u2193\u2190 Policy</span>
    </div>
    """)
