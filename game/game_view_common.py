"""Reusable game view components for all rooms."""

from collections.abc import Sequence

import streamlit as st

from game.html_rendering import render_html
from game.theme import get_theme
from game.canvas_renderer import render_grid_canvas
from game.models import RoomTransition
from game.achievements import AchievementTracker
from game.room_transitions import render_transition_content
from game.home_page import ROOM_DEFS
from game.constants import SHOWCASE_MODE
from game.presentation import go_to_lab, go_to_showcase_room


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
        if target_mode == SHOWCASE_MODE:
            go_to_showcase_room(target_room)
        else:
            st.session_state.game_room = target_room
            st.session_state.mode = target_mode
            st.rerun()


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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Room Selection", key=f"{room_id}_return", type="primary"):
            go_to_showcase_room(None)
    with col2:
        if st.button("Open Lab Analysis", key=f"{room_id}_transition_lab"):
            go_to_lab(room_id)
    
    return True


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
