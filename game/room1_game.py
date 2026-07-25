"""Room 1 — Frozen Maze game view. Full vertical slice."""

import time
from dataclasses import replace

import streamlit as st
import numpy as np

from game.html_rendering import render_html
from game.theme import get_theme

from core.types import (
    Action, CellType, SlipConfig, ValueIterationConfig, Position,
)
from environments.room1_dp import Room1DP
from agents.dynamic_programming import ValueIterationAgent, rollout_policy
from game.theme import get_theme
from game.canvas_renderer import render_grid_canvas, render_vi_animation_frame
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, render_replay_bar, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.models import ReplayState, RoomTransition
from game.home_page import render_home_page, ROOM_DEFS
from game.achievements import AchievementTracker, AchievementId
from game.room_transitions import render_transition_content


def _compute_vi_frames(agent: ValueIterationAgent, max_frames: int = 20) -> list[np.ndarray]:
    # Re-runs the same Bellman sweeps used by Value Iteration, but records a
    # small number of value matrices for the convergence animation.
    env = agent.env
    states = list(env.states)
    actions = env.actions
    goal_states = {s for s in states if env.is_terminal_state(s)}
    config = agent.config
    rows, cols = env.grid_shape

    values: dict[Position, float] = {s: 0.0 for s in states}
    frames: list[np.ndarray] = []
    record_every = max(1, config.max_iterations // max_frames)

    for iteration in range(config.max_iterations):
        new_values: dict[Position, float] = {}
        max_delta = 0.0
        for s in states:
            if s in goal_states:
                new_values[s] = 0.0
                continue
            best_value: float | None = None
            for a in actions:
                q = agent.calculate_action_value(s, a, values)
                if best_value is None or q > best_value:
                    best_value = q
            new_values[s] = best_value if best_value is not None else 0.0
            delta = abs(new_values[s] - values[s])
            if delta > max_delta:
                max_delta = delta
        values = new_values

        if iteration % record_every == 0 or iteration == config.max_iterations - 1 or max_delta < config.tolerance:
            mat = np.full((rows, cols), np.nan)
            for r in range(rows):
                for c in range(cols):
                    pos = (r, c)
                    if CellType(int(env.grid[r, c])) != CellType.WALL:
                        mat[r, c] = values.get(pos, 0.0)
            frames.append(mat.copy())

        if max_delta < config.tolerance:
            break

    return frames


def _extract_q_from_values(values, env) -> dict[str, float]:
    # Local action values at the agent's current cell, used by the explanation
    # panel to show why the policy prefers one action.
    agent_r, agent_c = env.agent_position
    state = (agent_r, agent_c)
    q_vals = {}
    for action in Action:
        outcomes = env.get_transition_distribution(state, action)
        q = 0.0
        for outcome in outcomes:
            q += outcome.probability * (outcome.reward + 0.95 * values.get(outcome.next_state, 0))
        q_vals[action.name] = q
    return q_vals


def render_room1_game():
    # Full vertical slice for Room 1: solve DP, roll out the policy, animate
    # value convergence, render the grid, and update achievements.
    theme = get_theme("room1")

    render_html(
        f'<div class="narrative-box">'
        f'The agent wakes inside a frozen chamber. The floor is unstable, and intended movements '
        f'may cause sideways slips. Because the full model is known, it calculates the optimal '
        f'escape policy before moving.</div>'
    )

    with st.sidebar:
        if st.button("\u2190 Back to Room Selection", key="r1g_back", use_container_width=True):
            st.session_state.game_room = None
            st.session_state.mode = "\U0001f3ae Escape Room Showcase"
            st.rerun()

        st.header("Room 1 Controls")
        gamma = st.slider("Discount (\u03b3)", 0.50, 0.99, 0.95, step=0.01, key="r1g_gamma",
                          help="How much future rewards are valued vs immediate rewards. Higher = more far-sighted.")
        tolerance = st.select_slider("Tolerance", options=[1e-2, 1e-4, 1e-6], value=1e-6, key="r1g_tol",
                                     help="Stop iterating when max value change per iteration falls below this threshold.")
        max_it = st.number_input("Max Iterations", 100, 50000, 10000, step=100, key="r1g_maxit",
                                 help="Hard cap on iterations. Value Iteration stops when converged or this limit is reached.")
        p_int = st.slider("Intended", 0.0, 1.0, 0.80, step=0.05, key="r1g_pint",
                          help="Probability the agent moves in the intended direction.")
        p_left = st.slider("Left", 0.0, 1.0, 0.10, step=0.05, key="r1g_pleft",
                           help="Probability the agent slips left (counter-clockwise) from intended direction.")
        p_right = st.slider("Right", 0.0, 1.0, 0.10, step=0.05, key="r1g_pright",
                            help="Probability the agent slips right (clockwise) from intended direction.")
        slip_cfg = SlipConfig(p_int, p_left, p_right)
        seed = st.number_input("Seed", 0, 2**31 - 1, 42, key="r1g_seed",
                               help="Random seed for environment stochasticity (slip outcomes).")

        st.markdown("---")
        col1, col2 = st.columns(2)
        solve_btn = col1.button("Solve Maze", type="primary", key="r1g_solve")
        reset_btn = col2.button("Reset", key="r1g_reset")

        st.markdown("**Display Toggles**")
        show_policy = st.checkbox("Policy Arrows", value=True, key="r1g_pol",
                                  help="Show greedy policy arrows on the grid.")
        show_values = st.checkbox("State Values", value=False, key="r1g_val",
                                  help="Display numeric state values in each cell.")
        show_labels = st.checkbox("Cell Labels", value=False, key="r1g_lbl",
                                  help="Show cell type labels (S=Start, G=Goal, etc.).")
        show_animation = st.checkbox("VI Animation", value=True, key="r1g_anim",
                                     help="Animate Value Iteration convergence as a heatmap.")

    for key in ["r1g_env", "r1g_vi_result", "r1g_replay", "r1g_anim_frames",
                 "r1g_anim_iter", "r1g_last_action", "r1g_slip_effect", "r1g_q_explain",
                 "r1g_anim_playing", "r1g_last_replay_advance", "r1g_last_anim_advance"]:
        # Room-specific session keys keep this game view independent from the
        # Learning Laboratory state in app.py.
        if key not in st.session_state:
            st.session_state[key] = None

    # Non-blocking auto-advance replay
    replay = st.session_state.r1g_replay
    if replay and replay.playing and replay.current_index < len(replay.steps) - 1:
        now = time.time()
        delay = 0.4 / replay.speed
        last_advance = st.session_state.r1g_last_replay_advance
        if last_advance is None:
            st.session_state.r1g_last_replay_advance = now
        elif now - last_advance >= delay:
            st.session_state.r1g_last_replay_advance = now
            st.session_state.r1g_replay = replace(replay, current_index=replay.current_index + 1)
            st.rerun()

    # Non-blocking auto-advance VI animation
    if st.session_state.get("r1g_anim_playing") and st.session_state.r1g_anim_frames is not None:
        frames = st.session_state.r1g_anim_frames
        anim_iter = st.session_state.r1g_anim_iter
        if anim_iter < len(frames) - 1:
            now = time.time()
            last_advance = st.session_state.r1g_last_anim_advance
            if last_advance is None:
                st.session_state.r1g_last_anim_advance = now
            elif now - last_advance >= 0.3:
                st.session_state.r1g_last_anim_advance = now
                st.session_state.r1g_anim_iter = anim_iter + 1
                st.rerun()
        else:
            st.session_state.r1g_anim_playing = False

    # Solve — only on explicit button click
    if solve_btn:
        with st.spinner("Running Value Iteration..."):
            env = Room1DP(slip_config=slip_cfg, max_steps=200, seed=seed)
            config = ValueIterationConfig(gamma=gamma, tolerance=tolerance, max_iterations=max_it)
            agent = ValueIterationAgent(env, config)
            vi_result = agent.solve()

            st.session_state.r1g_env = env
            st.session_state.r1g_vi_result = vi_result

            roll = rollout_policy(env, vi_result.policy, seed=seed)
            replay = build_replay_from_rollout(roll, "room1", stage_label="Final")
            st.session_state.r1g_replay = replay

            frames = _compute_vi_frames(agent)
            st.session_state.r1g_anim_frames = frames
            st.session_state.r1g_anim_iter = 0
            st.session_state.r1g_anim_playing = False

            st.rerun()

    if reset_btn:
        for key in ["r1g_env", "r1g_vi_result", "r1g_replay", "r1g_anim_frames",
                     "r1g_q_explain", "r1g_last_action"]:
            st.session_state[key] = None
        st.rerun()

    vi_result = st.session_state.r1g_vi_result
    env = st.session_state.r1g_env

    if vi_result is None or env is None:
        st.info("Press **Solve Maze** to compute the optimal escape policy.")
        return

    # VI Animation
    if show_animation and st.session_state.r1g_anim_frames is not None:
        frames = st.session_state.r1g_anim_frames
        total = len(frames)
        anim_iter = st.session_state.r1g_anim_iter

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            anim_cols = st.columns([1, 1, 1, 1])
            with anim_cols[0]:
                if st.button("\u23ee", key="r1g_anim_begin", disabled=anim_iter <= 0):
                    st.session_state.r1g_anim_iter = 0
                    st.session_state.r1g_anim_playing = False
                    st.rerun()
            with anim_cols[1]:
                if st.button("\u25c0", key="r1g_anim_prev", disabled=anim_iter <= 0):
                    st.session_state.r1g_anim_iter = max(0, anim_iter - 1)
                    st.session_state.r1g_anim_playing = False
                    st.rerun()
            with anim_cols[2]:
                anim_playing = st.session_state.get("r1g_anim_playing", False)
                btn_lbl = "\u23f8" if anim_playing else "\u25b6"
                if st.button(btn_lbl, key="r1g_anim_play", disabled=anim_iter >= total - 1):
                    st.session_state.r1g_anim_playing = not anim_playing
                    st.rerun()
            with anim_cols[3]:
                if st.button("\u25b6", key="r1g_anim_next", disabled=anim_iter >= total - 1):
                    st.session_state.r1g_anim_iter = min(total - 1, anim_iter + 1)
                    st.session_state.r1g_anim_playing = False
                    st.rerun()
            st.caption(f"Frame {anim_iter + 1} / {total}")

            frame_values = frames[anim_iter]
            try:
                svg = render_vi_animation_frame(
                    env.grid, frame_values, anim_iter + 1, total,
                    cell_size=48, room_id="room1",
                )
                render_html(f'<div style="overflow:hidden;">{svg}</div>')
            except Exception as e:
                st.error(f"Failed to render animation frame: {e}")
                st.code(str(frame_values))

        st.markdown("---")

    replay = st.session_state.r1g_replay
    current_step_data = get_current_step(replay) if replay else None

    # Determine slip display
    slip_info = None
    slip_effect = False
    if current_step_data and current_step_data.slipped:
        slip_info = {
            "intended": current_step_data.action,
            "actual": current_step_data.effective_action,
        }
        slip_effect = True

    # HUD
    status_badges = []
    if replay and replay.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif replay and not replay.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    render_html(render_hud(
        room_name="\u2744\ufe0f Room 1: The Frozen Maze",
        algorithm=f"Value Iteration (DP) | Converged in {vi_result.iterations} iters",
        state_str=str(env.agent_position) if current_step_data else None,
        action=current_step_data.action if current_step_data else None,
        reward=current_step_data.reward if current_step_data else None,
        total_reward=replay.total_reward if replay else None,
        epsilon=current_step_data.epsilon_at_time if current_step_data else None,
        status_badges=status_badges,
        slip_info=slip_info,
    ))

    col_grid, col_info = st.columns([3, 1])

    with col_grid:
        policy = vi_result.policy
        values = vi_result.values

        current_pos = None
        trajectory = None
        if current_step_data:
            current_pos = current_step_data.state
            if replay:
                trajectory = [s.state for s in replay.steps[:replay.current_index + 1]]

        # Try SVG rendering, fall back to text grid on error
        try:
            svg = render_grid_canvas(
                env.grid,
                agent_pos=current_pos,
                room_id="room1",
                cell_size=48,
                policy=policy,
                values=values,
                show_policy=show_policy,
                show_values=show_values,
                show_labels=show_labels,
                slip_effect=slip_effect,
                trajectory=trajectory,
            )
            render_html(f'<div class="grid-container" style="overflow:hidden;">{svg}</div>')
        except Exception as e:
            st.error(f"SVG rendering failed: {e}")
            st.code(env.render_ansi(), language="text")

    with col_info:
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
        else:
            st.markdown("No step data.")

        st.markdown("---")
        st.markdown("### Explain Action")
        q_vals = _extract_q_from_values(values, env) if values else None
        sel_action = current_step_data.action if current_step_data else None
        explanation = get_algorithm_explanation("vi")
        render_html(render_explain_panel(
            q_vals,
            selected_action=sel_action,
            algorithm="Value Iteration",
            explanation=explanation,
        ))

    # Replay controls — real st.button widgets (not JS onclick)
    if replay:
        render_html(render_replay_bar(replay, replay_key="r1g"))

        rk = "r1g"
        cur = replay.current_index
        total = len(replay.steps)
        rb_cols = st.columns([1, 1, 1, 1, 1, 2, 1, 1, 1, 1])
        with rb_cols[0]:
            if st.button("\u23ee", key=f"{rk}_begin", disabled=cur == 0):
                st.session_state.r1g_replay = replace(replay, current_index=0, playing=False)
                st.rerun()
        with rb_cols[1]:
            if st.button("\u23f4", key=f"{rk}_prev", disabled=cur == 0):
                st.session_state.r1g_replay = replace(replay, current_index=cur - 1, playing=False)
                st.rerun()
        with rb_cols[2]:
            btn_label = "\u23f8" if replay.playing else "\u25b6"
            if st.button(btn_label, key=f"{rk}_play"):
                st.session_state.r1g_replay = replace(replay, playing=not replay.playing)
                st.rerun()
        with rb_cols[3]:
            if st.button("\u23f5", key=f"{rk}_next", disabled=cur >= total - 1):
                st.session_state.r1g_replay = replace(replay, current_index=cur + 1, playing=False)
                st.rerun()
        with rb_cols[4]:
            if st.button("\u23ed", key=f"{rk}_end", disabled=cur >= total - 1):
                st.session_state.r1g_replay = replace(replay, current_index=total - 1, playing=False)
                st.rerun()

        with rb_cols[5]:
            st.markdown(f"Speed: {replay.speed}x")
        with rb_cols[6]:
            if st.button("0.5x", key=f"{rk}_sp05"):
                st.session_state.r1g_replay = replace(replay, speed=0.5)
                st.rerun()
        with rb_cols[7]:
            if st.button("1x", key=f"{rk}_sp1"):
                st.session_state.r1g_replay = replace(replay, speed=1.0)
                st.rerun()
        with rb_cols[8]:
            if st.button("2x", key=f"{rk}_sp2"):
                st.session_state.r1g_replay = replace(replay, speed=2.0)
                st.rerun()
        with rb_cols[9]:
            if st.button("4x", key=f"{rk}_sp4"):
                st.session_state.r1g_replay = replace(replay, speed=4.0)
                st.rerun()

    # Room Transition Overlay — show when replay completes successfully
    if replay and replay.success and replay.current_index >= len(replay.steps) - 1:
        # Unlock achievements based on replay data
        tracker = AchievementTracker.from_session_state()
        newly_unlocked: list = []

        # FIRST_ESCAPE
        ach = tracker.try_unlock_first_escape()
        if ach:
            newly_unlocked.append(ach)

        # ICE_MASTER - no unintended slips
        slip_count = sum(1 for s in replay.steps if s.slipped)
        ach = tracker.try_unlock_ice_master(slip_count)
        if ach:
            newly_unlocked.append(ach)

        # SPEED_RUNNER - new best time
        # Check if this is a new best for room1
        room_status = None
        for room_def in ROOM_DEFS:
            if room_def.room_id == "room1":
                room_status = room_def.status
                break
        is_new_best = room_status is not None and (
            room_status.best_steps is None or replay.total_steps < room_status.best_steps
        )
        ach = tracker.try_unlock_speed_runner(is_new_best)
        if ach:
            newly_unlocked.append(ach)

        # Show achievement toasts using Streamlit's native toast
        for ach in newly_unlocked:
            st.toast(f"{ach.emoji} {ach.name}: {ach.description}")

        # Build and render transition
        transition = RoomTransition(
            room_id="room1",
            success=True,
            steps=replay.total_steps,
            total_reward=replay.total_reward,
            new_best=is_new_best,
            message="The frozen maze shatters behind you. The path forward clears.",
            achievements_unlocked=tuple(newly_unlocked),
        )
        theme = get_theme("room1")
        transition_html, _ = render_transition_content(transition, theme.primary)
        render_html(f'<div class="transition-overlay">{transition_html}</div>')
        
        # Continue button to return to room selection
        if st.button("Continue to Room Selection", key="r1g_continue", type="primary"):
            st.session_state.game_room = None
            st.session_state.mode = "\U0001f3ae Escape Room Showcase"
            st.rerun()

    render_html("""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:#1a3a5c;"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#455a64;"></span> Wall</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#4fc3f7;"></span> Start</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#76ff03;"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#81d4fa;"></span> Slippery</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#29b6f6;"></span> Agent</span>
        <span class="legend-item">\u2191\u2192\u2193\u2190 Policy</span>
    </div>
    """)
