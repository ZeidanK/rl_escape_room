"""Room 4 — Momentum Chamber game view. Continuous state space with trajectory visualization."""

import streamlit as st
import numpy as np

from game.theme import get_theme
from game.game_view_common import (
    render_back_button,
    check_and_unlock_achievements,
    render_room_transition,
    render_game_legend,
)

from core.types import (
    SlipConfig, ContinuousRewardConfig, Room4MotionConfig, StartMode,
    VELOCITY_BY_ACTION, VelocityAction,
)
from environments.room4_continuous import Room4Continuous
from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    LinearTileQFunction,
    load_approximate_model,
    rollout_approximate_policy,
)
from features.tile_coding import TileCoder, TileCodingConfig
from game.achievements import AchievementTracker
from game.room_transitions import render_transition_content
from game.home_page import ROOM_DEFS


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


def render_room4_game():
    # Showcase view for a trained approximate SARSA model in continuous space.
    # It loads weights, rebuilds the tile coder, then draws the trajectory.
    theme = get_theme("room4")

    st.markdown(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        f'The discrete grid disappears. The agent must control velocity in continuous space '
        f'and generalize from overlapping tile-coded features.</div>',
        unsafe_allow_html=True,
    )

    render_back_button("r4g_back")

    # Initialize session state before controls so first entry can auto-load
    # bundled showcase artifacts without waiting for a manual sidebar click.
    for key in ["r4g_weights", "r4g_meta", "r4g_rollout", "r4g_loaded", "r4g_model_stem", "r4g_load_error"]:
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
            for key in ["r4g_weights", "r4g_meta", "r4g_rollout", "r4g_loaded", "r4g_model_stem", "r4g_load_error"]:
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
    rollout = st.session_state.r4g_rollout

    if weights is None or meta is None:
        load_error = st.session_state.get("r4g_load_error")
        if load_error:
            st.caption(f"Room 4 model auto-load skipped: {load_error}")
        else:
            st.caption("No Room 4 model is loaded.")
        return

    # Build environment from saved metadata so the replay uses the same motion
    # and reward settings as training.
    tc_cfg = _tile_config_from_meta(meta)
    motion_cfg = _motion_config_from_meta(meta)
    reward_cfg = _reward_config_from_meta(meta)
    start_mode = _start_mode_from_meta(meta)
    max_steps = _max_steps_from_meta(meta)
    seed = _seed_from_meta(meta)
    
    env = Room4Continuous(
        motion_config=motion_cfg,
        reward_config=reward_cfg,
        max_steps=max_steps,
        start_mode=start_mode,
        seed=seed,
    )

    # Build rollout once and cache it; replay/render controls should not change
    # the evaluated trajectory unless the model is reset/reloaded.
    if rollout is None:
        q_func = _q_function_from_weights(weights, tc_cfg, motion_cfg)
        make_env = lambda: Room4Continuous(
            motion_config=motion_cfg,
            reward_config=reward_cfg,
            max_steps=max_steps,
            start_mode=start_mode,
            seed=seed,
        )
        rollout = rollout_approximate_policy(make_env, q_func, seed=seed, max_steps=max_steps)
        st.session_state.r4g_rollout = rollout
        st.rerun()

    # HUD
    status_badges = []
    if rollout.success:
        status_badges.append('<span class="badge-success">SUCCESS</span>')
    elif not rollout.success:
        status_badges.append('<span class="badge-failure">FAILED</span>')

    cfg = _training_config(meta)
    final_distance = _final_distance_to_exit_m(rollout, motion_cfg)

    from game.hud import render_hud
    st.markdown(render_hud(
        room_name="\U0001f300 Room 4: The Momentum Chamber",
        algorithm=f"Approximate SARSA (Tile Coding) | \u03b1={float(cfg.get('alpha', meta.get('alpha', 0.1))):.2f} \u03b3={float(cfg.get('gamma', meta.get('gamma', 0.99))):.2f} | Tilings={tc_cfg.num_tilings}",
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
    ), unsafe_allow_html=True)

    # Continuous trajectory visualization
    st.markdown("### Continuous Trajectory")
    
    # Create a 2D discretized view of the trajectory
    room_w = motion_cfg.room_width_m
    room_h = motion_cfg.room_height_m
    grid_size = grid_res
    cell_w = room_w / grid_size
    cell_h = room_h / grid_size
    
    # Build discretized grid
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Mark exit
    ex, ey = motion_cfg.exit_center
    er = motion_cfg.exit_radius_m
    for row in range(grid_size):
        for col in range(grid_size):
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            if (cx - ex) ** 2 + (cy - ey) ** 2 <= er ** 2:
                grid[row][col] = "E"
    
    # Mark start
    sx, sy = rollout.start_state[0], rollout.start_state[1]
    sr = int(sy / cell_h)
    sc = int(sx / cell_w)
    if 0 <= sr < grid_size and 0 <= sc < grid_size:
        grid[sr][sc] = "S"
    
    # Mark trajectory
    for step in rollout.trajectory:
        x, y, _, _ = step.state
        r = int(y / cell_h)
        c = int(x / cell_w)
        if 0 <= r < grid_size and 0 <= c < grid_size:
            if grid[r][c] in (".", "S"):
                grid[r][c] = "*"
    
    # Mark collisions
    for step in rollout.trajectory:
        if step.collision:
            x, y, _, _ = step.state
            r = int(y / cell_h)
            c = int(x / cell_w)
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r][c] = "X"
    
    # Direction arrows at intervals
    arrow_interval = max(1, len(rollout.trajectory) // 20)
    for idx, step in enumerate(rollout.trajectory):
        if idx % arrow_interval == 0:
            x, y, vx, vy = step.state
            r = int(y / cell_h)
            c = int(x / cell_w)
            if 0 <= r < grid_size and 0 <= c < grid_size:
                grid[r][c] = _velocity_arrow(vx, vy)
    
    # Display as code block
    st.code("\n".join(" ".join(row) for row in grid), language="text")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Steps", rollout.steps)
    col2.metric("Reward", f"{rollout.total_reward:.1f}")
    col3.metric("Distance", f"{rollout.distance_travelled_m:.2f}m")
    col4.metric("Final Dist.", f"{final_distance:.2f}m")
    
    if rollout.collision_count > 0:
        st.warning(f"Collisions: {rollout.collision_count}")
    
    # Room transition
    achievements = check_and_unlock_achievements("room4", rollout)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")
    
    render_room_transition("room4", rollout, achievements)
    
    # Legend
    st.markdown(f"""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Trajectory</span>
        <span class="legend-item">X Collision</span>
        <span class="legend-item">↑→↓← Velocity</span>
    </div>
    """, unsafe_allow_html=True)


def _velocity_arrow(vx: int, vy: int) -> str:
    """Convert velocity to arrow symbol."""
    if vx == 0 and vy == 1:
        return "\u2191"  # UP
    elif vx == 0 and vy == -1:
        return "\u2193"  # DOWN
    elif vx == 1 and vy == 0:
        return "\u2192"  # RIGHT
    elif vx == -1 and vy == 0:
        return "\u2190"  # LEFT
    elif vx == 1 and vy == 1:
        return "\u2197"  # UP-RIGHT
    elif vx == 1 and vy == -1:
        return "\u2198"  # DOWN-RIGHT
    elif vx == -1 and vy == 1:
        return "\u2196"  # UP-LEFT
    elif vx == -1 and vy == -1:
        return "\u2199"  # DOWN-LEFT
    return "*"
