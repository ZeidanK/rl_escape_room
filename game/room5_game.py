from __future__ import annotations

import glob
import os

import streamlit as st

from agents.dqn import (
    DQNNetwork,
    extract_dqn_action_values,
    load_dqn_model,
    rollout_dqn_policy,
)
from core.types import (
    DQNConfig,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    Room4MotionConfig,
    Room5ObstacleConfig,
    Room5RewardConfig,
)
from environments.room5_obstacles import Room5Obstacles
from game.game_view_common import (
    check_and_unlock_achievements,
    render_back_button,
    render_room_transition,
)
from game.html_rendering import render_html
from game.hud import render_hud
from game.presentation import (
    final_summary_success,
    render_assignment_proof,
    render_model_provenance,
    render_open_lab_button,
)
from game.theme import get_theme


def _epsilon_config_from_metadata(raw: dict, fallback: EpsilonScheduleConfig) -> EpsilonScheduleConfig:
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


def _dqn_config_from_metadata(metadata: dict | None, fallback: DQNConfig | None = None) -> DQNConfig:
    fallback = fallback or DQNConfig()
    cfg = (metadata or {}).get("training_config", {})
    epsilon = _epsilon_config_from_metadata(cfg.get("epsilon", {}), fallback.epsilon)
    return DQNConfig(
        episodes=int(cfg.get("episodes", fallback.episodes)),
        learning_rate=float(cfg.get("learning_rate", fallback.learning_rate)),
        gamma=float(cfg.get("gamma", fallback.gamma)),
        max_steps=int(cfg.get("max_steps", fallback.max_steps)),
        seed=int((metadata or {}).get("training_seed", fallback.seed)),
        epsilon=epsilon,
        replay_capacity=int(cfg.get("replay_capacity", fallback.replay_capacity)),
        batch_size=int(cfg.get("batch_size", fallback.batch_size)),
        warmup_steps=int(cfg.get("warmup_steps", fallback.warmup_steps)),
        target_update_interval=int(cfg.get("target_update_interval", fallback.target_update_interval)),
        hidden_units=int((metadata or {}).get("hidden_units", fallback.hidden_units)),
    )


def _preferred_room5_model_stem() -> str | None:
    model_dir = os.path.join("storage", "models", "room5_dqn")
    showcase = os.path.join(model_dir, "showcase_dqn")
    if os.path.exists(showcase + ".json") and os.path.exists(showcase + ".npz"):
        return showcase

    files = glob.glob(os.path.join(model_dir, "*.json"))
    files = [f for f in files if os.path.exists(f.replace(".json", ".npz"))]
    if not files:
        return None
    return max(files, key=os.path.getmtime).replace(".json", "")


def _load_room5_game_model(filepath_stem: str) -> None:
    network, meta = load_dqn_model(filepath_stem)
    st.session_state.r5g_network = network
    st.session_state.r5g_meta = meta
    st.session_state.r5g_rollout = None
    st.session_state.r5g_rollout_key = None
    st.session_state.r5g_loaded = True
    st.session_state.r5g_model_stem = filepath_stem
    st.session_state.r5g_load_error = None
    st.session_state.r5g_autoload_disabled = False


def _autoload_room5_game_model() -> None:
    if st.session_state.r5g_network is not None or st.session_state.get("r5g_autoload_disabled"):
        return
    stem = _preferred_room5_model_stem()
    if stem is None:
        return
    try:
        _load_room5_game_model(stem)
    except ValueError as exc:
        st.session_state.r5g_load_error = str(exc)


def _make_room5_env(
    *,
    metadata: dict | None,
    fixed_layout: bool,
    layout_seed: int,
    max_steps: int,
) -> Room5Obstacles:
    env_meta = metadata.get("environment_config", {}) if isinstance(metadata, dict) else {}
    min_obstacles = int(env_meta.get("min_obstacles", 3))
    max_obstacles = max(min_obstacles, int(env_meta.get("max_obstacles", 5)))
    obstacle_config = Room5ObstacleConfig(
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
        obstacle_width_m=float(env_meta.get("obstacle_width_m", 0.5)),
        observation_distance_m=float(env_meta.get("observation_distance_m", 2.5)),
        nearest_obstacles=int(env_meta.get("nearest_obstacles", 4)),
        layout_seed=int(layout_seed),
        fixed_layout=bool(fixed_layout),
    )
    motion_config = Room4MotionConfig(time_step_s=float(env_meta.get("time_step_s", 0.05)))
    reward_config = Room5RewardConfig()
    return Room5Obstacles(
        motion_config=motion_config,
        obstacle_config=obstacle_config,
        reward_config=reward_config,
        max_steps=int(max_steps),
    )


def _render_room5_svg(env: Room5Obstacles, rollout=None) -> str:
    state = env.render()
    margin = 24.0
    canvas = 520.0
    span = canvas - 2 * margin
    sx = span / env.motion.room_width_m
    sy = span / env.motion.room_height_m

    def pt(x: float, y: float) -> tuple[float, float]:
        return margin + x * sx, canvas - margin - y * sy

    if rollout is not None:
        points = [rollout.start_state[:2]]
        points.extend(step.next_raw_state[:2] for step in rollout.trajectory)
    else:
        points = list(state.trajectory)
    path_points = " ".join(f"{pt(x, y)[0]:.1f},{pt(x, y)[1]:.1f}" for x, y in points)
    visible = {(round(o.center_x, 6), round(o.center_y, 6)) for o in state.visible_obstacles}

    parts = [
        f'<svg class="room5-showcase-canvas" viewBox="0 0 {canvas:.0f} {canvas:.0f}" width="100%" '
        'style="max-width:620px;background:#111827;border:1px solid #334155;border-radius:8px;">',
        '<rect x="24" y="24" width="472" height="472" fill="#0f172a" stroke="#475569" stroke-width="2"/>',
    ]
    for i in range(11):
        x = margin + i * span / 10
        y = margin + i * span / 10
        parts.append(f'<line x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="496" stroke="#1e293b" stroke-width="1"/>')
        parts.append(f'<line x1="24" y1="{y:.1f}" x2="496" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>')

    ex, ey = pt(*state.exit_center)
    parts.append(
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{state.exit_radius_m * sx:.1f}" '
        'fill="#22c55e" fill-opacity="0.28" stroke="#86efac" stroke-width="2"/>'
    )
    if points:
        ox, oy = pt(*points[0])
        parts.append(
            f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{state.observation_distance_m * sx:.1f}" '
            'fill="#38bdf8" fill-opacity="0.08" stroke="#38bdf8" stroke-opacity="0.5" stroke-dasharray="5 5"/>'
        )

    for obstacle in state.obstacles:
        cx, cy = pt(obstacle.center_x, obstacle.center_y)
        size = obstacle.width_m * sx
        stroke = "#facc15" if (round(obstacle.center_x, 6), round(obstacle.center_y, 6)) in visible else "#f97316"
        parts.append(
            f'<rect class="room5-obstacle" x="{cx - size / 2:.1f}" y="{cy - size / 2:.1f}" '
            f'width="{size:.1f}" height="{size:.1f}" fill="#7f1d1d" stroke="{stroke}" stroke-width="2"/>'
        )

    if len(points) >= 2:
        parts.append(
            f'<polyline points="{path_points}" fill="none" stroke="#67e8f9" stroke-width="3" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if points:
        start_x, start_y = pt(*points[0])
        end_x, end_y = pt(*points[-1])
        parts.append(f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="#60a5fa"/>')
        parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="8" fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def _rollout_rows(rollout) -> list[dict]:
    return [
        {
            "step": step.index,
            "action": step.requested_action.name,
            "reward": step.reward,
            "cumulative": step.cumulative_reward,
            "visible_obstacles": step.visible_obstacle_count,
            "event": step.event or "",
            "distance_to_exit_m": step.distance_to_exit_m,
        }
        for step in rollout.trajectory[:50]
    ]


def render_room5_game() -> None:
    theme = get_theme("room5")
    render_html(
        f'<div class="narrative-box" style="border-left-color:{theme.primary};">'
        "The chamber now contains avoidable square obstacles. The agent observes only nearby "
        "obstacle records and escapes with a replay-buffer DQN policy.</div>"
    )
    render_assignment_proof("room5")
    render_open_lab_button("room5", key="r5g_open_lab")
    render_back_button("r5g_back")

    for key in [
        "r5g_network",
        "r5g_meta",
        "r5g_rollout",
        "r5g_rollout_key",
        "r5g_loaded",
        "r5g_model_stem",
        "r5g_load_error",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "r5g_autoload_disabled" not in st.session_state:
        st.session_state.r5g_autoload_disabled = False

    _autoload_room5_game_model()

    meta = st.session_state.r5g_meta
    config = _dqn_config_from_metadata(meta)

    with st.sidebar:
        st.header("Room 5 Controls")
        replay_seed = st.number_input("Replay Seed", min_value=0, max_value=2**31 - 1, value=7, step=1, key="r5g_seed")
        layout_seed = st.number_input(
            "Layout Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=1007,
            step=1,
            key="r5g_layout_seed",
        )
        max_steps = st.number_input(
            "Max Steps",
            min_value=50,
            max_value=1500,
            value=int(config.max_steps),
            step=10,
            key="r5g_max_steps",
        )
        fixed_layout = st.checkbox("Fixed Layout", value=False, key="r5g_fixed_layout")

        load_col, reset_col = st.columns(2)
        if load_col.button("Load Latest Model", key="r5g_load"):
            stem = _preferred_room5_model_stem()
            if stem is None:
                st.session_state.r5g_load_error = "No Room 5 DQN model found."
            else:
                try:
                    _load_room5_game_model(stem)
                    st.rerun()
                except ValueError as exc:
                    st.session_state.r5g_load_error = str(exc)
        if reset_col.button("Reset Replay", key="r5g_reset"):
            st.session_state.r5g_rollout = None
            st.session_state.r5g_rollout_key = None
            st.rerun()

    network: DQNNetwork | None = st.session_state.r5g_network
    load_error = st.session_state.r5g_load_error
    if network is None:
        if load_error:
            st.error(f"Room 5 model load failed: {load_error}")
        else:
            st.info("No Room 5 DQN model was found. Open the lab to train or load one.")
        preview_env = _make_room5_env(
            metadata=meta,
            fixed_layout=bool(fixed_layout),
            layout_seed=int(layout_seed),
            max_steps=int(max_steps),
        )
        preview_env.reset(seed=int(replay_seed), layout_seed=int(layout_seed))
        render_html(_render_room5_svg(preview_env))
        return

    rollout_key = (
        st.session_state.r5g_model_stem,
        int(replay_seed),
        int(layout_seed),
        int(max_steps),
        bool(fixed_layout),
    )
    if st.session_state.r5g_rollout is None or st.session_state.r5g_rollout_key != rollout_key:
        with st.spinner("Generating Room 5 greedy replay..."):
            st.session_state.r5g_rollout = rollout_dqn_policy(
                lambda: _make_room5_env(
                    metadata=meta,
                    fixed_layout=bool(fixed_layout),
                    layout_seed=int(layout_seed),
                    max_steps=int(max_steps),
                ),
                network,
                seed=int(replay_seed),
                layout_seed=int(layout_seed),
                max_steps=int(max_steps),
            )
            st.session_state.r5g_rollout_key = rollout_key

    rollout = st.session_state.r5g_rollout
    if meta:
        render_model_provenance(
            title="Room 5 - Dynamic Obstacles",
            model_stem=st.session_state.r5g_model_stem,
            metadata=meta,
            evaluation_success=final_summary_success("Room 5"),
        )

    env = _make_room5_env(
        metadata=meta,
        fixed_layout=bool(fixed_layout),
        layout_seed=int(layout_seed),
        max_steps=int(max_steps),
    )
    env.reset(seed=int(replay_seed), layout_seed=int(layout_seed))

    if rollout is None:
        render_html(_render_room5_svg(env))
        return

    status_badges = [
        '<span class="badge-success">Success</span>' if rollout.success else '<span class="badge-failure">No Escape</span>',
    ]
    if rollout.obstacle_collisions:
        status_badges.append('<span class="badge-collision">Obstacle Hit</span>')
    render_html(
        render_hud(
            room_name="The Obstacle Lab",
            algorithm="NumPy DQN",
            step=rollout.steps,
            max_steps=int(max_steps),
            state_str=(
                f"x={rollout.final_state[0]:.2f}, y={rollout.final_state[1]:.2f}, "
                f"vx={int(rollout.final_state[2])}, vy={int(rollout.final_state[3])}"
            ),
            total_reward=rollout.total_reward,
            status_badges=status_badges,
            custom_items=[
                ("Layout Seed", str(rollout.layout_seed)),
                ("Visible Obstacle Steps", str(rollout.visible_obstacle_steps)),
            ],
        )
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Success", "Yes" if rollout.success else "No")
    c2.metric("Steps", rollout.steps)
    c3.metric("Return", f"{rollout.total_reward:.2f}")
    c4.metric("Obstacle Hits", rollout.obstacle_collisions)

    render_html(_render_room5_svg(env, rollout))

    t1, t2 = st.tabs(["Replay", "Action Values"])
    with t1:
        st.dataframe(_rollout_rows(rollout), width="stretch", hide_index=True)
    with t2:
        obs = env.reset(seed=int(replay_seed), layout_seed=int(layout_seed))
        q_vals = extract_dqn_action_values(network, obs)
        st.dataframe(
            [{"action": action, "q_value": value} for action, value in q_vals.items()],
            width="stretch",
            hide_index=True,
        )

    achievements = check_and_unlock_achievements("room5", rollout)
    for ach in achievements:
        st.toast(f"{ach.emoji} {ach.name}: {ach.description}")
    render_room_transition("room5", rollout, achievements)

    render_html(f"""
    <div class="game-legend">
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_trap};"></span> Obstacle</span>
        <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Trajectory</span>
    </div>
    """)
