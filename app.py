import streamlit as st

import numpy as np

from core.types import (
    Action,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    QLearningConfig,
    SarsaConfig,
    SlipConfig,
    StartMode,
    StepResult,
    ValueIterationConfig,
    VelocityAction,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import ROOM2_GRID, ROOM2_MAP, Room2SARSA
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
    build_policy_symbols,
    build_value_matrix,
)
from visualization.sarsa_visualization import (
    build_greedy_policy_symbols,
    build_q_value_tables,
    build_training_dataframe,
    render_sarsa_trajectory_overlay,
)
from visualization.q_learning_visualization import (
    build_q_learning_training_dataframe,
    build_room3_policy_symbols,
    build_room3_q_value_table,
    render_q_learning_trajectory_overlay,
)
from training.algorithm_comparison import (
    print_summary as print_comparison_summary,
    run_matched_comparison,
    run_tuned_comparison,
    save_comparison,
)
from environments.room4_continuous import ContinuousRewardConfig, Room4Continuous, Room4MotionConfig
from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    evaluate_approximate_policy,
    load_approximate_model,
    rollout_approximate_policy,
    save_approximate_model,
)
from features.tile_coding import TileCodingConfig
from visualization.approximate_sarsa_visualization import (
    build_action_field as build_approx_action_field,
    build_training_dataframe as build_approx_training_dataframe,
    build_value_surface as build_approx_value_surface,
    render_continuous_trajectory as render_approx_trajectory,
)
from training.approximate_sarsa_experiments import (
    run_confirmation_experiments as run_approx_confirmation,
    run_screening_stage_a,
    run_screening_stage_b,
)

ROOM_CLASSES = {
    "Room 1 — Ice Maze (DP)": Room1DP,
    "Room 2 — Laser Corridor (SARSA)": Room2SARSA,
    "Room 3 — Key Vault (Q-Learning)": Room3QLearning,
}

ACTION_BUTTONS = {
    "UP": Action.UP,
    "RIGHT": Action.RIGHT,
    "DOWN": Action.DOWN,
    "LEFT": Action.LEFT,
}

st.set_page_config(page_title="RL Escape Room", layout="wide", page_icon="🧊")
st.title("RL Escape Room")

# --- Session state ---
for key in [
    "env", "last_result", "room_key",
    "vi_result", "vi_solve_key", "vi_rollout_result", "vi_rollout_key",
    "vi_eval_summary", "vi_eval_key", "dp_env",
    "sarsa_result", "sarsa_train_key", "sarsa_eval_summary", "sarsa_eval_key",
    "sarsa_rollout", "sarsa_rollout_key", "sarsa_env_factory",
    "ql_result", "ql_train_key", "ql_eval_summary", "ql_eval_key",
    "ql_rollout", "ql_rollout_key", "ql_env_factory",
    "comp_matched", "comp_tuned", "comp_key",
    "approx_result", "approx_train_key",
    "approx_eval_fixed", "approx_eval_fixed_key",
    "approx_eval_gen", "approx_eval_gen_key",
    "approx_rollout", "approx_rollout_key",
    "approx_env_factory",
    "game_mode", "game_room", "show_lab",
]:
    if key not in st.session_state:
        st.session_state[key] = None
if "mode" not in st.session_state:
    st.session_state.mode = "\U0001f3ae Escape Room Showcase"

# ============================================================
# Game mode imports
# ============================================================
from game.home_page import render_home_page, ROOM_DEFS, ROOM_NARRATIVES
from game.room1_game import render_room1_game
from game.room2_game import render_room2_game
from game.room3_game import render_room3_game
from game.room4_game import render_room4_game
from game.comparison_theater import render_comparison_theater
from game.theme import render_global_styles
from game.achievements import AchievementTracker

MODE_LABELS = [
    "\U0001f3ae Escape Room Showcase",
    "Home",
    "---",
    "\U0001f579\ufe0f Manual Play",
    "---",
    "\U0001f52c Learning Laboratory",
    "Room 1 \u2014 DP",
    "Room 2 \u2014 SARSA",
    "Room 3 \u2014 Q-Learning",
    "Room 4 \u2014 Function Approximation",
    "Algorithm Comparison",
    "---",
    "\U0001f4d6 About the Project",
]

# --- Mode selector ---
GAME_LABEL = "\U0001f3ae Escape Room Showcase"
LAB_LABEL = "\U0001f52c Learning Laboratory"
ABOUT_LABEL = "\U0001f4d6 About the Project"

# Selectable = game showcase, analysis rooms, manual, about
SELECTABLE_MODES = [
    GAME_LABEL,
    "\U0001f579\ufe0f Manual Play",
    ABOUT_LABEL,
    "Home",
    "Room 1 \u2014 DP",
    "Room 2 \u2014 SARSA",
    "Room 3 \u2014 Q-Learning",
    "Room 4 \u2014 Function Approximation",
    "Algorithm Comparison",
]

_MODE_NAME_MAP = {
    GAME_LABEL: GAME_LABEL,
    "\U0001f579\ufe0f Manual Play": "Manual Environment",
    ABOUT_LABEL: ABOUT_LABEL,
    "Home": "Home",
    "Room 1 \u2014 DP": "Room 1 \u2014 DP",
    "Room 2 \u2014 SARSA": "Room 2 \u2014 SARSA",
    "Room 3 \u2014 Q-Learning": "Room 3 \u2014 Q-Learning",
    "Room 4 \u2014 Function Approximation": "Room 4 \u2014 Function Approximation",
    "Algorithm Comparison": "Algorithm Comparison",
}

# Custom sidebar with categorized radio buttons
st.sidebar.markdown(
    '<div style="font-size:0.75em;color:#616161;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Navigation</div>',
    unsafe_allow_html=True,
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
    st.markdown("""
    <script>
    document.body.classList.add('high-contrast');
    </script>
    """, unsafe_allow_html=True)

# Determine which radio index to show based on current mode
mode = st.session_state.mode
_reverse_map = {v: k for k, v in _MODE_NAME_MAP.items()}
_sidebar_to_show = _reverse_map.get(mode, GAME_LABEL)
_default_idx = SELECTABLE_MODES.index(_sidebar_to_show) if _sidebar_to_show in SELECTABLE_MODES else 0

sidebar_selection = st.sidebar.radio(
    "Mode",
    SELECTABLE_MODES,
    index=_default_idx,
    key="mode_selector",
    label_visibility="collapsed",
)
sidebar_effective = _MODE_NAME_MAP.get(sidebar_selection, sidebar_selection)

# Sidebar always overrides; clear deep-linked game room on nav change
if sidebar_effective != mode:
    st.session_state.game_room = None
    st.session_state.mode = sidebar_effective
    st.rerun()

mode = st.session_state.mode

# Ensure achievement tracker exists
AchievementTracker.from_session_state()

# ============================================================
# MODE: Escape Room Showcase
# ============================================================
if st.session_state.mode == GAME_LABEL:
    st.markdown(render_global_styles(), unsafe_allow_html=True)
    game_room = st.session_state.get("game_room")
    if game_room == "room1":
        render_room1_game()
    elif game_room == "room2":
        render_room2_game()
    elif game_room == "room3":
        render_room3_game()
    elif game_room == "room4":
        render_room4_game()
    else:
        render_home_page()

# ============================================================
# MODE: About the Project
# ============================================================
elif st.session_state.mode == ABOUT_LABEL:
    st.markdown(render_global_styles(), unsafe_allow_html=True)
    st.markdown("## About RL Escape Room")
    st.markdown("""
    This project applies four reinforcement learning algorithms of increasing difficulty to
    navigate a series of escape-room environments. Each room introduces a new challenge:

    | Room | Algorithm | Key Concept |
    |------|-----------|-------------|
    | 1 — Frozen Maze | Value Iteration | Dynamic Programming on known MDP |
    | 2 — Laser Corridor | SARSA | On-policy TD learning with risk sensitivity |
    | 3 — Key Vault | Q-Learning | Off-policy TD with augmented state space |
    | 4 — Momentum Chamber | Approximate SARSA | Linear function approximation with tile coding |

    The **Escape Room Showcase** presents the agents as a campaign-style game with animated
    replay, while the **Learning Laboratory** provides full analysis tools including training
    curves, policy visualization, Q-value inspection, and algorithm comparison.
    """)

    # Screenshots
    st.markdown("### Screenshots")
    screenshots = [
        ("docs/screenshots/home.png", "Home / Campaign Selection"),
        ("docs/screenshots/room1_value_policy.png", "Room 1 — Value Iteration Convergence & Policy"),
        ("docs/screenshots/room2_training.png", "Room 2 — SARSA Training Progress"),
        ("docs/screenshots/room3_policy_no_key.png", "Room 3 — Q-Learning Policy (No Key)"),
        ("docs/screenshots/room4_trajectory.png", "Room 4 — Approximate SARSA Continuous Trajectory"),
        ("docs/screenshots/comparison.png", "Algorithm Comparison — SARSA vs Q-Learning"),
    ]
    for i, (path, caption) in enumerate(screenshots):
        if i % 2 == 0:
            cols = st.columns(2)
        try:
            cols[i % 2].image(path, caption=caption, use_container_width=True)
        except Exception:
            cols[i % 2].markdown(f"*Screenshot not found: {path}*")

    st.markdown("### Technical Stack")
    st.markdown("""
    - **Framework:** Streamlit
    - **Runtime:** Python 3.11+
    - **Numerics:** NumPy
    - **RL Algorithms:** Value Iteration, SARSA, Q-Learning, Semi-Gradient SARSA
    - **Function Approximation:** Tile Coding with linear basis functions
    - **Visualization:** SVG via `st.components.v1.html` and inline CSS
    """)

    st.markdown("### Repository")
    st.markdown("[GitHub](https://github.com/anomalyco/rilearningPro)")

# ============================================================
# MODE: Home (original, kept for backward compat / lab entry)
# ============================================================
elif st.session_state.mode == "Home":
    st.header("Project Objective")
    st.markdown("""
    Apply four reinforcement learning algorithms of increasing difficulty to
    navigate a series of escape-room grids. Each room introduces a new challenge:
    stochastic transitions, trap cells, key-collection mechanics, and continuous
    state spaces.
    """)

    st.header("The Four Rooms")
    cols = st.columns(4)
    cols[0].markdown("**Room 1 — Ice Maze**")
    cols[0].markdown("Value Iteration on known MDP with slippery cells.")
    cols[1].markdown("**Room 2 — Laser Corridor**")
    cols[1].markdown("SARSA learning risk-aware behaviour under slip and traps.")
    cols[2].markdown("**Room 3 — Key Vault**")
    cols[2].markdown("Q-Learning with key-collection and locked-exit states.")
    cols[3].markdown("**Room 4 — Momentum Chamber**")
    cols[3].markdown("Continuous state (x,y,vx,vy) with tile coding + linear approx SARSA.")

    st.header("Room & Algorithm Summary")
    st.dataframe({
        "Room": ["Room 1", "Room 2", "Room 3", "Room 4"],
        "Algorithm": ["Value Iteration", "SARSA", "Q-Learning", "Approximate SARSA"],
        "State Space": ["10×10 grid", "10×10 grid", "200 states (grid × key)", "Continuous (x,y,vx,vy)"],
        "On/Off Policy": ["—", "On-policy", "Off-policy", "On-policy"],
        "Model Known": ["Yes", "No", "No", "No"],
    }, use_container_width=True)

    st.header("Instructions")
    st.markdown("""
    1. Use the **sidebar** to select a mode.
    2. For Rooms 1–3, select a room, configure parameters, and run the algorithm.
    3. For Room 4, configure tile-coding and training parameters.
    4. View training curves, policies, and trajectory replays.
    5. The **Algorithm Comparison** mode compares SARSA and Q-Learning.
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
    import csv, os
    csv_path = "storage/experiments/final/final_summary.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        st.dataframe(rows, use_container_width=True)
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
                    st.rerun()
            st.markdown("---")
            st.markdown("**Status**")
            st.metric("Step", env.step_count)
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
    
    if env is not None:
        # Determine room_id for theme
        room_id_map = {
            "Room 1 — Ice Maze (DP)": "room1",
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
            st.markdown(f'<div style="overflow:hidden;">{svg}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render grid: {e}")
            st.code(env.render_ansi())
        
        # Legend
        st.markdown(f"""
        <div class="game-legend">
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_empty};"></span> Empty</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_wall};"></span> Wall</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_start};"></span> Start</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_exit};"></span> Exit</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.cell_slippery};"></span> Slippery</span>
            <span class="legend-item"><span class="legend-swatch" style="background:{theme.agent_color};"></span> Agent</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MODE: Room 1 — DP
# ============================================================
elif st.session_state.mode == "Room 1 \u2014 DP":
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
        slip_cfg = SlipConfig(p_int, p_left, p_right)
        st.markdown("---")
        rollout_seed = st.number_input("Rollout Seed", min_value=0, max_value=2**31 - 1, value=0, step=1,
                                       help="Random seed for the policy rollout (trajectory simulation).")
        eval_ep = st.number_input("Eval Episodes", min_value=1, max_value=1000, value=100, step=1,
                                  help="Number of episodes to run for policy evaluation.")
        solve_params = (gamma, tolerance, max_it, p_int, p_left, p_right)
        roll_params = solve_params + (rollout_seed,)
        ev_params = solve_params + (eval_ep,)
        col1, col2 = st.columns(2)
        solve_clicked = col1.button("Solve", type="primary", disabled=not slip_valid)
        if st.session_state.get("vi_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all DP results.")
            if col2.button("Confirm Reset", key="vi_confirm"):
                st.session_state.vi_result = None
                st.session_state.vi_rollout_result = None
                st.session_state.vi_eval_summary = None
                st.session_state.vi_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="vi_cancel_reset"):
                st.session_state.vi_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results"):
            st.session_state.vi_confirm_reset = True
            st.rerun()
        rollout_clicked = st.button("Run Rollout", disabled=st.session_state.vi_result is None)
        eval_clicked = st.button("Evaluate Policy", disabled=st.session_state.vi_result is None)

    if solve_clicked or (st.session_state.vi_solve_key != solve_params and st.session_state.vi_result is None):
        with st.spinner("Running Value Iteration..."):
            env = Room1DP(slip_config=slip_cfg, max_steps=200, seed=42)
            st.session_state.dp_env = env
            config = ValueIterationConfig(gamma=gamma, tolerance=tolerance, max_iterations=max_it)
            vi_r = ValueIterationAgent(env, config).solve()
            st.session_state.vi_result = vi_r
            st.session_state.vi_solve_key = solve_params
            st.session_state.vi_rollout_result = None
            st.session_state.vi_eval_summary = None
            st.rerun()

    vi_result = st.session_state.vi_result
    if vi_result is not None:
        env = st.session_state.dp_env
        if st.session_state.vi_rollout_key != roll_params:
            if not solve_clicked:
                with st.spinner("Running rollout..."):
                    st.session_state.vi_rollout_result = rollout_policy(env, vi_result.policy, seed=rollout_seed)
                    st.session_state.vi_rollout_key = roll_params
        if rollout_clicked:
            with st.spinner("Running rollout..."):
                st.session_state.vi_rollout_result = rollout_policy(env, vi_result.policy, seed=rollout_seed)
                st.session_state.vi_rollout_key = roll_params
                st.rerun()
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                st.session_state.vi_eval_summary = evaluate_policy(env, vi_result.policy, n_episodes=eval_ep)
                st.session_state.vi_eval_key = ev_params
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
            st.dataframe(np.round(build_value_matrix(env, vi_result.values), 2), use_container_width=True)
        with t3:
            lines = [" | ".join(r) for r in build_policy_symbols(env, vi_result.policy)]
            st.code("\n".join(lines), language="text")
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
elif st.session_state.mode == "Room 2 \u2014 SARSA":
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
        slip_cfg = SlipConfig(p_int, p_left, p_right)

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

        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train SARSA", type="primary", disabled=not slip_valid)
        if st.session_state.get("sarsa_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all SARSA results.")
            if col2.button("Confirm Reset", key="sarsa_confirm"):
                st.session_state.sarsa_result = None
                st.session_state.sarsa_eval_summary = None
                st.session_state.sarsa_rollout = None
                st.session_state.sarsa_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="sarsa_cancel_reset"):
                st.session_state.sarsa_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results"):
            st.session_state.sarsa_confirm_reset = True
            st.rerun()
        eval_clicked = st.button("Evaluate Policy", disabled=st.session_state.sarsa_result is None)
        save_clicked = st.button("Save Model", disabled=st.session_state.sarsa_result is None)
        load_clicked = st.button("Load Model")

    # Factory
    def make_env():
        return Room2SARSA(max_steps=max_steps, slip_config=slip_cfg)

    # Build config
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

    # --- Train ---
    if train_clicked or (st.session_state.sarsa_train_key != train_key and st.session_state.sarsa_result is None):
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
            st.session_state.sarsa_result = result
            st.session_state.sarsa_train_key = train_key
            st.session_state.sarsa_eval_summary = None
            st.session_state.sarsa_rollout = None
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    sarsa_result = st.session_state.sarsa_result

    if sarsa_result is not None:
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                summary = evaluate_sarsa_policy(make_env, sarsa_result.q_values, n_episodes=eval_ep)
                st.session_state.sarsa_eval_summary = summary
                st.session_state.sarsa_eval_key = eval_key
                st.rerun()

        # --- Save ---
        if save_clicked:
            import os
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room2_sarsa", f"sarsa_{ts}")
            save_model(sarsa_result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM2_GRID)
            st.success(f"Model saved to {stem}")

        # --- Load ---
        if load_clicked:
            import glob, os
            model_dir = os.path.join("storage", "models", "room2_sarsa")
            pattern = os.path.join(model_dir, "*.json")
            files = glob.glob(pattern)
            if files:
                latest = max(files).replace(".json", "")
                try:
                    q_vals, meta = load_model(latest, map_grid=ROOM2_GRID)
                    st.success(f"Loaded model from {latest}")
                except ValueError as e:
                    st.error(f"Load failed: {e}")
            else:
                st.info("No saved models found.")

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
            df = build_training_dataframe(sarsa_result.metrics)
            window = min(rw, max(10, episodes // 20))

            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(sarsa_result.metrics))
            c2.metric("Final Epsilon", f"{sarsa_result.final_epsilon:.4f}")
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

        # Tab 2: Learned Policy
        with t2:
            pol_sym = build_greedy_policy_symbols(env_sample, sarsa_result.q_values, greedy_policy)
            grid_lines = [" | ".join(r) for r in pol_sym]
            st.code("\n".join(grid_lines), language="text")
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
                    st.dataframe(overlay, use_container_width=True)
            else:
                st.info("No snapshots available.")

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
            else:
                st.info("Run an evaluation from the sidebar.")

# ============================================================
# MODE: Room 3 — Q-Learning
# ============================================================
elif st.session_state.mode == "Room 3 \u2014 Q-Learning":
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
        slip_cfg = SlipConfig(p_int, p_left, p_right)

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

        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train Q-Learning", type="primary", disabled=not slip_valid)
        if st.session_state.get("ql_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all Q-Learning results.")
            if col2.button("Confirm Reset", key="ql_confirm"):
                st.session_state.ql_result = None
                st.session_state.ql_eval_summary = None
                st.session_state.ql_rollout = None
                st.session_state.ql_confirm_reset = False
                st.rerun()
            if st.button("Cancel", key="ql_cancel_reset"):
                st.session_state.ql_confirm_reset = False
                st.rerun()
        elif col2.button("Reset Results", key="ql_reset"):
            st.session_state.ql_confirm_reset = True
            st.rerun()
        eval_clicked = st.button("Evaluate Policy", key="ql_eval_btn",
                                  disabled=st.session_state.ql_result is None)
        save_clicked = st.button("Save Model", key="ql_save_btn",
                                  disabled=st.session_state.ql_result is None)
        load_clicked = st.button("Load Model", key="ql_load_btn")

    def make_ql_env():
        return Room3QLearning(max_steps=max_steps, slip_config=slip_cfg)

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

    if train_clicked or (st.session_state.ql_train_key != train_key and st.session_state.ql_result is None):
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
            st.session_state.ql_result = result
            st.session_state.ql_train_key = train_key
            st.session_state.ql_eval_summary = None
            st.session_state.ql_rollout = None
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    ql_result = st.session_state.ql_result

    if ql_result is not None:
        if eval_clicked:
            with st.spinner(f"Evaluating {eval_ep} episodes..."):
                summary = evaluate_q_learning_policy(make_ql_env, ql_result.q_values, n_episodes=eval_ep)
                st.session_state.ql_eval_summary = summary
                st.session_state.ql_eval_key = eval_key
                st.rerun()

        if save_clicked:
            import os
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = os.path.join("storage", "models", "room3_q_learning", f"ql_{ts}")
            save_q_model(ql_result, stem, reward_config=None, slip_config=slip_cfg, map_grid=ROOM3_GRID)
            st.success(f"Model saved to {stem}")

        if load_clicked:
            import glob, os
            model_dir = os.path.join("storage", "models", "room3_q_learning")
            files = glob.glob(os.path.join(model_dir, "*.json"))
            if files:
                latest = max(files).replace(".json", "")
                try:
                    q_vals, meta = load_q_model(latest, map_grid=ROOM3_GRID)
                    st.success(f"Loaded model from {latest}")
                except ValueError as e:
                    st.error(f"Load failed: {e}")
            else:
                st.info("No saved models found.")

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
            df = build_q_learning_training_dataframe(ql_result.metrics)
            window = min(rw, max(10, episodes // 20))

            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(ql_result.metrics))
            c2.metric("Final Epsilon", f"{ql_result.final_epsilon:.4f}")
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

        with t2:
            pol_sym = build_room3_policy_symbols(env_sample, policy_no_key, has_key=False)
            st.code("\n".join([" | ".join(r) for r in pol_sym]), language="text")

        with t3:
            pol_sym = build_room3_policy_symbols(env_sample, policy_with_key, has_key=True)
            st.code("\n".join([" | ".join(r) for r in pol_sym]), language="text")

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
                    st.dataframe(overlay, use_container_width=True)
            else:
                st.info("No snapshots available.")

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
            else:
                st.info("Run an evaluation from the sidebar.")

# ============================================================
# MODE: Room 4 — Function Approximation
# ============================================================
elif st.session_state.mode == "Room 4 \u2014 Function Approximation":
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

        col1, col2 = st.columns(2)
        train_clicked = col1.button("Train Approx SARSA", type="primary")
        if st.session_state.get("approx_confirm_reset"):
            st.warning("Click again to confirm reset — this will clear all Approximate SARSA results.")
            if col2.button("Confirm Reset", key="approx_confirm"):
                st.session_state.approx_result = None
                st.session_state.approx_eval_fixed = None
                st.session_state.approx_eval_gen = None
                st.session_state.approx_rollout = None
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
        load_clicked = st.button("Load Model", key="approx_load_btn")

    def make_approx_env(start_mode=None):
        sm = start_mode if start_mode is not None else StartMode(approx_start_mode)
        return Room4Continuous(
            motion_config=Room4MotionConfig(),
            reward_config=ContinuousRewardConfig(distance_progress_scale=approx_progress_scale),
            max_steps=approx_max_steps,
            start_mode=sm,
        )

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

    # --- Train ---
    if train_clicked or (st.session_state.approx_train_key != train_key and st.session_state.approx_result is None):
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
            st.session_state.approx_result = result
            st.session_state.approx_train_key = train_key
            st.session_state.approx_eval_fixed = None
            st.session_state.approx_eval_gen = None
            st.session_state.approx_rollout = None
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    approx_result = st.session_state.approx_result

    if approx_result is not None:
        # --- Eval fixed ---
        if eval_fixed_clicked:
            with st.spinner(f"Evaluating fixed start ({eval_ep} episodes)..."):
                factory = lambda: make_approx_env(start_mode=StartMode.FIXED)
                ev = evaluate_approximate_policy(
                    factory, approx_result.weights, tc_cfg, Room4MotionConfig(),
                    n_episodes=eval_ep, start_mode=StartMode.FIXED,
                )
                st.session_state.approx_eval_fixed = ev
                st.session_state.approx_eval_fixed_key = eval_key_fixed
                st.rerun()

        # --- Eval generalization ---
        if eval_gen_clicked:
            with st.spinner(f"Evaluating generalization ({eval_ep} episodes)..."):
                factory = lambda: make_approx_env(start_mode=StartMode.RANDOM_LOWER_LEFT)
                ev_gen = evaluate_approximate_policy(
                    factory, approx_result.weights, tc_cfg, Room4MotionConfig(),
                    n_episodes=eval_ep, start_mode=StartMode.RANDOM_LOWER_LEFT,
                )
                st.session_state.approx_eval_gen = ev_gen
                st.session_state.approx_eval_gen_key = eval_key_gen
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
            st.success(f"Model saved to {stem}")

        # --- Load ---
        if load_clicked:
            import glob, os
            model_dir = os.path.join("storage", "models", "room4_approximate_sarsa")
            files = glob.glob(os.path.join(model_dir, "*.json"))
            if files:
                latest = max(files).replace(".json", "")
                try:
                    weights, meta = load_approximate_model(latest, expected_tile_coding=tc_cfg)
                    st.success(f"Loaded model from {latest}")
                except ValueError as e:
                    st.error(f"Load failed: {e}")
            else:
                st.info("No saved models found.")

        # --- Tabs ---
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "Training Progress", "Final Trajectory", "Training-Stage Replay",
            "Greedy Action Field", "Value Surface", "Evaluation", "Experiments",
        ])

        # Tab 1: Training Progress
        with t1:
            df = build_approx_training_dataframe(approx_result.metrics)
            window = min(rw, max(10, approx_episodes // 20))

            c1, c2, c3 = st.columns(3)
            c1.metric("Episodes", len(approx_result.metrics))
            c2.metric("Final Epsilon", f"{approx_result.final_epsilon:.4f}")
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

        # Tab 2: Final Trajectory
        with t2:
            last_rollout = None
            snap_keys = sorted(approx_result.snapshots.keys())
            if snap_keys:
                last_snap = approx_result.snapshots[snap_keys[-1]]
                if last_snap.rollout:
                    last_rollout = last_snap.rollout
                    env_disp = make_approx_env(start_mode=StartMode.FIXED)
                    env_disp.reset(seed=99)
                    traj_data = render_approx_trajectory(env_disp, last_rollout, max_arrows=20)
                    grid = traj_data["grid"]
                    st.code("\n".join([" ".join(r) for r in grid]), language="text")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Success", "Yes" if last_rollout.success else "No")
                    c2.metric("Steps", last_rollout.steps)
                    c3.metric("Reward", f"{last_rollout.total_reward:.1f}")
                    c4.metric("Distance", f"{last_rollout.distance_travelled_m:.1f}m")
            if not last_rollout:
                st.info("No trajectory available. Train first.")

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
                    traj_data = render_approx_trajectory(env_disp, snap.rollout, max_arrows=20)
                    grid = traj_data["grid"]
                    st.code("\n".join([" ".join(r) for r in grid]), language="text")
            else:
                st.info("No snapshots available.")

        # Tab 4: Greedy Action Field
        with t4:
            vx_choice = st.selectbox("Vx", [-1, 0, 1], index=1, key="af_vx")
            vy_choice = st.selectbox("Vy", [-1, 0, 1], index=1, key="af_vy")
            af_size = st.slider("Grid Resolution", 5, 30, 10, key="af_size")
            env_disp = make_approx_env(start_mode=StartMode.FIXED)
            field = build_approx_action_field(env_disp, approx_result.weights, tc_cfg,
                                              fixed_vx=vx_choice, fixed_vy=vy_choice, grid_size=af_size)
            action_names = [a.name for a in VelocityAction]
            field_labels = np.vectorize(lambda x: action_names[x][:4])(field)
            st.dataframe(field_labels, use_container_width=True)

        # Tab 5: Value Surface
        with t5:
            vs_vx = st.selectbox("Vx", [-1, 0, 1], index=1, key="vs_vx")
            vs_vy = st.selectbox("Vy", [-1, 0, 1], index=1, key="vs_vy")
            vs_size = st.slider("Grid Resolution", 5, 40, 20, key="vs_size")
            env_disp = make_approx_env(start_mode=StartMode.FIXED)
            surface = build_approx_value_surface(env_disp, approx_result.weights, tc_cfg,
                                                 fixed_vx=vs_vx, fixed_vy=vs_vy, grid_size=vs_size)
            st.dataframe(np.round(surface, 2), use_container_width=True)

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
                st.info("Run an evaluation from the sidebar.")

        # Tab 7: Experiments
        with t7:
            st.subheader("Hyperparameter Experiments")
            st.markdown("**Stage A — One Factor at a Time**")
            if st.button("Run Stage A Screening", key="approx_stage_a"):
                with st.spinner("Running Stage A screening..."):
                    stage_a = run_screening_stage_a(n_episodes=200, eval_episodes=20, seed=train_seed)
                    st.dataframe(stage_a, use_container_width=True)
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
                    st.dataframe(conf, use_container_width=True)

# ============================================================
# MODE: Algorithm Comparison
# ============================================================
elif st.session_state.mode == "Algorithm Comparison":
    st.header("SARSA vs Q-Learning — Controlled Comparison")

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

        comp_key = (comp_episodes, comp_alpha, comp_gamma, comp_decay, comp_seeds, comp_eval_ep)

        if st.button("Run Comparison", type="primary", key="comp_run"):
            with st.spinner("Running matched comparison..."):
                matched = run_matched_comparison(
                    alpha=comp_alpha, gamma=comp_gamma,
                    episodes=comp_episodes, epsilon_decay=comp_decay,
                    training_seeds=list(range(comp_seeds)),
                    eval_seeds=range(comp_eval_ep),
                )
                st.session_state.comp_matched = matched

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
                st.session_state.comp_tuned = tuned

            save_comparison(matched, tuned)
            st.session_state.comp_key = comp_key
            st.rerun()

    comp_matched = st.session_state.comp_matched
    comp_tuned = st.session_state.comp_tuned

    if comp_matched is not None and comp_tuned is not None:
        # --- Matched comparison summary ---
        st.subheader("Comparison A — Matched Parameters")
        sarsa_m, q_m = comp_matched

        data = []
        for i in range(len(sarsa_m)):
            s = sarsa_m[i]
            q = q_m[i]
            data.append({
                "Seed": s.seed,
                "SARSA SR": f"{s.success_rate:.1%}",
                "Q-Learn SR": f"{q.success_rate:.1%}",
                "SARSA Return": f"{s.mean_return:.1f}",
                "Q-Learn Return": f"{q.mean_return:.1f}",
                "SARSA Steps": f"{s.mean_steps:.1f}",
                "Q-Learn Steps": f"{q.mean_steps:.1f}",
                "SARSA Traps": s.total_traps,
                "Q-Learn Traps": q.total_traps,
            })
        st.dataframe(data, use_container_width=True)

        s_sr = [r.success_rate for r in sarsa_m]
        q_sr = [r.success_rate for r in q_m]
        s_ret = [r.mean_return for r in sarsa_m]
        q_ret = [r.mean_return for r in q_m]

        c1, c2, c3 = st.columns(3)
        c1.metric("SARSA Mean SR", f"{np.mean(s_sr):.1%}", delta=None)
        c2.metric("Q-Learn Mean SR", f"{np.mean(q_sr):.1%}", delta=None)
        c3.metric("Mean Paired SR Diff", f"{np.mean(np.array(q_sr) - np.array(s_sr)):.1%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("SARSA Mean Return", f"{np.mean(s_ret):.1f}")
        c2.metric("Q-Learn Mean Return", f"{np.mean(q_ret):.1f}")
        c3.metric("Mean Paired Return Diff", f"{np.mean(np.array(q_ret) - np.array(s_ret)):.1f}")

        # Charts
        st.subheader("Per-Seed Success Rate")
        chart = {"SARSA": s_sr, "Q-Learning": q_sr}
        st.bar_chart(chart)

        st.subheader("Per-Seed Mean Return")
        st.bar_chart({"SARSA": s_ret, "Q-Learning": q_ret})

        # --- Tuned comparison ---
        st.subheader("Comparison B — Tuned Models")
        tuned_data = []
        for r in comp_tuned:
            tuned_data.append({
                "Algorithm": r.algorithm,
                "Config": str(r.config),
                "Mean SR": f"{r.success_rate_mean:.1%}",
                "SR Std": f"{r.success_rate_std:.2%}",
                "Mean Return": f"{r.mean_return_mean:.1f}",
                "Mean Steps": f"{r.mean_steps_mean:.1f}",
                "Traps": r.total_traps,
            })
        st.dataframe(tuned_data, use_container_width=True)
    else:
        st.info("Press **Run Comparison** in the sidebar to start.")
