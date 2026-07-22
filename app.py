import streamlit as st

from core.types import Action, StepResult
from environments.room1_dp import Room1DP
from environments.room2_sarsa import Room2SARSA
from environments.room3_qlearning import Room3QLearning

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

st.set_page_config(page_title="RL Escape Room — Grid Demo", layout="wide")
st.title("RL Escape Room — Manual Grid Demonstrator")

if "env" not in st.session_state:
    st.session_state.env = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "room_key" not in st.session_state:
    st.session_state.room_key = None

with st.sidebar:
    st.header("Controls")
    room_name = st.selectbox("Room", list(ROOM_CLASSES.keys()), key="room_selector")
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
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
            if cols[i].button(label, disabled=disabled, key=f"btn_{action}"):
                result: StepResult = env.step(action)
                st.session_state.last_result = result
                st.rerun()

        st.markdown("---")
        st.markdown("**Status**")
        st.metric("Step", env.step_count)
        if st.session_state.last_result is not None:
            r = st.session_state.last_result
            st.metric("Last Reward", f"{r.reward:.1f}")
            if isinstance(r.info, dict):
                st.markdown(f"**Requested:** {Action(r.info.get('requested_action', '?')).name}")
                st.markdown(f"**Effective:** {Action(r.info.get('effective_action', '?')).name}")
                st.markdown(f"**Slipped:** {r.info.get('slipped', False)}")
                st.markdown(f"**Collision:** {r.info.get('collision', '—')}")
                st.markdown(f"**Event:** {r.info.get('event', '—')}")
        if env.is_done:
            if env._terminated:
                st.success("EXIT REACHED — episode terminated")
            elif env._truncated:
                st.error("TIMEOUT — episode truncated")

if env is not None:
    st.subheader("Grid")
    grid_str = env.render_ansi()
    st.code(grid_str, language="text")
