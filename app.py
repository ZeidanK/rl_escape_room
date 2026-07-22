import streamlit as st

from config.rooms import ROOM_SPECS

st.set_page_config(page_title="RL Escape Room — Phase 1", layout="centered")

st.title("Reinforcement Learning Escape Room")
st.markdown("---")
st.info("This application is in **Phase 1** (scaffold and design). No algorithms have been implemented yet. Training controls will appear in Phase 2+.")

st.header("Planned Rooms")

room_data = [
    (spec.name, spec.algorithm, spec.state_description, spec.action_description)
    for spec in ROOM_SPECS.values()
]

st.table(
    {
        "Room": [r[0] for r in room_data],
        "Algorithm": [r[1] for r in room_data],
        "State": [r[2] for r in room_data],
        "Actions": [r[3] for r in room_data],
    }
)

st.markdown("---")
st.subheader("Configuration")

st.code(
    """
import streamlit run app.py
""",
    language="bash",
)

room_choice = st.selectbox("Select a room to view its spec:", list(ROOM_SPECS.keys()))
spec = ROOM_SPECS[room_choice]

st.markdown(f"### {spec.name}")
st.markdown(f"**Algorithm:** {spec.algorithm}")
st.markdown(f"**Kind:** {'Continuous' if spec.is_continuous else 'Grid'}")
if spec.grid_size:
    st.markdown(f"**Grid size:** {spec.grid_size[0]}×{spec.grid_size[1]}")
if spec.continuous_size:
    st.markdown(f"**Room size:** {spec.continuous_size[0]}×{spec.continuous_size[1]} metres")
    st.markdown(f"**Time step:** {spec.dt}s")
    st.markdown(f"**Velocity values:** {spec.velocity_values}")
st.markdown(f"**State:** {spec.state_description}")
st.markdown(f"**Actions:** {spec.action_description}")
st.markdown(f"**Description:** {spec.description}")

st.markdown("#### Reward Defaults")
r = spec.rewards
st.table({
    "Event": ["Step", "Exit", "Wall", "Trap", "Key", "Locked exit (no key)", "Time bonus scale"],
    "Default value": [r.step_penalty, r.exit_reward, r.wall_penalty, r.trap_penalty, r.key_reward, r.locked_exit_penalty, r.time_bonus_scale],
})
