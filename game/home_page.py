"""Game-style home page with room cards and campaign progression."""

import html as html_mod

import streamlit as st
from game.theme import difficulty_badge
from game.models import GameRoomState, RoomUnlockStatus, Achievement, AchievementId
from game.achievements import AchievementTracker, ALL_ACHIEVEMENTS


ROOM_DEFS = [
    GameRoomState(
        room_id="room1",
        room_name="The Frozen Maze",
        room_index=1,
        algorithm="Value Iteration",
        state_description="(row, col) \u2014 100 discrete states",
        challenge="Stochastic transitions \u2014 agent may slip on ice",
        difficulty=1,
        status=RoomUnlockStatus(room_id="room1", unlocked=True, best_steps=None, best_return=None, best_time_s=None),
    ),
    GameRoomState(
        room_id="room2",
        room_name="The Laser Corridor",
        room_index=2,
        algorithm="SARSA (On-Policy TD)",
        state_description="(row, col) \u2014 100 discrete states",
        challenge="Unknown model with traps \u2014 learn risk-aware behaviour",
        difficulty=2,
        status=RoomUnlockStatus(room_id="room2", unlocked=True, best_steps=None, best_return=None, best_time_s=None),
    ),
    GameRoomState(
        room_id="room3",
        room_name="The Key Vault",
        room_index=3,
        algorithm="Q-Learning (Off-Policy TD)",
        state_description="(row, col, has_key) \u2014 200 states",
        challenge="Key collection \u2014 state must include key possession",
        difficulty=3,
        status=RoomUnlockStatus(room_id="room3", unlocked=True, best_steps=None, best_return=None, best_time_s=None),
    ),
    GameRoomState(
        room_id="room4",
        room_name="The Momentum Chamber",
        room_index=4,
        algorithm="Approximate SARSA (Tile Coding)",
        state_description="(x, y, vx, vy) \u2014 continuous",
        challenge="Continuous state space \u2014 velocity control with function approximation",
        difficulty=4,
        status=RoomUnlockStatus(room_id="room4", unlocked=True, best_steps=None, best_return=None, best_time_s=None),
    ),
]

ROOM_NARRATIVES = {
    "room1": (
        "The agent wakes inside a frozen chamber. The floor is unstable, and intended movements "
        "may cause sideways slips. Because the full model is known, it calculates the optimal "
        "escape policy before moving."
    ),
    "room2": (
        "The map is unknown. The agent must learn from experience while deciding whether to risk "
        "a short path through laser traps or take a safer route."
    ),
    "room3": (
        "The exit is locked. The agent must remember whether it has collected the key because "
        "the same location has different meaning before and after collection."
    ),
    "room4": (
        "The discrete grid disappears. The agent must control velocity in continuous space and "
        "generalize from overlapping tile-coded features."
    ),
}


def _get_room_emoji(idx: int) -> str:
    return ["\u2744\ufe0f", "\u26a1", "\U0001f511", "\U0001f300"][idx - 1]


def render_home_page():
    # Global styles are injected by app.py — do not duplicate here

    st.markdown("""
    <div class="home-title">
        <h1>RL ESCAPE ROOM</h1>
        <p>Can four learning agents escape four increasingly difficult chambers?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px;color:#616161;font-size:0.85em;">
        Known Model \u2192 Unknown Model \u2192 Memory/State Extension \u2192 Continuous Control
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;justify-content:center;gap:8px;margin-bottom:24px;">
        <span style="background:#1b5e20;color:#76ff03;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 1</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#e65100;color:#ffd740;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 2</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#b71c1c;color:#ff5252;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 3</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#4a148c;color:#ea80fc;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 4</span>
    </div>
    """, unsafe_allow_html=True)

    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        if st.button("\U0001f3ae Enter the Escape Room", use_container_width=True, type="primary"):
            st.session_state.mode = "\U0001f3ae Escape Room Showcase"
            st.session_state.game_room = None
            st.rerun()
    with nav_col2:
        if st.button("\U0001f52c View Learning Laboratory", use_container_width=True):
            st.session_state.mode = "Home"
            st.rerun()

    st.markdown("<hr style='border-color:#333;margin:24px 0;'>", unsafe_allow_html=True)

    tracker = AchievementTracker.from_session_state()
    unlocked_list = tracker.get_unlocked()
    unlocked_ids = {a.id for a in unlocked_list}

    cols = st.columns(2)
    for i, room in enumerate(ROOM_DEFS):
        with cols[i % 2]:
            locked = not room.status.unlocked
            card_class = "room-card" + (" room-card-locked" if locked else "")
            emoji = _get_room_emoji(room.room_index)
            diff_badge = difficulty_badge(room.difficulty)

            room_ach_ids = _room_achievements(room.room_id)
            ach_str = "".join(
                f'<span style="font-size:0.8em;margin-right:4px;" title="{html_mod.escape(ALL_ACHIEVEMENTS[a].name)}">{ALL_ACHIEVEMENTS[a].emoji}</span>'
                for a in room_ach_ids if a in unlocked_ids
            )

            best_str = ""
            if room.status.best_steps is not None:
                best_str = (
                    f'<div class="hud-row" style="margin-top:6px;">'
                    f'<div class="hud-item"><div class="hud-label">Best Steps</div>'
                    f'<div class="hud-value">{room.status.best_steps}</div></div>'
                    f'<div class="hud-item"><div class="hud-label">Best Return</div>'
                    f'<div class="hud-value">{room.status.best_return:.1f}</div></div>'
                    f'</div>'
                )

            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex;justify-content:space-between;align-items:start;">
                    <div>
                        <div class="room-card-title">{emoji} {room.room_name}</div>
                        <div style="font-size:0.8em;color:#90a4ae;">{room.algorithm}</div>
                    </div>
                    <div>{diff_badge}</div>
                </div>
                <div style="font-size:0.8em;color:#aaa;margin:6px 0;">
                    {room.state_description}
                </div>
                <div style="font-size:0.8em;color:#b0bec5;font-style:italic;">
                    {room.challenge}
                </div>
                {best_str}
                {f'<div style="margin-top:6px;">{ach_str}</div>' if ach_str else ''}
            </div>
            """, unsafe_allow_html=True)

            btn_label = "Enter Room" if not locked else "Locked"
            disabled = locked
            if st.button(btn_label, key=f"enter_{room.room_id}", disabled=disabled, use_container_width=True):
                st.session_state.mode = "\U0001f3ae Escape Room Showcase"
                st.session_state.game_room = room.room_id
                st.rerun()

    st.markdown("<hr style='border-color:#333;margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("### Achievements")
    if unlocked_list:
        ach_html = "".join(
            f'<span style="display:inline-block;margin:4px 8px 4px 0;padding:4px 12px;'
            f'background:rgba(255,215,64,0.1);border:1px solid #ffd740;border-radius:16px;'
            f'font-size:0.85em;">{a.emoji} {a.name}</span>'
            for a in unlocked_list
        )
        st.markdown(f'<div>{ach_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#616161;font-size:0.9em;">Complete rooms to unlock achievements.</div>',
                    unsafe_allow_html=True)


def _room_achievements(room_id: str) -> list[AchievementId]:
    mapping = {
        "room1": [AchievementId.FIRST_ESCAPE, AchievementId.ICE_MASTER, AchievementId.SPEED_RUNNER],
        "room2": [AchievementId.FIRST_ESCAPE, AchievementId.LASER_DODGER, AchievementId.SPEED_RUNNER],
        "room3": [AchievementId.FIRST_ESCAPE, AchievementId.VAULT_EXPERT, AchievementId.SPEED_RUNNER],
        "room4": [AchievementId.FIRST_ESCAPE, AchievementId.MOMENTUM_MASTER, AchievementId.SPEED_RUNNER],
    }
    return mapping.get(room_id, [AchievementId.FIRST_ESCAPE])
