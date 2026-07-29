"""Game-style home page with room cards and campaign progression."""

import html as html_mod

import streamlit as st
from game.constants import ROOM5_BONUS_MODE
from game.html_rendering import normalize_html, render_html
from game.theme import difficulty_badge
from game.models import GameRoomState, RoomUnlockStatus, AchievementId
from game.achievements import AchievementTracker, ALL_ACHIEVEMENTS
from game.presentation import go_to_lab, go_to_mode, go_to_showcase_room


# Campaign-card metadata.  The rooms are currently all unlocked, but the status
# structure allows progression rules or best-run stats to be added later.
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
        state_description="(row, col, has_key) \u2014 92 states",
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
    GameRoomState(
        room_id="room5",
        room_name="The Obstacle Lab",
        room_index=5,
        algorithm="NumPy DQN (Replay + Target Network)",
        state_description="22-feature continuous observation",
        challenge="Seeded 0.5m obstacles \u2014 learn from local obstacle visibility",
        difficulty=5,
        status=RoomUnlockStatus(room_id="room5", unlocked=True, best_steps=None, best_return=None, best_time_s=None),
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
    "room5": (
        "The chamber now contains avoidable square obstacles. The agent observes only nearby "
        "obstacle records and learns with a replay-buffer DQN."
    ),
}


def _get_room_emoji(idx: int) -> str:
    return ["\u2744\ufe0f", "\u26a1", "\U0001f511", "\U0001f300", "\U0001f9e0"][idx - 1]


def _render_room_card_html(
    room: GameRoomState,
    *,
    card_class: str,
    emoji: str,
    diff_badge: str,
    unlocked_ids: set[AchievementId],
) -> str:
    room_ach_ids = _room_achievements(room.room_id)
    ach_icons = "".join(
        f'<span style="font-size:0.8em;margin-right:4px;" title="{html_mod.escape(ALL_ACHIEVEMENTS[a].name)}">{ALL_ACHIEVEMENTS[a].emoji}</span>'
        for a in room_ach_ids if a in unlocked_ids
    )

    best_stats = ""
    if room.status.best_steps is not None:
        best_stats = (
            f'<div class="hud-row" style="margin-top:6px;">'
            f'<div class="hud-item"><div class="hud-label">Best Steps</div>'
            f'<div class="hud-value">{room.status.best_steps}</div></div>'
            f'<div class="hud-item"><div class="hud-label">Best Return</div>'
            f'<div class="hud-value">{room.status.best_return:.1f}</div></div>'
            f'</div>'
        )

    achievements = f'<div style="margin-top:6px;">{ach_icons}</div>' if ach_icons else ""

    return normalize_html(
        f'<div class="{card_class}">'
        f'<div style="display:flex;justify-content:space-between;align-items:start;">'
        f'<div>'
        f'<div class="room-card-title">{emoji} {html_mod.escape(room.room_name)}</div>'
        f'<div style="font-size:0.8em;color:#90a4ae;">{html_mod.escape(room.algorithm)}</div>'
        f'</div>'
        f'<div>{diff_badge}</div>'
        f'</div>'
        f'<div style="font-size:0.8em;color:#aaa;margin:6px 0;">{html_mod.escape(room.state_description)}</div>'
        f'<div style="font-size:0.8em;color:#b0bec5;font-style:italic;">{html_mod.escape(room.challenge)}</div>'
        f'{best_stats}'
        f'{achievements}'
        f'</div>'
    )


def render_home_page():
    # Global styles are injected by app.py — do not duplicate here

    # This page is the game-style entry point.  The Learning Laboratory remains
    # available separately for training curves and detailed analysis.
    render_html("""
    <div class="home-title">
        <h1>RL ESCAPE ROOM</h1>
        <p>Can five learning agents escape five increasingly difficult chambers?</p>
    </div>
    """)

    render_html("""
    <div style="text-align:center;padding:10px 0 20px;color:#616161;font-size:0.85em;">
        Known Model \u2192 Unknown Model \u2192 Memory/State Extension \u2192 Continuous Control \u2192 Obstacle DQN
    </div>
    """)

    render_html("""
    <div style="display:flex;justify-content:center;gap:8px;margin-bottom:24px;">
        <span style="background:#1b5e20;color:#76ff03;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 1</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#e65100;color:#ffd740;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 2</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#b71c1c;color:#ff5252;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 3</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#4a148c;color:#ea80fc;padding:2px 12px;border-radius:12px;font-size:0.75em;">Room 4</span>
        <span style="color:#444;">\u2192</span>
        <span style="background:#0f766e;color:#5eead4;padding:2px 12px;border-radius:12px;font-size:0.75em;">Bonus</span>
    </div>
    """)

    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        if st.button("Start Campaign", width="stretch", type="primary"):
            go_to_showcase_room("room1")
    with nav_col2:
        if st.button("Open Learning Laboratory", width="stretch"):
            go_to_lab(None)

    render_html("<hr style='border-color:#333;margin:24px 0;'>")

    tracker = AchievementTracker.from_session_state()
    # Achievement icons are shown on each room card only after they unlock.
    unlocked_list = tracker.get_unlocked()
    unlocked_ids = {a.id for a in unlocked_list}

    st.markdown("### Required Campaign")
    cols = st.columns(2)
    for i, room in enumerate(ROOM_DEFS[:4]):
        with cols[i % 2]:
            locked = not room.status.unlocked
            card_class = "room-card" + (" room-card-locked" if locked else "")
            emoji = _get_room_emoji(room.room_index)
            diff_badge = difficulty_badge(room.difficulty)

            render_html(_render_room_card_html(
                room,
                card_class=card_class,
                emoji=emoji,
                diff_badge=diff_badge,
                unlocked_ids=unlocked_ids,
            ))

            btn_label = "Enter Room" if not locked else "Locked"
            disabled = locked
            if st.button(btn_label, key=f"enter_{room.room_id}", disabled=disabled, width="stretch"):
                go_to_showcase_room(room.room_id)

    st.markdown("### Bonus Room - Dynamic Obstacles")
    bonus_room = ROOM_DEFS[4]
    render_html(_render_room_card_html(
        bonus_room,
        card_class="room-card",
        emoji=_get_room_emoji(bonus_room.room_index),
        diff_badge=difficulty_badge(bonus_room.difficulty),
        unlocked_ids=unlocked_ids,
    ))
    if st.button("Open Bonus Room", key="enter_bonus_room5", width="stretch"):
        go_to_mode(ROOM5_BONUS_MODE, view="bonus")

    render_html("<hr style='border-color:#333;margin:24px 0;'>")
    st.markdown("### Achievements")
    if unlocked_list:
        ach_html = "".join(
            f'<span style="display:inline-block;margin:4px 8px 4px 0;padding:4px 12px;'
            f'background:rgba(255,215,64,0.1);border:1px solid #ffd740;border-radius:16px;'
            f'font-size:0.85em;">{a.emoji} {a.name}</span>'
            for a in unlocked_list
        )
        render_html(f'<div>{ach_html}</div>')
    else:
        render_html('<div style="color:#616161;font-size:0.9em;">Complete rooms to unlock achievements.</div>')


def _room_achievements(room_id: str) -> list[AchievementId]:
    # Used to decide which achievement icons belong on each room card.
    mapping = {
        "room1": [AchievementId.FIRST_ESCAPE, AchievementId.ICE_MASTER, AchievementId.SPEED_RUNNER],
        "room2": [AchievementId.FIRST_ESCAPE, AchievementId.LASER_DODGER, AchievementId.SPEED_RUNNER],
        "room3": [AchievementId.FIRST_ESCAPE, AchievementId.VAULT_EXPERT, AchievementId.SPEED_RUNNER],
        "room4": [AchievementId.FIRST_ESCAPE, AchievementId.MOMENTUM_MASTER, AchievementId.SPEED_RUNNER],
        "room5": [AchievementId.FIRST_ESCAPE, AchievementId.SPEED_RUNNER],
    }
    return mapping.get(room_id, [AchievementId.FIRST_ESCAPE])
