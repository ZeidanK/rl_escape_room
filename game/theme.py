"""Room themes and global Streamlit CSS for the showcase UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoomTheme:
    # Central style palette for one room.  Canvas rendering, HUDs, and home
    # cards read from this instead of hardcoding colors in each view.
    room_id: str
    name: str
    primary: str
    secondary: str
    accent: str
    bg_dark: str
    bg_medium: str
    bg_light: str
    text: str
    text_muted: str
    cell_empty: str
    cell_wall: str
    cell_start: str
    cell_exit: str
    cell_slippery: str
    cell_trap: str
    cell_key: str
    cell_locked: str
    agent_color: str
    success_color: str
    failure_color: str
    css_custom: str = ""


ROOM_THEMES: dict[str, RoomTheme] = {
    # One theme per room keeps the showcase visually distinct while preserving
    # a shared layout/component system.
    "room1": RoomTheme(
        room_id="room1",
        name="Frozen Maze",
        primary="#4fc3f7",
        secondary="#0288d1",
        accent="#e1f5fe",
        bg_dark="#0d1b2a",
        bg_medium="#1b2838",
        bg_light="#243447",
        text="#e0e0e0",
        text_muted="#90a4ae",
        cell_empty="#1a3a5c",
        cell_wall="#455a64",
        cell_start="#4fc3f7",
        cell_exit="#76ff03",
        cell_slippery="#81d4fa",
        cell_trap="#ff5252",
        cell_key="#ffd740",
        cell_locked="#ff6d00",
        agent_color="#29b6f6",
        success_color="#76ff03",
        failure_color="#ff1744",
        css_custom="""
.room1-cell-slippery { background: linear-gradient(135deg, #81d4fa 0%, #4fc3f7 100%); }
.room1-cell-empty { background: radial-gradient(circle at 30% 30%, #1a3a5c, #0d2137); }
.room1-cell-wall { background: repeating-linear-gradient(0deg, #546e7a, #546e7a 2px, #455a64 2px, #455a64 4px); }
""",
    ),
    "room2": RoomTheme(
        room_id="room2",
        name="Laser Corridor",
        primary="#ef5350",
        secondary="#c62828",
        accent="#ffebee",
        bg_dark="#1a0a0a",
        bg_medium="#2d1212",
        bg_light="#3d1a1a",
        text="#e0e0e0",
        text_muted="#90a4ae",
        cell_empty="#2a1a1a",
        cell_wall="#616161",
        cell_start="#ef5350",
        cell_exit="#76ff03",
        cell_slippery="#bdbdbd",
        cell_trap="#ff1744",
        cell_key="#ffd740",
        cell_locked="#ff6d00",
        agent_color="#ff5252",
        success_color="#76ff03",
        failure_color="#d50000",
        css_custom="""
.room2-cell-trap { animation: pulse-red 1.5s ease-in-out infinite; }
@keyframes pulse-red { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
.room2-cell-slippery { background: linear-gradient(180deg, #9e9e9e, #e0e0e0, #9e9e9e); }
""",
    ),
    "room3": RoomTheme(
        room_id="room3",
        name="Key Vault",
        primary="#ffd740",
        secondary="#ff8f00",
        accent="#fff8e1",
        bg_dark="#1a1200",
        bg_medium="#2a1e00",
        bg_light="#3a2a00",
        text="#e0e0e0",
        text_muted="#b0a070",
        cell_empty="#2a2010",
        cell_wall="#6d4c41",
        cell_start="#ffd740",
        cell_exit="#76ff03",
        cell_slippery="#a0a0a0",
        cell_trap="#ff5252",
        cell_key="#ffd740",
        cell_locked="#ff6d00",
        agent_color="#ffc107",
        success_color="#76ff03",
        failure_color="#d50000",
        css_custom="""
.room3-cell-key { animation: glow-gold 2s ease-in-out infinite; }
@keyframes glow-gold { 0%, 100% { filter: brightness(1); } 50% { filter: brightness(1.5); } }
.room3-cell-locked { background: repeating-linear-gradient(45deg, #795548, #795548 4px, #6d4c41 4px, #6d4c41 8px); }
""",
    ),
    "room4": RoomTheme(
        room_id="room4",
        name="Momentum Chamber",
        primary="#7c4dff",
        secondary="#651fff",
        accent="#f3e5f5",
        bg_dark="#0d001a",
        bg_medium="#1a002e",
        bg_light="#260042",
        text="#e0e0e0",
        text_muted="#b39ddb",
        cell_empty="#1a002e",
        cell_wall="#4a0072",
        cell_start="#7c4dff",
        cell_exit="#76ff03",
        cell_slippery="#7c4dff",
        cell_trap="#ff5252",
        cell_key="#ffd740",
        cell_locked="#ff6d00",
        agent_color="#b388ff",
        success_color="#76ff03",
        failure_color="#d50000",
        css_custom="""
.room4-agent { filter: drop-shadow(0 0 6px #7c4dff); }
""",
    ),
    "room5": RoomTheme(
        room_id="room5",
        name="Obstacle Lab",
        primary="#14b8a6",
        secondary="#0f766e",
        accent="#ccfbf1",
        bg_dark="#042f2e",
        bg_medium="#0f3f3c",
        bg_light="#115e59",
        text="#e0f2f1",
        text_muted="#99f6e4",
        cell_empty="#0f172a",
        cell_wall="#475569",
        cell_start="#38bdf8",
        cell_exit="#22c55e",
        cell_slippery="#14b8a6",
        cell_trap="#f97316",
        cell_key="#facc15",
        cell_locked="#ef4444",
        agent_color="#67e8f9",
        success_color="#86efac",
        failure_color="#fb7185",
        css_custom="""
.room5-obstacle { filter: drop-shadow(0 0 5px rgba(20,184,166,0.45)); }
""",
    ),
}


GLOBAL_CSS = """
<style>
/* Reset & base */
.stApp { background-color: #0a0a0a !important; color: #e0e0e0; }
.stSidebar { background-color: #111 !important; }

/* Game container */
.game-container { display: flex; gap: 20px; padding: 10px; font-family: 'Segoe UI', sans-serif; }
.game-main { flex: 1; min-width: 0; }
.game-sidebar-right { width: 320px; flex-shrink: 0; }

/* Grid canvas */
.grid-canvas { border: 2px solid #333; border-radius: 8px; overflow: hidden; background: #111; display: inline-block; max-width: 100%; height: auto; }

/* Responsive grid container */
.grid-container { width: 100%; max-width: 500px; }

/* HUD */
.game-hud { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 15px; }
.game-hud-title { font-size: 1.3em; font-weight: 700; margin: 0 0 4px 0; }
.game-hud-subtitle { font-size: 0.85em; color: #90a4ae; margin: 0 0 10px 0; }
.hud-row { display: flex; gap: 12px; flex-wrap: wrap; }
.hud-item { background: rgba(255,255,255,0.05); border-radius: 6px; padding: 6px 12px; min-width: 80px; }
.hud-label { font-size: 0.7em; color: #90a4ae; text-transform: uppercase; letter-spacing: 0.5px; }
.hud-value { font-size: 1em; font-weight: 600; margin-top: 2px; }

/* Status badges */
.badge-success { background: #1b5e20; color: #76ff03; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-failure { background: #b71c1c; color: #ff5252; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-timeout { background: #e65100; color: #ffd740; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-slip { background: #01579b; color: #81d4fa; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-trap { background: #b71c1c; color: #ff5252; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-key { background: #f57f17; color: #ffd740; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
.badge-collision { background: #4a148c; color: #ea80fc; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }

/* Replay controls */
.replay-bar { background: #1a1a2e; border: 1px solid #333; border-radius: 10px; padding: 12px 15px; margin-top: 15px; }
.replay-btn { background: rgba(255,255,255,0.1); border: 1px solid #444; color: #e0e0e0; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 1em; }
.replay-btn:hover { background: rgba(255,255,255,0.2); }
.replay-btn.active { background: #1976d2; border-color: #42a5f5; }
.replay-btn:disabled { opacity: 0.4; cursor: default; }
.replay-stage { background: rgba(255,255,255,0.05); border-radius: 6px; padding: 4px 10px; font-size: 0.85em; }
.replay-speed { background: rgba(255,255,255,0.05); border-radius: 6px; padding: 4px 10px; font-size: 0.85em; }

/* Home page */
.home-title { text-align: center; padding: 40px 20px 20px; }
.home-title h1 { font-size: 3em; margin: 0; background: linear-gradient(135deg, #4fc3f7, #7c4dff); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
.home-title p { color: #90a4ae; font-size: 1.1em; margin-top: 8px; }
.home-subtitle { color: #616161; font-size: 0.9em; margin: 0; }

/* Room card */
.room-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #333; border-radius: 12px; padding: 16px; margin-bottom: 16px; cursor: pointer; transition: all 0.2s; }
.room-card:hover { border-color: #555; transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.room-card-locked { opacity: 0.5; cursor: default; }
.room-card-locked:hover { transform: none; box-shadow: none; border-color: #333; }
.room-card-title { font-size: 1.1em; font-weight: 700; margin: 0 0 4px 0; }
.room-card-difficulty { display: inline-block; padding: 1px 8px; border-radius: 8px; font-size: 0.75em; font-weight: 600; }
.diff-1 { background: #1b5e20; color: #76ff03; }
.diff-2 { background: #e65100; color: #ffd740; }
.diff-3 { background: #b71c1c; color: #ff5252; }
.diff-4 { background: #4a148c; color: #ea80fc; }

/* Room transition overlay */
.transition-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 9999; }
.transition-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 2px solid #333; border-radius: 16px; padding: 40px; text-align: center; max-width: 500px; }
.transition-card h2 { font-size: 2em; margin: 0 0 16px; }
.transition-card .stats { display: flex; gap: 20px; justify-content: center; margin: 16px 0; }
.transition-card .stat-item { text-align: center; }
.transition-card .stat-value { font-size: 1.5em; font-weight: 700; }
.transition-card .stat-label { font-size: 0.8em; color: #90a4ae; }
.transition-card .btn-continue { background: #1976d2; color: white; border: none; border-radius: 8px; padding: 10px 30px; font-size: 1em; cursor: pointer; margin-top: 16px; }
.transition-card .btn-continue:hover { background: #1565c0; }

/* Achievement toast */
.achievement-toast { position: fixed; top: 20px; right: 20px; background: linear-gradient(135deg, #1a1a2e, #2a1a00); border: 1px solid #ffd740; border-radius: 10px; padding: 12px 20px; z-index: 9998; animation: slide-in 0.3s ease-out; }
@keyframes slide-in { from { transform: translateX(100px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

/* Explain panel */
.explain-panel { background: #1a1a2e; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-top: 15px; }
.explain-panel h4 { margin: 0 0 8px; color: #4fc3f7; }
.explain-panel table { width: 100%; border-collapse: collapse; }
.explain-panel td { padding: 4px 8px; border-bottom: 1px solid #222; }
.explain-panel .selected { color: #4fc3f7; font-weight: 700; }

/* Toggle group */
.toggle-group { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
.toggle-btn { background: rgba(255,255,255,0.08); border: 1px solid #444; color: #aaa; border-radius: 6px; padding: 5px 12px; cursor: pointer; font-size: 0.8em; }
.toggle-btn.active { background: rgba(79,195,247,0.2); border-color: #4fc3f7; color: #4fc3f7; }
.toggle-btn:hover { background: rgba(255,255,255,0.15); }

/* Comparison theater */
.comparison-container { display: flex; gap: 20px; }
.comparison-side { flex: 1; text-align: center; }

/* Slip indicator */
.slip-indicator { background: rgba(79,195,247,0.15); border: 1px solid #4fc3f7; border-radius: 8px; padding: 10px; margin: 8px 0; text-align: center; }
.slip-indicator .intended { color: #90a4ae; font-size: 0.85em; }
.slip-indicator .actual { color: #4fc3f7; font-size: 1.1em; font-weight: 700; }
.slip-indicator .cause { color: #81d4fa; font-size: 0.8em; margin-top: 4px; }

/* Policy arrow overlay */
.policy-arrow { font-size: 16px; font-weight: 700; text-shadow: 0 0 4px rgba(0,0,0,0.8); pointer-events: none; }

/* Narrative box */
.narrative-box { background: linear-gradient(135deg, #1a1a2e, #16213e); border-left: 4px solid #4fc3f7; border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 15px; font-style: italic; color: #b0bec5; }

/* Legend */
.game-legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 8px; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 0.8em; color: #aaa; }
.legend-swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #444; }
</style>
"""


def render_global_styles() -> str:
    # Streamlit accepts CSS through markdown, so this returns one combined
    # style string that app.py injects once per page.
    room_styles = "\n".join(
        f"<style>\n{t.css_custom}\n</style>" for t in ROOM_THEMES.values()
    )
    return GLOBAL_CSS + room_styles


def get_theme(room_id: str) -> RoomTheme:
    # Default to room1 so an unknown room id still renders instead of failing.
    return ROOM_THEMES.get(room_id, ROOM_THEMES["room1"])


def difficulty_badge(difficulty: int) -> str:
    return f'<span class="room-card-difficulty diff-{difficulty}">{"★" * difficulty}</span>'
