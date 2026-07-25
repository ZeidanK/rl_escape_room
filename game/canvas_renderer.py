"""SVG-based canvas renderer for grid-based and continuous game views."""

from collections.abc import Mapping
from html import escape
from typing import Any

import numpy as np
from core.types import (
    Action,
    CellType,
    ContinuousRolloutResult,
    Position,
    VELOCITY_BY_ACTION,
    VelocityAction,
)
from game.theme import RoomTheme, get_theme, GLOBAL_CSS, ROOM_THEMES

# Shared SVG symbols for grid policies and cell labels.
ARROW_CHARS: dict[Action, str] = {
    Action.UP: "\u2191",
    Action.RIGHT: "\u2192",
    Action.DOWN: "\u2193",
    Action.LEFT: "\u2190",
}

CELL_LABELS: dict[CellType, str] = {
    CellType.WALL: "",
    CellType.START: "S",
    CellType.EXIT: "G",
    CellType.SLIPPERY: "",
    CellType.TRAP: "",
    CellType.KEY: "K",
    CellType.LOCKED_EXIT: "L",
    CellType.EMPTY: "",
}

VELOCITY_ARROW_CHARS: dict[tuple[int, int], str] = {
    (0, 1): "\u2191",
    (1, 1): "\u2197",
    (1, 0): "\u2192",
    (1, -1): "\u2198",
    (0, -1): "\u2193",
    (-1, -1): "\u2199",
    (-1, 0): "\u2190",
    (-1, 1): "\u2196",
    (0, 0): "\u25cf",
}


def _svg_text(value: object) -> str:
    return escape(str(value), quote=True)


def _coerce_action_arrow(action: Any) -> str | None:
    if action is None:
        return None
    try:
        action_enum = action if isinstance(action, Action) else Action(int(action))
    except (TypeError, ValueError):
        return None
    return ARROW_CHARS.get(action_enum)


def _coerce_velocity(action: Any) -> tuple[int, int]:
    try:
        action_enum = action if isinstance(action, VelocityAction) else VelocityAction(int(action))
    except (TypeError, ValueError):
        return (0, 0)
    return VELOCITY_BY_ACTION[action_enum]


def _velocity_arrow(vx: int, vy: int) -> str:
    return VELOCITY_ARROW_CHARS.get((int(vx), int(vy)), "\u25cf")


def _room_point(
    x: float,
    y: float,
    *,
    margin: float,
    width: float,
    height: float,
    room_width: float,
    room_height: float,
) -> tuple[float, float]:
    sx = (width - 2 * margin) / room_width
    sy = (height - 2 * margin) / room_height
    return margin + x * sx, height - margin - y * sy


def _cell_bg_color(cell: CellType, theme: RoomTheme) -> str:
    mapping = {
        CellType.EMPTY: theme.cell_empty,
        CellType.WALL: theme.cell_wall,
        CellType.START: theme.cell_empty,
        CellType.EXIT: theme.cell_exit,
        CellType.SLIPPERY: theme.cell_slippery,
        CellType.TRAP: theme.cell_trap,
        CellType.KEY: theme.cell_key,
        CellType.LOCKED_EXIT: theme.cell_locked,
    }
    return mapping.get(cell, theme.cell_empty)


def _render_svg_cell(
    x: int, y: int, size: int,
    cell: CellType,
    theme: RoomTheme,
    agent_here: bool,
    agent_emoji: str,
    policy_arrow: str | None,
    show_values: bool,
    value: float | None,
    show_policy: bool,
    slip_effect: bool,
    show_label: bool,
    drop_shadow_id: str = "",
    is_locked: bool = True,
    has_key: bool = False,
) -> list[str]:
    # Render one grid cell as SVG fragments.  The caller assembles all cells
    # into a full board, which keeps room styling centralized here.
    lines: list[str] = []
    bg = _cell_bg_color(cell, theme) if not slip_effect else "#4fc3f7"
    rx = 3

    extra_cls = ""
    if cell == CellType.TRAP:
        extra_cls = f' class="{theme.room_id}-cell-trap"'
    elif cell == CellType.SLIPPERY:
        extra_cls = f' class="{theme.room_id}-cell-slippery"'
    elif cell == CellType.KEY:
        extra_cls = f' class="{theme.room_id}-cell-key"'
    elif cell == CellType.LOCKED_EXIT:
        extra_cls = f' class="{theme.room_id}-cell-locked"'

    lines.append(
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" '
        f'fill="{bg}" rx="{rx}" stroke="#333" stroke-width="0.5"{extra_cls}/>'
    )

    if cell == CellType.SLIPPERY and not agent_here:
        cs = size // 6
        cx = x + size // 2
        cy = y + size // 2
        lines.append(
            f'<text x="{cx}" y="{cy + cs // 2}" text-anchor="middle" '
            f'fill="rgba(255,255,255,0.3)" font-size="{cs}" font-weight="200">*</text>'
        )

    if cell == CellType.TRAP and not agent_here:
        cx = x + size // 2
        cy = y + size // 2
        r = size // 4
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
            f'stroke="rgba(255,0,0,0.3)" stroke-width="1.5" />'
        )
        lines.append(
            f'<line x1="{cx - r + 2}" y1="{cy - r + 2}" x2="{cx + r - 2}" y2="{cy + r - 2}" '
            f'stroke="rgba(255,0,0,0.3)" stroke-width="1.5"/>'
        )
        lines.append(
            f'<line x1="{cx - r + 2}" y1="{cy + r - 2}" x2="{cx + r - 2}" y2="{cy - r + 2}" '
            f'stroke="rgba(255,0,0,0.3)" stroke-width="1.5"/>'
        )

    if cell == CellType.START and not show_label:
        cx = x + size // 2
        cy = y + size // 2
        lines.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'fill="{theme.cell_start}" font-size="{size // 3}" font-weight="700">S</text>'
        )

    if cell == CellType.EXIT:
        cx = x + size // 2
        cy = y + size // 2
        lines.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{size // 3}" ry="{size // 3}" '
            f'fill="rgba(118,255,3,0.15)" stroke="{theme.cell_exit}" stroke-width="1.5"/>'
        )
        if not show_label:
            lines.append(
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                f'fill="{theme.cell_exit}" font-size="{size // 4}" font-weight="700">G</text>'
            )

    if cell == CellType.KEY:
        cx = x + size // 2
        cy = y + size // 2
        lines.append(
            f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" '
            f'fill="{theme.cell_key}" font-size="{size // 2}">\U0001f511</text>'
        )

    if cell == CellType.LOCKED_EXIT:
        cx = x + size // 2
        cy = y + size // 2
        if is_locked:
            lines.append(
                f'<rect x="{cx - size // 5}" y="{cy - size // 4}" '
                f'width="{size // 2.5}" height="{size // 2}" rx="3" '
                f'fill="none" stroke="{theme.cell_locked}" stroke-width="1.5"/>'
            )
            lines.append(
                f'<circle cx="{cx}" cy="{cy + size // 8}" r="{size // 10}" '
                f'fill="none" stroke="{theme.cell_locked}" stroke-width="1"/>'
            )
            lines.append(
                f'<rect x="{cx - size // 12}" y="{cy - size // 4}" '
                f'width="{size // 6}" height="{size // 6}" '
                f'fill="{theme.cell_locked}" rx="1"/>'
            )
        else:
            lines.append(
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                f'fill="{theme.cell_exit}" font-size="{size // 3}" font-weight="700">\u2713</text>'
            )

    if show_policy and policy_arrow and not agent_here and cell not in (
        CellType.WALL, CellType.EXIT, CellType.LOCKED_EXIT
    ):
        cx = x + size // 2
        cy = y + size // 2
        lines.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'fill="rgba(255,255,255,0.5)" font-size="{size // 3}" '
            f'class="policy-arrow">{policy_arrow}</text>'
        )

    if show_values and value is not None and cell not in (CellType.WALL,):
        cx = x + size // 2
        cy = y + size - 3
        lines.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'fill="rgba(255,255,255,0.4)" font-size="{size // 5}">{value:.1f}</text>'
        )

    if agent_here:
        cx = x + size // 2
        cy = y + size // 2
        r = size // 3
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="{theme.agent_color}" stroke="rgba(255,255,255,0.5)" stroke-width="2" '
            f'class="{theme.room_id}-agent" filter="url(#{drop_shadow_id})" />'
        )
        lines.append(
            f'<text x="{cx}" y="{cy + r // 2}" text-anchor="middle" '
            f'fill="white" font-size="{r}">{agent_emoji}</text>'
        )
        if policy_arrow and show_policy:
            ax = cx + r + 4
            ay = cy
            lines.append(
                f'<text x="{ax}" y="{ay + 4}" text-anchor="middle" '
                f'fill="{theme.agent_color}" font-size="{size // 3}" font-weight="700">{policy_arrow}</text>'
            )

    if show_label and cell in CELL_LABELS and not agent_here:
        label = CELL_LABELS[cell]
        if label:
            cx = x + size // 2
            cy = y + size // 2
            lines.append(
                f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
                f'fill="rgba(255,255,255,0.6)" font-size="{size // 3}" font-weight="600">{label}</text>'
            )

    return lines


def render_grid_canvas(
    grid: np.ndarray,
    agent_pos: Position | None = None,
    *,
    room_id: str = "room1",
    cell_size: int = 50,
    policy: dict[Any, Any] | None = None,
    values: dict[Any, float] | None = None,
    show_policy: bool = False,
    show_values: bool = False,
    show_labels: bool = False,
    slip_effect: bool = False,
    trajectory: list[Position] | None = None,
    agent_emoji: str = "\u25d8",
    has_key: bool | None = None,
    is_locked: bool = True,
    current_cell_highlight: str | None = None,
) -> str:
    # Main renderer for Rooms 1-3.  It can layer policy arrows, value numbers,
    # agent position, and replay trajectory on top of the same grid.
    theme = get_theme(room_id)
    rows, cols = grid.shape
    width = cols * cell_size
    height = rows * cell_size

    svg_parts: list[str] = []

    drop_shadow_id = f"ds-{room_id}"
    svg_parts.append(
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="grid-canvas">'
        f'<defs><filter id="{drop_shadow_id}" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="{theme.agent_color}" flood-opacity="0.8"/>'
        f'</filter></defs>'
    )

    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(grid[r, c]))
            x = c * cell_size
            y = r * cell_size
            agent_here = agent_pos is not None and (r, c) == agent_pos

            pol_arrow = None
            if show_policy and policy is not None:
                # Room 3 policies are keyed by (row, col, has_key); Rooms 1-2
                # are keyed by just (row, col).
                if has_key is not None:
                    action = policy.get((r, c, has_key))
                else:
                    action = policy.get((r, c))
                if action is not None:
                    pol_arrow = ARROW_CHARS.get(Action(action) if not isinstance(action, Action) else action)

            val = None
            if show_values and values is not None:
                vkey = (r, c, has_key) if has_key is not None else (r, c)
                val = values.get(vkey)

            slip = slip_effect and agent_here

            svg_parts.extend(
                _render_svg_cell(
                    x, y, cell_size, cell, theme, agent_here, agent_emoji,
                    pol_arrow, show_values, val, show_policy, slip,
                    show_labels, drop_shadow_id=drop_shadow_id,
                    is_locked=is_locked, has_key=has_key or False,
                )
            )

    if trajectory:
        # Draw visited states as a faint path so replay movement is visible
        # without hiding the policy arrows underneath.
        for i, pos in enumerate(trajectory):
            if agent_pos and pos == agent_pos:
                continue
            if pos == trajectory[0]:
                continue
            x = pos[1] * cell_size + cell_size // 2
            y = pos[0] * cell_size + cell_size // 2
            alpha = 0.15 + 0.15 * (i / max(1, len(trajectory) - 1))
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="3" '
                f'fill="rgba(255,255,255,{alpha})" />'
            )

        if len(trajectory) > 1:
            pts = " ".join(
                f"{p[1] * cell_size + cell_size // 2},{p[0] * cell_size + cell_size // 2}"
                for p in trajectory
                if not (agent_pos and p == agent_pos)
            )
            svg_parts.append(
                f'<polyline points="{pts}" fill="none" '
                f'stroke="rgba(255,255,255,0.2)" stroke-width="1" stroke-dasharray="3,3"/>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_policy_grid_canvas(
    grid: np.ndarray,
    policy: Mapping[Any, Any],
    *,
    room_id: str = "room1",
    cell_size: int = 48,
    has_key: bool | None = None,
) -> str:
    """Render a polished standalone policy grid for analysis tabs."""
    theme = get_theme(room_id)
    rows, cols = grid.shape
    width = cols * cell_size
    height = rows * cell_size
    glow_id = f"policy-glow-{room_id}"

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" class="policy-grid-canvas grid-canvas" '
        f'role="img" aria-label="{_svg_text(theme.name)} policy grid" '
        f'style="max-width:{width}px;background:{theme.bg_dark};border:2px solid #333;border-radius:8px;">',
        f'<defs><filter id="{glow_id}" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="{theme.primary}" flood-opacity="0.65"/>'
        '</filter></defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="{theme.bg_dark}"/>',
    ]

    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(grid[r, c]))
            x = c * cell_size
            y = r * cell_size
            cx = x + cell_size / 2
            cy = y + cell_size / 2
            cell_class = cell.name.lower().replace("_", "-")
            unlocked_exit = cell == CellType.LOCKED_EXIT and has_key is True
            terminal = cell == CellType.EXIT or unlocked_exit
            wall = cell == CellType.WALL
            bg = theme.cell_exit if unlocked_exit else _cell_bg_color(cell, theme)
            stroke = "#111827" if wall else "rgba(255,255,255,0.16)"

            parts.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell_size - 2}" height="{cell_size - 2}" '
                f'rx="5" fill="{bg}" stroke="{stroke}" stroke-width="1" '
                f'class="policy-cell {room_id}-policy-cell policy-cell-{cell_class}"/>'
            )

            if wall:
                for offset in range(-cell_size, cell_size * 2, 12):
                    parts.append(
                        f'<line x1="{x + offset}" y1="{y + cell_size}" '
                        f'x2="{x + offset + cell_size}" y2="{y}" '
                        'stroke="rgba(255,255,255,0.08)" stroke-width="2"/>'
                    )
                continue

            if cell == CellType.SLIPPERY:
                parts.append(
                    f'<path d="M{x + 10},{y + cell_size - 12} C{x + 20},{y + cell_size - 20} '
                    f'{x + 28},{y + cell_size - 4} {x + 38},{y + cell_size - 12}" '
                    'fill="none" stroke="rgba(255,255,255,0.36)" stroke-width="2" '
                    'stroke-linecap="round"/>'
                )
            elif cell == CellType.TRAP:
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell_size * 0.25:.1f}" '
                    'fill="rgba(0,0,0,0.12)" stroke="rgba(255,255,255,0.34)" stroke-width="2"/>'
                )
                parts.append(
                    f'<line x1="{cx - 9:.1f}" y1="{cy - 9:.1f}" '
                    f'x2="{cx + 9:.1f}" y2="{cy + 9:.1f}" '
                    'stroke="rgba(255,255,255,0.38)" stroke-width="2" stroke-linecap="round"/>'
                )
                parts.append(
                    f'<line x1="{cx - 9:.1f}" y1="{cy + 9:.1f}" '
                    f'x2="{cx + 9:.1f}" y2="{cy - 9:.1f}" '
                    'stroke="rgba(255,255,255,0.38)" stroke-width="2" stroke-linecap="round"/>'
                )

            marker = ""
            if cell == CellType.START:
                marker = "S"
            elif cell == CellType.EXIT:
                marker = "G"
            elif cell == CellType.SLIPPERY:
                marker = "I"
            elif cell == CellType.TRAP:
                marker = "T"
            elif cell == CellType.KEY:
                marker = "K"
            elif cell == CellType.LOCKED_EXIT:
                marker = "G" if unlocked_exit else "L"

            policy_key = (r, c, has_key) if has_key is not None else (r, c)
            arrow = None if terminal else _coerce_action_arrow(policy.get(policy_key))

            if terminal:
                radius = cell_size * 0.31
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
                    f'fill="rgba(118,255,3,0.18)" stroke="{theme.cell_exit}" stroke-width="2" '
                    'class="policy-goal"/>'
                )

            if arrow:
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
                    'dominant-baseline="central" '
                    f'fill="{theme.accent}" font-size="{cell_size * 0.54:.1f}" font-weight="800" '
                    f'class="policy-arrow" filter="url(#{glow_id})">{_svg_text(arrow)}</text>'
                )
            elif not marker:
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
                    'fill="rgba(255,255,255,0.22)" class="policy-empty-dot"/>'
                )

            if marker:
                if arrow and not terminal:
                    badge_size = max(18, int(cell_size * 0.38))
                    parts.append(
                        f'<rect x="{x + 5}" y="{y + 5}" width="{badge_size}" height="{badge_size}" '
                        'rx="4" fill="rgba(0,0,0,0.35)" stroke="rgba(255,255,255,0.18)" '
                        'class="policy-marker-badge"/>'
                    )
                    parts.append(
                        f'<text x="{x + 5 + badge_size / 2:.1f}" y="{y + 5 + badge_size / 2:.1f}" '
                        'text-anchor="middle" dominant-baseline="central" '
                        f'fill="{theme.text}" font-size="{badge_size * 0.62:.1f}" font-weight="800" '
                        f'class="policy-marker">{_svg_text(marker)}</text>'
                    )
                else:
                    parts.append(
                        f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
                        'dominant-baseline="central" '
                        f'fill="{theme.text}" font-size="{cell_size * 0.38:.1f}" font-weight="800" '
                        f'class="policy-marker">{_svg_text(marker)}</text>'
                    )

    for i in range(cols + 1):
        x = i * cell_size
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" '
            'stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        )
    for i in range(rows + 1):
        y = i * cell_size
        parts.append(
            f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" '
            'stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_continuous_trajectory_canvas(
    env: Any,
    rollout: ContinuousRolloutResult,
    *,
    max_arrows: int = 30,
    canvas_size: int = 520,
    grid_subdivisions: int = 20,
    show_path: bool = True,
    show_arrows: bool = True,
) -> str:
    """Render a Room 4 continuous rollout as a real 2D trajectory."""
    theme = get_theme("room4")
    motion = env.motion
    room_w = float(motion.room_width_m)
    room_h = float(motion.room_height_m)
    margin = 28.0
    size = float(canvas_size)
    span = size - 2 * margin
    grid_subdivisions = max(2, int(grid_subdivisions))
    arrow_id = "room4-trajectory-arrow"

    def pt(x: float, y: float) -> tuple[float, float]:
        return _room_point(
            float(x), float(y),
            margin=margin, width=size, height=size,
            room_width=room_w, room_height=room_h,
        )

    points = [(rollout.start_state[0], rollout.start_state[1])]
    points.extend((step.next_state[0], step.next_state[1]) for step in rollout.trajectory)
    point_attr = " ".join(f"{pt(x, y)[0]:.1f},{pt(x, y)[1]:.1f}" for x, y in points)
    ex, ey = pt(*motion.exit_center)
    exit_r = float(motion.exit_radius_m) * span / room_w

    parts: list[str] = [
        f'<svg viewBox="0 0 {canvas_size} {canvas_size}" width="100%" '
        'xmlns="http://www.w3.org/2000/svg" class="continuous-trajectory-canvas grid-canvas" '
        'role="img" aria-label="Room 4 continuous trajectory" '
        f'style="max-width:{canvas_size}px;background:{theme.bg_dark};border:2px solid #333;border-radius:8px;">',
        '<defs>'
        f'<marker id="{arrow_id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{theme.accent}"/></marker>'
        '</defs>',
        f'<rect x="0" y="0" width="{canvas_size}" height="{canvas_size}" rx="8" fill="{theme.bg_dark}"/>',
        f'<rect x="{margin:.1f}" y="{margin:.1f}" width="{span:.1f}" height="{span:.1f}" '
        f'rx="6" fill="{theme.bg_medium}" stroke="{theme.primary}" stroke-opacity="0.55" stroke-width="2"/>',
    ]

    for i in range(grid_subdivisions + 1):
        offset = margin + i * span / grid_subdivisions
        opacity = "0.18" if i % 5 == 0 else "0.08"
        parts.append(
            f'<line x1="{offset:.1f}" y1="{margin:.1f}" x2="{offset:.1f}" y2="{size - margin:.1f}" '
            f'stroke="rgba(255,255,255,{opacity})" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{margin:.1f}" y1="{offset:.1f}" x2="{size - margin:.1f}" y2="{offset:.1f}" '
            f'stroke="rgba(255,255,255,{opacity})" stroke-width="1"/>'
        )

    parts.append(
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{exit_r:.1f}" '
        f'fill="{theme.cell_exit}" fill-opacity="0.20" stroke="{theme.cell_exit}" '
        'stroke-width="3" class="exit-zone"/>'
    )

    if show_path and len(points) >= 2:
        parts.append(
            f'<polyline points="{point_attr}" fill="none" stroke="{theme.agent_color}" '
            'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" '
            'stroke-opacity="0.86" class="trajectory-path"/>'
        )
        for idx, (x, y) in enumerate(points[1:-1]):
            if idx % max(1, len(points) // 24) == 0:
                px, py = pt(x, y)
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" '
                    'fill="rgba(255,255,255,0.45)" class="trajectory-point"/>'
                )

    if show_arrows and rollout.trajectory:
        interval = max(1, len(rollout.trajectory) // max(1, max_arrows))
        for idx, step in enumerate(rollout.trajectory):
            if idx % interval != 0:
                continue
            x, y = step.state[0], step.state[1]
            vx, vy = _coerce_velocity(step.requested_action)
            px, py = pt(x, y)
            if vx == 0 and vy == 0:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
                    f'fill="{theme.accent}" fill-opacity="0.7" class="stationary-marker"/>'
                )
                continue
            scale = 14.0
            parts.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px + vx * scale:.1f}" y2="{py - vy * scale:.1f}" '
                f'stroke="{theme.accent}" stroke-width="2.4" stroke-linecap="round" '
                f'marker-end="url(#{arrow_id})" class="trajectory-arrow"/>'
            )

    sx, sy = pt(rollout.start_state[0], rollout.start_state[1])
    fx, fy = pt(rollout.final_state[0], rollout.final_state[1])
    parts.append(
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="{theme.cell_start}" '
        'stroke="white" stroke-opacity="0.65" stroke-width="2" class="start-point"/>'
    )
    parts.append(
        f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="7" fill="{theme.agent_color}" '
        'stroke="white" stroke-opacity="0.65" stroke-width="2" class="end-point"/>'
    )

    for step in rollout.trajectory:
        if not step.collision:
            continue
        px, py = pt(step.next_state[0], step.next_state[1])
        parts.append(
            f'<line x1="{px - 7:.1f}" y1="{py - 7:.1f}" x2="{px + 7:.1f}" y2="{py + 7:.1f}" '
            f'stroke="{theme.failure_color}" stroke-width="3" stroke-linecap="round" class="collision-marker"/>'
        )
        parts.append(
            f'<line x1="{px - 7:.1f}" y1="{py + 7:.1f}" x2="{px + 7:.1f}" y2="{py - 7:.1f}" '
            f'stroke="{theme.failure_color}" stroke-width="3" stroke-linecap="round" class="collision-marker"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_action_field_canvas(
    env: Any,
    field: np.ndarray,
    *,
    canvas_size: int = 520,
    fixed_velocity: tuple[int, int] | None = None,
) -> str:
    """Render sampled greedy Room 4 actions as a vector field."""
    theme = get_theme("room4")
    motion = env.motion
    action_grid = np.asarray(field)
    rows, cols = action_grid.shape
    margin = 28.0
    size = float(canvas_size)
    span = size - 2 * margin
    cell_w = span / max(1, cols)
    cell_h = span / max(1, rows)
    arrow_id = "room4-field-arrow"
    title = "Greedy action field"
    if fixed_velocity is not None:
        title = f"{title} for velocity {fixed_velocity}"

    def pt(x: float, y: float) -> tuple[float, float]:
        return _room_point(
            float(x), float(y),
            margin=margin, width=size, height=size,
            room_width=float(motion.room_width_m), room_height=float(motion.room_height_m),
        )

    parts: list[str] = [
        f'<svg viewBox="0 0 {canvas_size} {canvas_size}" width="100%" '
        'xmlns="http://www.w3.org/2000/svg" class="action-field-canvas grid-canvas" '
        f'role="img" aria-label="{_svg_text(title)}" '
        f'style="max-width:{canvas_size}px;background:{theme.bg_dark};border:2px solid #333;border-radius:8px;">',
        '<defs>'
        f'<marker id="{arrow_id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{theme.accent}"/></marker>'
        '</defs>',
        f'<rect x="0" y="0" width="{canvas_size}" height="{canvas_size}" rx="8" fill="{theme.bg_dark}"/>',
        f'<rect x="{margin:.1f}" y="{margin:.1f}" width="{span:.1f}" height="{span:.1f}" '
        f'rx="6" fill="{theme.bg_medium}" stroke="{theme.primary}" stroke-opacity="0.55" stroke-width="2"/>',
    ]

    for row in range(rows):
        for col in range(cols):
            x = margin + col * cell_w
            y = size - margin - (row + 1) * cell_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                'fill="rgba(255,255,255,0.025)" stroke="rgba(255,255,255,0.07)" stroke-width="1" '
                'class="action-field-cell"/>'
            )
            vx, vy = _coerce_velocity(action_grid[row, col])
            cx = x + cell_w / 2
            cy = y + cell_h / 2
            if vx == 0 and vy == 0:
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{max(3.0, min(cell_w, cell_h) * 0.11):.1f}" '
                    f'fill="{theme.accent}" fill-opacity="0.78" class="stationary-marker"/>'
                )
            else:
                line_len = min(cell_w, cell_h) * 0.30
                parts.append(
                    f'<line x1="{cx - vx * line_len * 0.45:.1f}" y1="{cy + vy * line_len * 0.45:.1f}" '
                    f'x2="{cx + vx * line_len:.1f}" y2="{cy - vy * line_len:.1f}" '
                    f'stroke="{theme.accent}" stroke-width="2.2" stroke-linecap="round" '
                    f'marker-end="url(#{arrow_id})" class="action-arrow"/>'
                )

    ex, ey = pt(*motion.exit_center)
    exit_r = float(motion.exit_radius_m) * span / float(motion.room_width_m)
    sx, sy = pt(*motion.start_position)
    parts.append(
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{exit_r:.1f}" fill="{theme.cell_exit}" '
        'fill-opacity="0.20" stroke="rgba(118,255,3,0.8)" stroke-width="2" class="exit-zone"/>'
    )
    parts.append(
        f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="{theme.cell_start}" '
        'stroke="white" stroke-opacity="0.55" stroke-width="1.5" class="start-point"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_vi_animation_frame(
    grid: np.ndarray,
    values: np.ndarray,
    iteration: int,
    total_iterations: int,
    cell_size: int = 50,
    room_id: str = "room1",
) -> str:
    # Heatmap frame used to show Value Iteration values changing over sweeps.
    theme = get_theme(room_id)
    rows, cols = grid.shape
    width = cols * cell_size
    height = rows * cell_size

    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="grid-canvas">'
    ]

    if np.all(values == 0):
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    vrange = vmax - vmin if vmax != vmin else 1

    for r in range(rows):
        for c in range(cols):
            cell = CellType(int(grid[r, c]))
            x = c * cell_size
            y = r * cell_size
            if cell == CellType.WALL:
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                    f'fill="{theme.cell_wall}" rx="3" stroke="#333" stroke-width="0.5"/>'
                )
            else:
                v = values[r, c] if not np.isnan(values[r, c]) else 0
                norm = (v - vmin) / vrange
                r_val = int(30 + 160 * norm)
                g_val = int(30 + 120 * (1 - norm))
                b_val = int(200 - 100 * norm)
                bg = f"rgb({r_val},{g_val},{b_val})"

                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                    f'fill="{bg}" rx="3" stroke="#333" stroke-width="0.5"/>'
                )

                if not np.isnan(v):
                    svg_parts.append(
                        f'<text x="{x + cell_size // 2}" y="{y + cell_size // 2 + 3}" '
                        f'text-anchor="middle" fill="white" font-size="{cell_size // 5}">'
                        f"{v:.1f}</text>"
                    )

    # Overlay iteration count
    pct = int(100 * iteration / max(1, total_iterations))
    svg_parts.append(
        f'<text x="{width - 10}" y="16" text-anchor="end" '
        f'fill="rgba(255,255,255,0.5)" font-size="12">'
        f"Iteration {iteration} / {total_iterations} ({pct}%)</text>"
    )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
