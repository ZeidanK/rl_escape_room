"""SVG-based canvas renderer for grid-based game views."""

from typing import Any

import numpy as np
from core.types import Action, CellType, Position
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
