"""Visualization helpers for Room 5 obstacle layouts."""

from __future__ import annotations

from typing import Any


def render_room5_svg(env: Any, rollout: Any | None = None) -> str:
    """Render the Room 5 10x10m obstacle layout and optional rollout path."""
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
        f'<svg viewBox="0 0 {canvas:.0f} {canvas:.0f}" width="100%" '
        'xmlns="http://www.w3.org/2000/svg" class="grid-canvas room5-obstacle-canvas" '
        'role="img" aria-label="Room 5 obstacle grid" '
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
            f'<rect x="{cx - size / 2:.1f}" y="{cy - size / 2:.1f}" width="{size:.1f}" height="{size:.1f}" '
            f'fill="#7f1d1d" stroke="{stroke}" stroke-width="2"/>'
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
