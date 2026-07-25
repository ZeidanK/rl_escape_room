"""Game HUD component — unified status display for all rooms."""

import html as html_mod

from core.types import Action


def _action_name(a: Action | int | None) -> str:
    if a is None:
        return "—"
    return Action(a).name if not isinstance(a, Action) else a.name


def render_hud(
    *,
    room_name: str,
    algorithm: str,
    episode: int | None = None,
    total_episodes: int | None = None,
    step: int | None = None,
    max_steps: int | None = None,
    state_str: str | None = None,
    action: Action | int | None = None,
    reward: float | None = None,
    total_reward: float | None = None,
    epsilon: float | None = None,
    status_badges: list[str] | None = None,
    slip_info: dict | None = None,
    inventory: str | None = None,
    custom_items: list[tuple[str, str]] | None = None,
) -> str:
    # Collect only the fields available for the current room/step.  This keeps
    # one HUD component usable for grid rooms, continuous rooms, and DQN.
    badges_html = ""
    if status_badges:
        badges_html = '<div class="hud-row" style="margin-top:6px;">' + "".join(status_badges) + "</div>"

    items: list[tuple[str, str]] = []

    if episode is not None and total_episodes is not None:
        items.append(("Episode", f"{episode:,} / {total_episodes:,}"))
    elif episode is not None:
        items.append(("Episode", f"{episode:,}"))

    if step is not None and max_steps is not None:
        items.append(("Step", f"{step} / {max_steps}"))
    elif step is not None:
        items.append(("Step", str(step)))

    if state_str:
        items.append(("State", state_str))

    if action is not None:
        items.append(("Action", _action_name(action)))

    if reward is not None:
        items.append(("Step Reward", f"{reward:+.1f}"))

    if total_reward is not None:
        items.append(("Total Reward", f"{total_reward:.1f}"))

    if epsilon is not None:
        items.append(("Epsilon", f"{epsilon:.3f}"))

    if inventory:
        items.append(("Inventory", inventory))

    if custom_items:
        items.extend(custom_items)

    rows_html = "".join(
        f'<div class="hud-item"><div class="hud-label">{html_mod.escape(str(label))}</div>'
        f'<div class="hud-value">{html_mod.escape(str(value))}</div></div>'
        for label, value in items
    )

    slip_html = ""
    if slip_info:
        intended = _action_name(slip_info.get("intended"))
        actual = _action_name(slip_info.get("actual"))
        slip_html = (
            f'<div class="slip-indicator">'
            f'<div class="intended">Intended: {html_mod.escape(intended)} → Actual: {html_mod.escape(actual)}</div>'
            f'<div class="cause">Cause: slippery floor</div>'
            f'</div>'
        )

    return f"""
    <div class="game-hud">
        <div class="game-hud-title">{room_name}</div>
        <div class="game-hud-subtitle">{algorithm}</div>
        <div class="hud-row">{rows_html}</div>
        {slip_html}
        {badges_html}
    </div>
    """
