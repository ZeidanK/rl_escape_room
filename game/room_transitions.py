"""Room transition cinematic overlay."""

import html as html_mod

from game.html_rendering import normalize_html
from game.models import RoomTransition


def render_transition_content(transition: RoomTransition, theme_color: str = "#4fc3f7") -> tuple[str, bool]:
    """Returns (card_html, is_success). Caller wraps in overlay div + st.button()."""
    # This function returns HTML only.  Streamlit button handling stays in the
    # caller because buttons cannot be embedded directly in markdown HTML.
    achievements_html = ""
    if transition.achievements_unlocked:
        ach_items = "".join(
            f'<div style="margin:4px 0;">{a.emoji} {html_mod.escape(a.name)} — {html_mod.escape(a.description)}</div>'
            for a in transition.achievements_unlocked
        )
        achievements_html = normalize_html(
            f'<div style="margin:16px 0;padding:10px;background:rgba(255,215,64,0.1);border:1px solid #ffd740;border-radius:8px;">'
            f'<div style="font-weight:700;color:#ffd740;margin-bottom:8px;">Achievements Unlocked</div>'
            f'{ach_items}'
            f'</div>'
        )

    new_best_html = ""
    if transition.new_best:
        new_best_html = '<div style="color:#76ff03;font-weight:600;margin:8px 0;">New best time!</div>'

    is_success = transition.success
    if is_success:
        body = normalize_html(
            f'<h2 style="color:#76ff03;">ROOM ESCAPED</h2>'
            f'<div class="stats">'
            f'<div class="stat-item"><div class="stat-value">{transition.steps}</div><div class="stat-label">Steps</div></div>'
            f'<div class="stat-item"><div class="stat-value">{transition.total_reward:.1f}</div><div class="stat-label">Return</div></div>'
            f'</div>'
            f'{new_best_html}'
            f'<p style="color:#b0bec5;">{html_mod.escape(transition.message)}</p>'
        )
    else:
        body = normalize_html(
            f'<h2 style="color:#ff5252;">ESCAPE FAILED</h2>'
            f'<p style="color:#b0bec5;">{html_mod.escape(transition.message)}</p>'
        )

    html = normalize_html(
        f'<div class="transition-card" style="border-color: {theme_color};">'
        f'{body}'
        f'{achievements_html}'
        f'</div>'
    )
    return html, is_success
