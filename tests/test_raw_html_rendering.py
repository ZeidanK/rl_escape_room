import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?:div|span|style|script|svg|rect|circle|line|polyline|p|h[1-6]|table|tr|td|th|strong|br|hr)\b",
    re.IGNORECASE,
)


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_literal_text(part) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_text(node.left) + _literal_text(node.right)
    return ""


def _has_unsafe_html_kw(call: ast.Call) -> bool:
    return any(
        kw.arg == "unsafe_allow_html"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in call.keywords
    )


def test_custom_html_goes_through_normalized_renderer():
    offenders: list[str] = []
    paths = [ROOT / "app.py", *sorted((ROOT / "game").rglob("*.py"))]

    for path in paths:
        if path.name == "html_rendering.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = _dotted_name(node.func)
            if not func_name.endswith(".markdown"):
                continue
            first_arg = node.args[0] if node.args else None
            if _has_unsafe_html_kw(node) or HTML_TAG_RE.search(_literal_text(first_arg)):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []


def test_home_room_card_achievement_html_stays_in_one_html_block():
    from game.achievements import AchievementId
    from game.home_page import ROOM_DEFS, _render_room_card_html
    from game.theme import difficulty_badge

    html = _render_room_card_html(
        ROOM_DEFS[0],
        card_class="room-card",
        emoji="❄️",
        diff_badge=difficulty_badge(1),
        unlocked_ids={AchievementId.ICE_MASTER},
    )

    assert html.startswith('<div class="room-card">')
    assert '<div style="margin-top:6px;"><span' in html
    assert 'title="Ice Master"' in html
    assert "\n" not in html
