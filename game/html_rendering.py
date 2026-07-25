"""Helpers for rendering custom HTML through Streamlit."""

from collections.abc import Callable
from textwrap import dedent

import streamlit as st


def normalize_html(markup: str) -> str:
    dedented = dedent(str(markup)).strip()
    return "\n".join(line.strip() for line in dedented.splitlines() if line.strip())


def render_html(markup: str, *, target: Callable[..., object] | None = None) -> None:
    renderer = target or st.markdown
    renderer(normalize_html(markup), unsafe_allow_html=True)
