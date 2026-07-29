"""Presentation helpers for professor-facing Streamlit views."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import streamlit as st

from game.constants import (
    ABOUT_MODE,
    COMPARISON_MODE,
    LAB_MODE,
    LAB_ROOM_MODES,
    LEGACY_HOME_MODE,
    MANUAL_MODE_LABEL,
    PENDING_MODE_SELECTOR_KEY,
    PENDING_SHOWCASE_ROOM_SELECTOR_KEY,
    PUBLIC_APP_URL,
    ROOM5_BONUS_MODE,
    SHOWCASE_MODE,
)

GITHUB_URL = "https://github.com/ZeidanK/rl_escape_room"


ROOM_PROOFS: dict[str, dict[str, str]] = {
    "room1": {
        "Algorithm": "Value Iteration / Dynamic Programming",
        "State Space": "(row, column) on a 10x10 grid",
        "Action Space": "UP, RIGHT, DOWN, LEFT",
        "Terminal Condition": "Episode terminates when the exit cell is reached.",
        "Reward Rule": "Each move has a step cost, so fewer steps produce a higher return.",
    },
    "room2": {
        "Algorithm": "SARSA, on-policy TD control",
        "State Space": "(row, column) on a 10x10 grid",
        "Action Space": "UP, RIGHT, DOWN, LEFT",
        "Terminal Condition": "Episode terminates at the exit or truncates at max steps.",
        "Reward Rule": "Trap and step penalties reduce return; faster safe routes score higher.",
    },
    "room3": {
        "Algorithm": "Q-Learning, off-policy TD control",
        "State Space": "(row, column, has_key), 92 tabular states",
        "Action Space": "UP, RIGHT, DOWN, LEFT",
        "Terminal Condition": "The locked exit is terminal only after the key is collected.",
        "Reward Rule": "The key adds reward, but every extra step still lowers total return.",
    },
    "room4": {
        "Algorithm": "Semi-gradient SARSA with tile coding",
        "State Space": "Continuous (X, Y, Vx, Vy) in a 10x10 metre room",
        "Action Space": "Nine velocity choices: stop, cardinal, and diagonal moves",
        "Terminal Condition": "Episode terminates inside the exit circle or truncates at max steps.",
        "Reward Rule": "A per-step cost and timeout penalty make shorter successful trajectories better.",
    },
    "room5": {
        "Algorithm": "NumPy DQN with replay buffer and target network",
        "State Space": "22-feature continuous observation vector",
        "Action Space": "Nine velocity choices shared with Room 4",
        "Terminal Condition": "Exit succeeds, obstacle collision fails, timeout truncates.",
        "Reward Rule": "Step, obstacle, boundary, and timeout penalties reward efficient safe escape.",
    },
}


STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("Beginning", "beginning"),
    ("25%", "25"),
    ("50%", "50"),
    ("75%", "75"),
    ("Final", "final"),
)

STAGE_DIRS = {
    "room2": Path("storage/models/room2_sarsa/showcase_stages"),
    "room3": Path("storage/models/room3_q_learning/showcase_stages"),
    "room4": Path("storage/models/room4_approximate_sarsa/showcase_stages"),
}


def normalize_mode(mode: str | None) -> str:
    if mode in (None, "", LEGACY_HOME_MODE):
        return LAB_MODE if mode == LEGACY_HOME_MODE else SHOWCASE_MODE
    return mode


def _query_value(name: str) -> str | None:
    try:
        value = st.query_params.get(name)
    except Exception:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def _set_query_params(**params: str | None) -> None:
    try:
        st.query_params.clear()
        for key, value in params.items():
            if value is not None:
                st.query_params[key] = value
    except Exception:
        pass


def apply_query_params_once() -> None:
    if st.session_state.get("_presentation_query_params_applied"):
        return

    view = _query_value("view")
    room = _query_value("room")
    legacy_mode = _query_value("mode")

    if view:
        view = view.lower()
        if view == "showcase":
            st.session_state.mode = SHOWCASE_MODE
            st.session_state.game_room = room if room in {"room1", "room2", "room3", "room4", "room5"} else None
        elif view == "lab":
            st.session_state.mode = LAB_ROOM_MODES.get(room or "", LAB_MODE)
            st.session_state.game_room = None
        elif view == "bonus":
            st.session_state.mode = SHOWCASE_MODE
            st.session_state.game_room = "room5"
        elif view == "comparison":
            st.session_state.mode = COMPARISON_MODE
            st.session_state.game_room = None
        elif view == "manual":
            st.session_state.mode = "Manual Environment"
            st.session_state.game_room = None
        elif view == "about":
            st.session_state.mode = ABOUT_MODE
            st.session_state.game_room = None
    elif legacy_mode:
        st.session_state.mode = normalize_mode(legacy_mode)
        st.session_state.game_room = None

    st.session_state._presentation_query_params_applied = True


def go_to_showcase_room(room_id: str | None = None) -> None:
    st.session_state.mode = SHOWCASE_MODE
    st.session_state.game_room = room_id
    st.session_state[PENDING_MODE_SELECTOR_KEY] = SHOWCASE_MODE
    st.session_state[PENDING_SHOWCASE_ROOM_SELECTOR_KEY] = True
    _set_query_params(view="showcase", room=room_id)
    st.rerun()


def go_to_lab(room_id: str | None = None) -> None:
    st.session_state.mode = LAB_ROOM_MODES.get(room_id or "", LAB_MODE)
    st.session_state.game_room = None
    st.session_state[PENDING_MODE_SELECTOR_KEY] = LAB_MODE
    _set_query_params(view="lab", room=room_id)
    st.rerun()


def go_to_mode(mode: str, *, view: str | None = None) -> None:
    st.session_state.mode = mode
    st.session_state.game_room = None
    selector_mode = mode
    if mode == "Manual Environment":
        selector_mode = MANUAL_MODE_LABEL
    elif mode == ROOM5_BONUS_MODE or mode in LAB_ROOM_MODES.values():
        selector_mode = LAB_MODE
    st.session_state[PENDING_MODE_SELECTOR_KEY] = selector_mode
    _set_query_params(view=view)
    st.rerun()


def render_assignment_proof(room_id: str) -> None:
    proof = ROOM_PROOFS[room_id]
    with st.container(border=True):
        st.markdown("#### Assignment Proof")
        st.dataframe(
            [{"Requirement": key, "Visible Evidence": value} for key, value in proof.items()],
            width="stretch",
            hide_index=True,
        )


def _training_config(meta: dict[str, Any]) -> dict[str, Any]:
    return dict(meta.get("training_config") or meta.get("config") or {})


def _epsilon_label(cfg: dict[str, Any]) -> str:
    eps = cfg.get("epsilon", {})
    if not isinstance(eps, dict):
        return "N/A"
    kind = eps.get("kind", "N/A")
    start = eps.get("start", "N/A")
    minimum = eps.get("minimum", "N/A")
    decay = eps.get("decay", "N/A")
    return f"{kind}, start={start}, min={minimum}, decay={decay}"


def render_model_provenance(
    *,
    title: str,
    model_stem: str | None,
    metadata: dict[str, Any] | None,
    evaluation_success: float | None = None,
) -> None:
    if not metadata:
        return

    cfg = _training_config(metadata)
    rows: list[dict[str, str]] = [
        {"Field": "Source", "Value": model_stem or "loaded session model"},
        {"Field": "Algorithm", "Value": str(metadata.get("algorithm", title))},
        {"Field": "Training Episodes", "Value": str(cfg.get("episodes", metadata.get("total_episodes", "N/A")))},
        {"Field": "Training Seed", "Value": str(metadata.get("training_seed", cfg.get("seed", "N/A")))},
    ]
    for name in ("alpha", "gamma", "learning_rate"):
        if name in cfg:
            rows.append({"Field": name, "Value": str(cfg[name])})
    rows.append({"Field": "Epsilon Schedule", "Value": _epsilon_label(cfg)})
    if metadata.get("map_signature"):
        rows.append({"Field": "Map Signature", "Value": str(metadata["map_signature"])})
    if metadata.get("weights_sha256"):
        rows.append({"Field": "Weights Checksum", "Value": str(metadata["weights_sha256"])[:16] + "..."})
    if evaluation_success is not None:
        rows.append({"Field": "Saved Evaluation Success", "Value": f"{evaluation_success:.1%}"})

    with st.container(border=True):
        st.markdown(f"#### Loaded Model Provenance - {title}")
        st.dataframe(rows, width="stretch", hide_index=True)


def render_open_lab_button(room_id: str, *, key: str) -> None:
    if st.button("Open Lab Analysis", key=key):
        go_to_lab(room_id)


def stage_options(room_id: str, final_stem: str | None) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    stage_dir = STAGE_DIRS.get(room_id)
    if stage_dir is not None:
        for label, slug in STAGE_LABELS:
            stem = stage_dir / slug
            if stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
                options.append((label, str(stem)))
    if final_stem and not any(label == "Final" for label, _ in options):
        options.append(("Final", final_stem))
    return options


def render_grid_stage_summary(stage_label: str, rollout: Any) -> None:
    slips = getattr(rollout, "slipped_actions", None)
    collisions = getattr(rollout, "collisions", None)
    traps = getattr(rollout, "trap_count", None)
    if hasattr(rollout, "steps"):
        steps = tuple(getattr(rollout, "steps", ()))
        if slips is None:
            slips = sum(1 for step in steps if getattr(step, "slipped", False))
        if collisions is None:
            collisions = sum(1 for step in steps if getattr(step, "collision", None))
        if traps is None:
            traps = sum(1 for step in steps if getattr(step, "event", None) == "trap")
    with st.container(border=True):
        st.markdown(f"#### Selected Policy Stage - {stage_label}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Success", "Yes" if rollout.success else "No")
        c2.metric("Steps", rollout.total_steps)
        c3.metric("Return", f"{rollout.total_reward:.1f}")
        c4.metric("Trap Visits", traps or 0)
        c5.metric("Slips / Collisions", f"{slips or 0} / {collisions or 0}")


def load_final_summary_rows() -> list[dict[str, str]]:
    path = Path("storage/experiments/final/final_summary.csv")
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def final_summary_success(room_name: str) -> float | None:
    for row in load_final_summary_rows():
        if row.get("room") == room_name:
            try:
                return float(row.get("success_rate_mean", ""))
            except ValueError:
                return None
    return None


def render_public_project_links() -> None:
    st.markdown(f"**Public Streamlit app:** {PUBLIC_APP_URL}")
    st.markdown(f"**GitHub:** {GITHUB_URL}")


def read_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    import json

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
