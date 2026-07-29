"""Regression tests for Streamlit app wiring and game-view behavior."""

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from agents.approximate_sarsa import (
    ApproximateSarsaAgent,
    evaluate_approximate_policy,
    save_approximate_model,
)
from agents.q_learning import QLearningAgent, save_q_model
from agents.sarsa import SarsaAgent, save_model
from core.types import (
    ApproximateSarsaConfig,
    QLearningConfig,
    SarsaConfig,
    StartMode,
    TileCodingConfig,
)
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA
from environments.room3_qlearning import ROOM3_GRID, Room3QLearning
from environments.room4_continuous import ContinuousRewardConfig, Room4Continuous, Room4MotionConfig


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _select_lab_room(at: AppTest, label: str) -> AppTest:
    at.sidebar.radio[0].set_value("Learning Laboratory").run()
    for selectbox in at.sidebar.selectbox:
        if selectbox.label == "Learning Laboratory Room":
            return selectbox.set_value(label).run()
    raise AssertionError("Learning Laboratory Room selectbox not found")


def _select_showcase_room(at: AppTest, label: str) -> AppTest:
    at.sidebar.radio[0].set_value("Escape Room Showcase").run()
    for selectbox in at.sidebar.selectbox:
        if selectbox.label == "Showcase Room":
            return selectbox.set_value(label).run()
    raise AssertionError("Showcase Room selectbox not found")


def _render_room2_game():
    from game.room2_game import render_room2_game
    render_room2_game()


def _render_room3_game():
    from game.room3_game import render_room3_game
    render_room3_game()


def _render_room4_game():
    from game.room4_game import render_room4_game
    render_room4_game()


def _render_home_with_unlocked_achievements():
    import streamlit as st

    from game.achievements import AchievementId, AchievementTracker
    from game.html_rendering import render_html
    from game.home_page import render_home_page
    from game.theme import render_global_styles

    tracker = AchievementTracker()
    tracker._unlocked = {
        AchievementId.FIRST_ESCAPE,
        AchievementId.ICE_MASTER,
        AchievementId.SPEED_RUNNER,
    }
    st.session_state.achievement_tracker = tracker
    render_html(render_global_styles())
    render_home_page()


def _click_button_by_label(at: AppTest, label: str) -> AppTest:
    for button in at.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"button not found: {label}")


def _click_button_by_key(at: AppTest, key: str) -> AppTest:
    for button in at.button:
        if getattr(button, "key", None) == key:
            return button.click().run()
    raise AssertionError(f"button not found: {key}")


def _set_number_by_label(at: AppTest, label: str, value) -> AppTest:
    for widget in at.sidebar.number_input:
        if widget.label == label:
            return widget.set_value(value).run()
    raise AssertionError(f"number input not found: {label}")


def _set_sidebar_slider_by_label(at: AppTest, label: str, value) -> AppTest:
    for widget in at.sidebar.slider:
        if widget.label == label:
            return widget.set_value(value).run()
    raise AssertionError(f"slider not found: {label}")


def _prepare_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage" / "models").mkdir(parents=True)


def _assert_no_placeholder_messages(at: AppTest) -> None:
    bad_fragments = [
        "appears after local training",
        "available yet",
        "No trajectory available",
        "Run an evaluation from the sidebar",
        "Generate a greedy replay from the sidebar",
        "Train or load",
        "Press **Load Latest Model**",
    ]
    messages = [
        getattr(element, "value", "")
        for group in (at.info, at.caption, at.warning)
        for element in group
    ]
    assert not any(fragment in message for fragment in bad_fragments for message in messages)


def _assert_no_raw_html_text(at: AppTest) -> None:
    raw_fragments = ["<div", "<span", "<script", "<style"]
    messages = [
        getattr(element, "value", "")
        for group_name in ("code", "text")
        for element in getattr(at, group_name, [])
    ]
    assert not any(fragment in message for fragment in raw_fragments for message in messages)


def _assert_no_code_blocks(at: AppTest) -> None:
    assert len(at.code) == 0


def _assert_markdown_contains(at: AppTest, fragment: str) -> None:
    messages = [getattr(element, "value", "") for element in at.markdown]
    assert any(fragment in message for message in messages)


def _assert_error_contains(at: AppTest, fragment: str) -> None:
    messages = [getattr(element, "value", "") for element in at.error]
    assert any(fragment in message for message in messages)


def test_sidebar_shows_five_primary_modes_without_home():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    assert at.sidebar.radio[0].options == [
        "Escape Room Showcase",
        "Learning Laboratory",
        "Manual Play",
        "Algorithm Comparison",
        "About the Project",
    ]
    assert "Home" not in at.sidebar.radio[0].options


def test_legacy_home_mode_maps_to_learning_laboratory():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.session_state["mode"] = "Home"
    at.run()

    assert len(at.exception) == 0
    assert at.session_state["mode"] == "Learning Laboratory"


def test_home_page_with_unlocked_achievements_does_not_show_raw_html():
    at = AppTest.from_function(_render_home_with_unlocked_achievements, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    _assert_no_raw_html_text(at)


def test_room1_analysis_policy_grid_uses_svg_not_code():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at = _select_lab_room(at, "Room 1 — Frozen Maze")

    assert len(at.exception) == 0
    _assert_no_code_blocks(at)
    _assert_markdown_contains(at, "policy-grid-canvas")


def test_lab_slip_probability_validation_survives_restored_slider_state():
    for room_label in ["Room 1 — Frozen Maze", "Room 2 — Laser Corridor", "Room 3 — Key Vault"]:
        at = AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.run()
        at = _select_lab_room(at, room_label)
        at = _set_sidebar_slider_by_label(at, "Intended", 0.75)

        assert len(at.exception) == 0
        _assert_error_contains(at, "Slip probabilities must sum to 1.0")


def test_home_learning_laboratory_button_opens_lab_view():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    at = _click_button_by_label(at, "Open Learning Laboratory")

    assert len(at.exception) == 0
    assert at.session_state["mode"] == "Learning Laboratory"
    assert at.session_state["mode_selector"] == "Learning Laboratory"


def test_home_page_uses_uniform_room_selection_without_start_campaign():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    _assert_markdown_contains(at, "Rooms")
    assert not any(button.label == "Start Campaign" for button in at.button)
    assert not any(button.label == "Open Bonus Room" for button in at.button)
    assert sum(1 for button in at.button if button.label == "Enter Room") == 5

    at = _click_button_by_key(at, "enter_room5")

    assert len(at.exception) == 0
    assert at.session_state["mode"] == "Escape Room Showcase"
    assert at.session_state["game_room"] == "room5"


def test_showcase_sidebar_selects_each_room_without_lab_redirect():
    room_cases = [
        ("Overview", None),
        ("Room 1 — Frozen Maze", "room1"),
        ("Room 2 — Laser Corridor", "room2"),
        ("Room 3 — Key Vault", "room3"),
        ("Room 4 — Momentum Chamber", "room4"),
        ("Room 5 — Obstacle Lab", "room5"),
    ]
    for label, room_id in room_cases:
        at = AppTest.from_file(str(APP_PATH), default_timeout=90)
        at.run()
        at = _select_showcase_room(at, label)

        assert len(at.exception) == 0
        assert at.session_state["mode"] == "Escape Room Showcase"
        assert at.session_state["game_room"] == room_id
        assert at.session_state["mode_selector"] == "Escape Room Showcase"


def test_room1_back_button_returns_to_showcase_overview():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.session_state["mode"] = "Escape Room Showcase"
    at.session_state["game_room"] = "room1"
    at.run()

    at = _click_button_by_label(at, "\u2190 Back to Room Selection")

    assert len(at.exception) == 0
    assert at.session_state["mode"] == "Escape Room Showcase"
    assert at.session_state["game_room"] is None
    assert at.session_state["mode_selector"] == "Escape Room Showcase"


def test_room2_analysis_mode_auto_loads_showcase_outputs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at = _select_lab_room(at, "Room 2 — Laser Corridor")

    assert len(at.exception) == 0
    assert at.session_state["sarsa_result"] is not None
    assert at.session_state["sarsa_eval_summary"] is not None
    assert at.session_state["sarsa_rollout"] is not None
    assert len(at.info) == 0
    _assert_no_placeholder_messages(at)
    _assert_no_code_blocks(at)
    _assert_markdown_contains(at, "policy-grid-canvas")


def test_room2_game_stage_selector_loads_stage_artifacts():
    at = AppTest.from_function(_render_room2_game, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    stage_select = next(select for select in at.selectbox if select.label == "Policy Stage")
    assert list(stage_select.options) == ["Beginning", "25%", "50%", "75%", "Final"]


def test_room3_analysis_mode_auto_loads_showcase_outputs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at = _select_lab_room(at, "Room 3 — Key Vault")

    assert len(at.exception) == 0
    assert at.session_state["ql_result"] is not None
    assert at.session_state["ql_eval_summary"] is not None
    assert at.session_state["ql_rollout"] is not None
    assert len(at.info) == 0
    _assert_no_placeholder_messages(at)
    _assert_no_code_blocks(at)
    _assert_markdown_contains(at, "policy-grid-canvas")


def test_room3_game_stage_selector_loads_stage_artifacts():
    at = AppTest.from_function(_render_room3_game, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    stage_select = next(select for select in at.selectbox if select.label == "Policy Stage")
    assert list(stage_select.options) == ["Beginning", "25%", "50%", "75%", "Final"]


def test_room4_analysis_mode_auto_loads_showcase_outputs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()
    at = _select_lab_room(at, "Room 4 — Momentum Chamber")

    assert len(at.exception) == 0
    assert at.session_state["approx_result"] is not None
    assert at.session_state["approx_eval_fixed"] is not None
    assert at.session_state["approx_eval_gen"] is not None
    assert at.session_state["approx_rollout"] is not None
    assert len(at.info) == 0
    _assert_no_placeholder_messages(at)
    _assert_no_code_blocks(at)
    _assert_markdown_contains(at, "continuous-trajectory-canvas")
    _assert_markdown_contains(at, "action-field-canvas")


def test_room4_game_stage_selector_and_continuous_state_labels():
    at = AppTest.from_function(_render_room4_game, default_timeout=90)
    at.run()

    assert len(at.exception) == 0
    stage_select = next(select for select in at.selectbox if select.label == "Policy Stage")
    assert list(stage_select.options) == ["Beginning", "25%", "50%", "75%", "Final"]
    metric_labels = [metric.label for metric in at.metric]
    for label in ["X", "Y", "Vx", "Vy", "Decision Interval"]:
        assert label in metric_labels
    assert any(metric.value == "0.02 seconds" for metric in at.metric)


def test_room5_analysis_mode_auto_loads_showcase_outputs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at = _select_lab_room(at, "Room 5 — Obstacle Lab")

    assert len(at.exception) == 0
    assert at.session_state["dqn_result"] is not None
    assert at.session_state["dqn_result_source"] == "loaded"
    assert at.session_state["dqn_network"] is not None
    assert at.session_state["dqn_eval_fixed"] is not None
    assert at.session_state["dqn_eval_random"] is not None
    assert at.session_state["dqn_eval_unseen"] is not None
    assert at.session_state["dqn_rollout"] is not None
    assert len(at.info) == 0
    _assert_no_placeholder_messages(at)


def test_comparison_page_loads_saved_results_by_default():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Algorithm Comparison").run()

    assert len(at.exception) == 0
    assert at.session_state["comp_matched"] is not None
    assert at.session_state["comp_tuned"] is not None
    assert at.session_state["comp_source"] == "Final saved comparison"
    assert any(button.label == "Load Final Saved Comparison" for button in at.button)


def test_about_page_and_readme_show_public_deployment_url():
    public_url = "https://rlescaperoom-etswi8z5v9b48mejvamdqw.streamlit.app/"
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("About the Project").run()

    assert len(at.exception) == 0
    _assert_markdown_contains(at, public_url)
    assert public_url in (APP_PATH.parent / "README.md").read_text(encoding="utf-8")


def test_about_page_resolves_screenshots_from_app_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("About the Project").run()

    assert len(at.exception) == 0
    assert len(at.image) == 6
    warning_messages = [getattr(element, "value", "") for element in at.warning]
    assert not any("Screenshot not found" in message for message in warning_messages)


def test_room5_tiny_training_evaluation_and_replay():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()
    at = _select_lab_room(at, "Room 5 — Obstacle Lab")

    for label, value in [
        ("Episodes", 10),
        ("Max Steps", 50),
        ("Replay Capacity", 100),
        ("Batch Size", 4),
        ("Warmup Steps", 4),
        ("Target Update Interval", 5),
        ("Hidden Units", 8),
        ("Eval Episodes", 1),
    ]:
        at = _set_number_by_label(at, label, value)

    at = _click_button_by_label(at, "Train DQN")
    assert len(at.exception) == 0
    assert at.session_state["dqn_result"] is not None
    assert at.session_state["dqn_result_source"] == "live"
    assert len(at.session_state["dqn_result"].metrics) == 10
    assert at.session_state["dqn_eval_fixed"] is None
    assert at.session_state["dqn_eval_random"] is None
    assert at.session_state["dqn_eval_unseen"] is None
    assert at.session_state["dqn_rollout"] is None

    metric_labels = [metric.label for metric in at.metric]
    assert "Episodes" in metric_labels
    assert "Recent Success (10)" in metric_labels
    assert "Final Epsilon" in metric_labels
    assert "Recent Obstacle Rate" in metric_labels
    subheaders = [subheader.value for subheader in at.subheader]
    for label in [
        "Reward per Episode",
        "Steps per Episode",
        "Success and Obstacle Collisions",
        "Epsilon and Loss",
    ]:
        assert label in subheaders
    assert len(at.dataframe) >= 1

    at = _click_button_by_label(at, "Evaluate Fixed Layout")
    assert len(at.exception) == 0
    assert at.session_state["dqn_eval_fixed"] is not None

    at = _click_button_by_label(at, "Generate Greedy Replay")
    assert len(at.exception) == 0
    assert at.session_state["dqn_rollout"] is not None


def test_room2_game_auto_loads_saved_sarsa_model(tmp_path, monkeypatch):
    _prepare_storage(tmp_path, monkeypatch)
    model_dir = tmp_path / "storage" / "models" / "room2_sarsa"
    model_dir.mkdir()

    result = SarsaAgent(
        lambda: Room2SARSA(max_steps=20),
        SarsaConfig(episodes=2, max_steps=20, seed=1),
    ).train()
    save_model(result, str(model_dir / "sarsa_regression"), slip_config=Room2SARSA().slip_config, map_grid=ROOM2_GRID)

    at = AppTest.from_function(_render_room2_game, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    assert at.session_state["r2g_replay"] is not None
    _assert_no_placeholder_messages(at)


def test_room3_game_auto_loads_saved_q_learning_model(tmp_path, monkeypatch):
    _prepare_storage(tmp_path, monkeypatch)
    model_dir = tmp_path / "storage" / "models" / "room3_q_learning"
    model_dir.mkdir()

    result = QLearningAgent(
        lambda: Room3QLearning(max_steps=20),
        QLearningConfig(episodes=2, max_steps=20, seed=2),
    ).train()
    save_q_model(result, str(model_dir / "ql_regression"), slip_config=Room3QLearning().slip_config, map_grid=ROOM3_GRID)

    at = AppTest.from_function(_render_room3_game, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    assert at.session_state["r3g_replay"] is not None
    _assert_no_placeholder_messages(at)


def test_room4_game_auto_loads_saved_approximate_model(tmp_path, monkeypatch):
    _prepare_storage(tmp_path, monkeypatch)
    model_dir = tmp_path / "storage" / "models" / "room4_approximate_sarsa"
    model_dir.mkdir()

    motion = Room4MotionConfig()
    reward = ContinuousRewardConfig()
    result = ApproximateSarsaAgent(
        lambda: Room4Continuous(motion_config=motion, reward_config=reward, max_steps=5),
        ApproximateSarsaConfig(episodes=1, max_steps=5, seed=3, start_mode=StartMode.FIXED),
    ).train()
    save_approximate_model(
        result,
        str(model_dir / "approx_regression"),
        tile_coding_config=TileCodingConfig(),
        motion_config=motion,
        reward_config=reward,
    )

    at = AppTest.from_function(_render_room4_game, default_timeout=60)
    at.run()

    assert len(at.exception) == 0
    assert at.session_state["r4g_rollout"] is not None
    _assert_no_placeholder_messages(at)


def test_approximate_policy_evaluation_honors_start_mode():
    motion = Room4MotionConfig()
    tile_config = TileCodingConfig()
    result = ApproximateSarsaAgent(
        lambda: Room4Continuous(motion_config=motion, max_steps=1, start_mode=StartMode.FIXED),
        ApproximateSarsaConfig(episodes=1, max_steps=1, seed=4, start_mode=StartMode.FIXED),
    ).train()

    summary = evaluate_approximate_policy(
        lambda: Room4Continuous(motion_config=motion, max_steps=1, start_mode=StartMode.FIXED),
        result.weights,
        tile_config,
        motion,
        n_episodes=5,
        seeds=range(5),
        start_mode=StartMode.RANDOM_ROOM,
        max_steps=1,
    )

    fixed_start = (*motion.start_position, *motion.start_velocity)
    assert summary.start_category == StartMode.RANDOM_ROOM.value
    assert any(not np.allclose(rollout.start_state, fixed_start) for rollout in summary.rollouts)
