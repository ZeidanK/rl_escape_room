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


def _render_room2_game():
    from game.room2_game import render_room2_game
    render_room2_game()


def _render_room3_game():
    from game.room3_game import render_room3_game
    render_room3_game()


def _render_room4_game():
    from game.room4_game import render_room4_game
    render_room4_game()


def _click_button_by_label(at: AppTest, label: str) -> AppTest:
    for button in at.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"button not found: {label}")


def _set_number_by_label(at: AppTest, label: str, value) -> AppTest:
    for widget in at.sidebar.number_input:
        if widget.label == label:
            return widget.set_value(value).run()
    raise AssertionError(f"number input not found: {label}")


def _prepare_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage" / "models").mkdir(parents=True)


def test_room4_analysis_mode_renders_without_auto_training():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    room4_option = next(option for option in at.sidebar.radio[0].options if "Room 4" in option)
    at.sidebar.radio[0].set_value(room4_option).run()

    assert len(at.exception) == 0
    assert at.session_state["approx_result"] is None


def test_room5_analysis_mode_renders_without_auto_training():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    room5_option = next(option for option in at.sidebar.radio[0].options if "Room 5" in option)
    at.sidebar.radio[0].set_value(room5_option).run()

    assert len(at.exception) == 0
    assert at.session_state["dqn_result"] is None


def test_room5_tiny_training_evaluation_and_replay():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()
    room5_option = next(option for option in at.sidebar.radio[0].options if "Room 5" in option)
    at.sidebar.radio[0].set_value(room5_option).run()

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

    at = _click_button_by_label(at, "Evaluate Fixed Layout")
    assert len(at.exception) == 0
    assert at.session_state["dqn_eval_fixed"] is not None

    at = _click_button_by_label(at, "Generate Greedy Replay")
    assert len(at.exception) == 0
    assert at.session_state["dqn_rollout"] is not None


def test_room2_game_loads_saved_sarsa_model(tmp_path, monkeypatch):
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
    _click_button_by_label(at, "Load Latest Model")

    assert len(at.exception) == 0
    assert at.session_state["r2g_replay"] is not None


def test_room3_game_loads_saved_q_learning_model(tmp_path, monkeypatch):
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
    _click_button_by_label(at, "Load Latest Model")

    assert len(at.exception) == 0
    assert at.session_state["r3g_replay"] is not None


def test_room4_game_loads_saved_approximate_model(tmp_path, monkeypatch):
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
    _click_button_by_label(at, "Load Latest Model")

    assert len(at.exception) == 0
    assert at.session_state["r4g_rollout"] is not None


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
