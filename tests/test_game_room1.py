"""Tests for Room 1 game view — verifies SVG rendering, replay, and VI frames."""

import numpy as np

from core.types import SlipConfig, ValueIterationConfig, Action, CellType
from environments.room1_dp import Room1DP
from agents.dynamic_programming import ValueIterationAgent, rollout_policy
from game.canvas_renderer import render_grid_canvas, render_vi_animation_frame
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, render_replay_bar, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.html_rendering import normalize_html
from game.theme import render_global_styles, get_theme, ROOM_THEMES
from game.models import ReplayState, ReplayStep, Achievement, AchievementId, RoomTransition
from game.achievements import AchievementTracker, ALL_ACHIEVEMENTS
from game.room_transitions import render_transition_content
from game.room1_game import _compute_vi_frames, _extract_q_from_values


class TestGameModels:
    def test_achievement_creation(self):
        ach = ALL_ACHIEVEMENTS[AchievementId.FIRST_ESCAPE]
        assert ach.name == "First Escape"
        assert not ach.unlocked

    def test_achievement_tracker(self):
        tracker = AchievementTracker()
        assert not tracker.is_unlocked(AchievementId.FIRST_ESCAPE)
        result = tracker.try_unlock_first_escape()
        assert result is not None
        assert tracker.is_unlocked(AchievementId.FIRST_ESCAPE)
        # Second unlock returns None
        assert tracker.try_unlock_first_escape() is None

    def test_achievement_conditions(self):
        tracker = AchievementTracker()
        assert tracker.try_unlock_ice_master(0) is not None
        assert tracker.try_unlock_ice_master(1) is None
        assert tracker.try_unlock_laser_dodger(0) is not None
        assert tracker.try_unlock_vault_expert(True, True) is not None
        assert tracker.try_unlock_vault_expert(True, False) is None

    def test_replay_state_immutability(self):
        step = ReplayStep(
            step_index=0, state=(1, 0), action=Action.RIGHT,
            effective_action=Action.RIGHT,
            reward=-1.0, next_state=(1, 1), slipped=False,
            collision=None, event=None, terminated=False,
            truncated=False, cumulative_reward=-1.0,
        )
        replay = ReplayState(
            room_id="room1", steps=(step,), current_index=0,
            playing=False, speed=1.0, total_steps=1,
            total_reward=-1.0, success=False, stage_label="Final",
        )
        assert replay.current_index == 0
        assert replay.steps[0].action == Action.RIGHT


class TestCanvasRenderer:
    def setup_method(self):
        self.env = Room1DP(seed=42)

    def test_render_grid_canvas_returns_svg(self):
        svg = render_grid_canvas(self.env.grid, agent_pos=(1, 0), room_id="room1")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert 'class="grid-canvas"' in svg

    def test_render_grid_with_policy(self):
        agent = ValueIterationAgent(self.env)
        result = agent.solve()
        svg = render_grid_canvas(
            self.env.grid, agent_pos=(1, 0), room_id="room1",
            policy=result.policy, show_policy=True,
        )
        assert svg.startswith("<svg")

    def test_render_grid_with_values(self):
        agent = ValueIterationAgent(self.env)
        result = agent.solve()
        svg = render_grid_canvas(
            self.env.grid, agent_pos=(1, 0), room_id="room1",
            values=result.values, show_values=True,
        )
        assert svg.startswith("<svg")

    def test_render_grid_with_trajectory(self):
        agent = ValueIterationAgent(self.env)
        result = agent.solve()
        roll = rollout_policy(self.env, result.policy, seed=42)
        traj = [s.state for s in roll.steps]
        svg = render_grid_canvas(
            self.env.grid, agent_pos=traj[-1] if traj else None,
            room_id="room1", trajectory=traj,
        )
        assert svg.startswith("<svg")

    def test_all_room_themes_render(self):
        for room_id in ROOM_THEMES:
            env = Room1DP(seed=42)
            svg = render_grid_canvas(env.grid, agent_pos=(1, 0), room_id=room_id)
            assert svg.startswith("<svg"), f"Failed for room_id={room_id}"


class TestVIAnimation:
    def test_compute_vi_frames(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        frames = _compute_vi_frames(agent, max_frames=5)
        assert len(frames) >= 1
        assert all(f.shape == (10, 10) for f in frames)

    def test_vi_animation_frame_svg(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        frames = _compute_vi_frames(agent, max_frames=3)
        svg = render_vi_animation_frame(env.grid, frames[0], 1, len(frames))
        assert svg.startswith("<svg")

    def test_extract_q_from_values(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        result = agent.solve()
        q_vals = _extract_q_from_values(result.values, env)
        assert isinstance(q_vals, dict)
        for action_name in ["UP", "RIGHT", "DOWN", "LEFT"]:
            assert action_name in q_vals


class TestHUD:
    def test_normalize_html_dedents_home_page_achievement_card(self):
        raw = """
            <div style="margin-top:6px;"><span style="font-size:0.8em;margin-right:4px;" title="First Escape">🏆</span><span style="font-size:0.8em;margin-right:4px;" title="Ice Master">❄️</span><span style="font-size:0.8em;margin-right:4px;" title="Speed Runner">⏱️</span></div>
        """
        html = normalize_html(raw)
        assert html.startswith('<div style="margin-top:6px;">')
        assert html == html.strip()

    def test_render_hud_basic(self):
        html = render_hud(room_name="Test", algorithm="VI")
        assert 'class="game-hud"' in html
        assert html.startswith('<div class="game-hud">')
        assert html == html.strip()

    def test_render_hud_with_all_fields(self):
        html = render_hud(
            room_name="Room 1", algorithm="Value Iteration",
            episode=1, total_episodes=100,
            step=5, max_steps=200,
            state_str="(1, 0)", action=Action.RIGHT,
            reward=-1.0, total_reward=-5.0, epsilon=0.1,
            inventory="Key collected",
        )
        assert "Room 1" in html
        assert "RIGHT" in html
        assert "Key collected" in html

    def test_render_hud_slip_indicator(self):
        html = render_hud(
            room_name="Test", algorithm="VI",
            slip_info={"intended": Action.RIGHT, "actual": Action.UP},
        )
        assert "slippery floor" in html


class TestEpisodeReplay:
    def test_build_replay_from_rollout(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        result = agent.solve()
        roll = rollout_policy(env, result.policy, seed=42)
        replay = build_replay_from_rollout(roll, "room1", stage_label="Final")
        assert replay.total_steps == roll.total_steps
        assert replay.total_reward == roll.total_reward
        assert replay.success == roll.success
        assert replay.stage_label == "Final"

    def test_replay_controls_html(self):
        step = ReplayStep(
            step_index=0, state=(1, 0), action=Action.RIGHT,
            effective_action=Action.RIGHT,
            reward=-1.0, next_state=(1, 1), slipped=False,
            collision=None, event=None, terminated=False,
            truncated=False, cumulative_reward=-1.0,
        )
        replay = ReplayState(
            room_id="room1", steps=(step,), current_index=0,
            playing=False, speed=1.0, total_steps=1,
            total_reward=-1.0, success=False, stage_label="Final",
        )
        html = render_replay_bar(replay, replay_key="test")
        assert "replay-bar" in html
        assert html.startswith('<div class="replay-bar">')
        assert html == html.strip()

    def test_get_current_step(self):
        step = ReplayStep(
            step_index=0, state=(1, 0), action=Action.RIGHT,
            effective_action=Action.RIGHT,
            reward=-1.0, next_state=(1, 1), slipped=False,
            collision=None, event=None, terminated=False,
            truncated=False, cumulative_reward=-1.0,
        )
        replay = ReplayState(
            room_id="room1", steps=(step,), current_index=0,
            playing=False, speed=1.0, total_steps=1,
            total_reward=-1.0, success=False, stage_label="Final",
        )
        assert get_current_step(replay) is step
        empty = ReplayState(
            room_id="room1", steps=(), current_index=0,
            playing=False, speed=1.0, total_steps=0,
            total_reward=0, success=False, stage_label="Final",
        )
        assert get_current_step(empty) is None


class TestExplainPanel:
    def test_render_explain_panel(self):
        q_vals = {"UP": 42.1, "RIGHT": 51.8, "DOWN": 38.4, "LEFT": 44.0}
        html = render_explain_panel(q_vals, selected_action=Action.RIGHT, algorithm="Value Iteration")
        assert "Why this action?" in html
        assert "51.80" in html or "51.8" in html

    def test_render_explain_without_q(self):
        html = render_explain_panel(algorithm="Value Iteration")
        assert "Why this action?" in html

    def test_get_algorithm_explanation(self):
        for key in ["vi", "sarsa", "q_learning", "approximate"]:
            expl = get_algorithm_explanation(key)
            assert len(expl) > 20
        assert get_algorithm_explanation("unknown") == ""


class TestTransitions:
    def test_render_transition_content_html_is_normalized(self):
        transition = RoomTransition(
            room_id="room1",
            success=True,
            steps=12,
            total_reward=42.0,
            new_best=True,
            message="Escaped cleanly.",
            achievements_unlocked=(ALL_ACHIEVEMENTS[AchievementId.FIRST_ESCAPE],),
        )
        html, is_success = render_transition_content(transition)
        assert is_success is True
        assert html.startswith('<div class="transition-card"')
        assert html == html.strip()


class TestGlobalStyles:
    def test_render_global_styles_includes_css(self):
        css = render_global_styles()
        assert "<style>" in css
        assert ".game-container" in css
        assert ".game-hud" in css

    def test_get_theme_returns_valid(self):
        for room_id in ["room1", "room2", "room3", "room4"]:
            theme = get_theme(room_id)
            assert theme.room_id == room_id
        assert get_theme("invalid").room_id == "room1"  # fallback
