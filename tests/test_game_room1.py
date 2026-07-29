"""Tests for Room 1 game view — verifies SVG rendering, replay, and VI frames."""

import numpy as np

from core.types import (
    Action,
    CellType,
    ContinuousRolloutResult,
    ContinuousTrajectoryStep,
    VelocityAction,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import Room2SARSA
from environments.room3_qlearning import Room3QLearning
from environments.room4_continuous import Room4Continuous
from agents.dynamic_programming import ValueIterationAgent, rollout_policy
from game.canvas_renderer import (
    render_action_field_canvas,
    render_continuous_trajectory_canvas,
    render_grid_canvas,
    render_policy_grid_canvas,
    render_vi_animation_frame,
)
from game.hud import render_hud
from game.episode_replay import build_replay_from_rollout, render_replay_bar, get_current_step
from game.explain_panel import render_explain_panel, get_algorithm_explanation
from game.html_rendering import normalize_html
from game.theme import render_global_styles, get_theme, ROOM_THEMES
from game.models import ReplayState, ReplayStep, AchievementId, RoomTransition
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

    def test_policy_grid_canvas_room1_includes_arrows_and_markers(self):
        agent = ValueIterationAgent(self.env)
        result = agent.solve()
        svg = render_policy_grid_canvas(self.env.grid, result.policy, room_id="room1")

        assert svg.startswith("<svg")
        assert 'class="policy-grid-canvas grid-canvas"' in svg
        assert 'class="policy-arrow"' in svg
        assert 'class="policy-marker">S</text>' in svg
        assert 'class="policy-goal"' in svg

    def test_policy_grid_canvas_room2_shows_traps_and_slippery_cells(self):
        env = Room2SARSA(seed=42)
        policy = {
            (r, c): Action.RIGHT
            for r in range(env.grid_shape[0])
            for c in range(env.grid_shape[1])
            if CellType(int(env.grid[r, c])) != CellType.WALL
        }

        svg = render_policy_grid_canvas(env.grid, policy, room_id="room2")

        assert "policy-cell-trap" in svg
        assert "policy-cell-slippery" in svg
        assert 'class="policy-marker">I</text>' in svg
        assert 'class="policy-marker">T</text>' in svg
        assert 'class="policy-arrow"' in svg

    def test_policy_grid_canvas_room3_distinguishes_key_state(self):
        env = Room3QLearning(seed=42)
        policy = {
            (r, c, has_key): Action.RIGHT
            for has_key in (False, True)
            for r in range(env.grid_shape[0])
            for c in range(env.grid_shape[1])
            if CellType(int(env.grid[r, c])) != CellType.WALL
        }

        no_key_svg = render_policy_grid_canvas(env.grid, policy, room_id="room3", has_key=False)
        with_key_svg = render_policy_grid_canvas(env.grid, policy, room_id="room3", has_key=True)

        assert 'class="policy-marker">L</text>' in no_key_svg
        assert "policy-cell-locked-exit" in no_key_svg
        assert 'class="policy-marker">G</text>' in with_key_svg
        assert 'class="policy-goal"' in with_key_svg

    def test_continuous_trajectory_canvas_shows_path_exit_and_collisions(self):
        env = Room4Continuous()
        step0 = ContinuousTrajectoryStep(
            index=0,
            state=(0.5, 0.5, 0, 0),
            requested_action=VelocityAction.NORTH_EAST,
            reward=-0.01,
            next_state=(0.7, 0.7, 1, 1),
            collision=None,
            event=None,
            terminated=False,
            truncated=False,
            distance_to_exit_m=12.7,
        )
        step1 = ContinuousTrajectoryStep(
            index=1,
            state=(0.7, 0.7, 1, 1),
            requested_action=VelocityAction.WEST,
            reward=-1.01,
            next_state=(0.0, 0.7, 0, 1),
            collision="boundary",
            event=None,
            terminated=False,
            truncated=False,
            distance_to_exit_m=12.4,
        )
        rollout = ContinuousRolloutResult(
            seed=7,
            start_state=(0.5, 0.5, 0, 0),
            final_state=(0.0, 0.7, 0, 1),
            total_reward=-1.02,
            steps=2,
            simulated_time_s=0.04,
            success=False,
            terminated=False,
            truncated=False,
            collision_count=1,
            distance_travelled_m=0.9,
            trajectory=(step0, step1),
        )

        svg = render_continuous_trajectory_canvas(env, rollout, max_arrows=4)

        assert 'class="continuous-trajectory-canvas grid-canvas"' in svg
        assert 'class="trajectory-path"' in svg
        assert 'class="exit-zone"' in svg
        assert 'class="collision-marker"' in svg
        assert 'class="trajectory-arrow"' in svg

    def test_action_field_canvas_shows_arrows_and_stop_markers(self):
        env = Room4Continuous()
        field = np.array([
            [VelocityAction.STOP, VelocityAction.NORTH],
            [VelocityAction.EAST, VelocityAction.SOUTH],
        ])

        svg = render_action_field_canvas(env, field, fixed_velocity=(0, 0))

        assert 'class="action-field-canvas grid-canvas"' in svg
        assert 'class="action-arrow"' in svg
        assert 'class="stationary-marker"' in svg
        assert 'class="exit-zone"' in svg


class TestVIAnimation:
    def test_compute_vi_frames(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        frames = _compute_vi_frames(agent, max_frames=5)
        assert len(frames) >= 1
        assert all(f["values"].shape == (10, 10) for f in frames)
        assert all("iteration" in f for f in frames)
        assert all("delta" in f for f in frames)
        assert all("converged" in f for f in frames)

    def test_vi_animation_frame_svg(self):
        env = Room1DP(seed=42)
        agent = ValueIterationAgent(env)
        frames = _compute_vi_frames(agent, max_frames=3)
        svg = render_vi_animation_frame(env.grid, frames[0]["values"], frames[0]["iteration"], len(frames))
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

    def test_render_hud_status_badges_stay_inside_html_block(self):
        html = render_hud(
            room_name="Room 2",
            algorithm="SARSA",
            state_str="(1, 1)",
            status_badges=['<span class="badge-success">SUCCESS</span>'],
        )
        assert '<div class="hud-row" style="margin-top:6px;"><span class="badge-success">SUCCESS</span></div>' in html
        assert "\n" not in html

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
        assert "\n" not in html

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
        assert "\n" not in html


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
