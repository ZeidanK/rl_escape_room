"""Room 5: continuous dynamic-obstacle escape room."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from core.types import (
    ContinuousState,
    Obstacle,
    Room4MotionConfig,
    Room5ObstacleConfig,
    Room5Observation,
    Room5RenderState,
    Room5RewardConfig,
    StepResult,
    VELOCITY_BY_ACTION,
    VelocityAction,
)
from environments.base_environment import BaseEnvironment


FIXED_ROOM5_OBSTACLES: tuple[Obstacle, ...] = (
    Obstacle(3.0, 4.2),
    Obstacle(4.8, 6.0),
    Obstacle(6.5, 4.0),
    Obstacle(7.5, 7.2),
)


class Room5Obstacles(BaseEnvironment):
    """Continuous 10x10m room with generated avoidable square obstacles."""

    def __init__(
        self,
        motion_config: Room4MotionConfig | None = None,
        obstacle_config: Room5ObstacleConfig | None = None,
        reward_config: Room5RewardConfig | None = None,
        max_steps: int = 260,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.motion = motion_config or Room4MotionConfig(time_step_s=0.05)
        self.obstacle_config = obstacle_config or Room5ObstacleConfig()
        self.rewards = reward_config or Room5RewardConfig()
        self._max_steps = max_steps
        self.pos = np.zeros(2, dtype=float)
        self.vel = np.array([0, 0], dtype=int)
        self._step_count = 0
        self._trajectory: list[tuple[float, float]] = []
        self._terminated = False
        self._truncated = False
        self._success = False
        self._layout_seed = self.obstacle_config.layout_seed
        self._obstacles: tuple[Obstacle, ...] = ()
        self._last_visible: tuple[Obstacle, ...] = ()

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(int(a) for a in VelocityAction)

    @property
    def is_done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def agent_position(self) -> tuple[float, float]:
        return (float(self.pos[0]), float(self.pos[1]))

    @property
    def raw_state(self) -> ContinuousState:
        return (float(self.pos[0]), float(self.pos[1]), int(self.vel[0]), int(self.vel[1]))

    @property
    def state(self) -> Room5Observation:
        return self._observation()

    @property
    def obstacles(self) -> tuple[Obstacle, ...]:
        return self._obstacles

    @property
    def layout_seed(self) -> int:
        return self._layout_seed

    @property
    def obstacle_width_m(self) -> float:
        return self.obstacle_config.obstacle_width_m

    @property
    def observation_distance_m(self) -> float:
        return self.obstacle_config.observation_distance_m

    def reset(
        self,
        seed: int | None = None,
        *,
        layout_seed: int | None = None,
        start_state: ContinuousState | None = None,
    ) -> Room5Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._layout_seed = self.obstacle_config.layout_seed if layout_seed is None else int(layout_seed)
        self._obstacles = self._make_layout(self._layout_seed)
        if start_state is None:
            sx, sy = self.motion.start_position
            svx, svy = self.motion.start_velocity
        else:
            sx, sy, svx, svy = start_state
            self._validate_start(sx, sy, svx, svy)
        self.pos = np.array([sx, sy], dtype=float)
        self.vel = np.array([svx, svy], dtype=int)
        self._step_count = 0
        self._trajectory = [(float(self.pos[0]), float(self.pos[1]))]
        self._terminated = False
        self._truncated = False
        self._success = False
        self._last_visible = self.visible_obstacles()
        return self.state

    def step(self, action: int | VelocityAction) -> StepResult:
        if self.is_done:
            raise RuntimeError("Episode already terminated; call reset() first")
        self._step_count += 1
        action_enum = VelocityAction(action) if isinstance(action, int) else action
        self.vel = np.array(VELOCITY_BY_ACTION[action_enum], dtype=int)

        distance_before = self._distance_to_exit()
        candidate = self.pos + self.vel.astype(float) * self.motion.time_step_s
        clipped = np.array(
            [
                np.clip(candidate[0], 0.0, self.motion.room_width_m),
                np.clip(candidate[1], 0.0, self.motion.room_height_m),
            ],
            dtype=float,
        )
        boundary_collision = bool(np.linalg.norm(candidate - clipped) > 1e-12)
        if boundary_collision:
            for idx, limit in enumerate((self.motion.room_width_m, self.motion.room_height_m)):
                if candidate[idx] < 0.0 or candidate[idx] > limit:
                    self.vel[idx] = 0
        self.pos = clipped
        self._trajectory.append((float(self.pos[0]), float(self.pos[1])))

        distance_after = self._distance_to_exit()
        reward = self.rewards.step
        info: dict = {
            "collision": None,
            "event": None,
            "success": False,
            "requested_action": int(action_enum),
            "velocity": (int(self.vel[0]), int(self.vel[1])),
            "distance_before": distance_before,
            "distance_after": distance_after,
            "visible_obstacle_count": len(self.visible_obstacles()),
            "obstacle_width_m": self.obstacle_config.obstacle_width_m,
            "observation_distance_m": self.obstacle_config.observation_distance_m,
            "progress_reward": 0.0,
            "exit_reward": 0.0,
            "boundary_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "timeout_penalty": 0.0,
        }

        if self.rewards.distance_progress_scale:
            progress_reward = self.rewards.distance_progress_scale * (distance_before - distance_after)
            reward += progress_reward
            info["progress_reward"] = progress_reward

        if boundary_collision:
            reward += self.rewards.boundary_collision
            info["collision"] = "boundary"
            info["boundary_penalty"] = self.rewards.boundary_collision

        obstacle = self._colliding_obstacle()
        if obstacle is not None:
            reward += self.rewards.obstacle_collision
            self._terminated = True
            info["collision"] = "obstacle"
            info["event"] = "obstacle_collision"
            info["obstacle_penalty"] = self.rewards.obstacle_collision
            info["obstacle_center"] = (obstacle.center_x, obstacle.center_y)

        if not self._terminated and distance_after <= self.motion.exit_radius_m:
            reward += self.rewards.exit
            self._terminated = True
            self._success = True
            info["event"] = "exit"
            info["success"] = True
            info["exit_reward"] = self.rewards.exit

        if not self._terminated and self._step_count >= self._max_steps:
            reward += self.rewards.timeout
            self._truncated = True
            info["event"] = "timeout"
            info["timeout_penalty"] = self.rewards.timeout

        self._last_visible = self.visible_obstacles()
        return StepResult(
            next_state=self.state,
            reward=float(reward),
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def visible_obstacles(self) -> tuple[Obstacle, ...]:
        visible: list[tuple[float, Obstacle]] = []
        for obstacle in self._obstacles:
            distance = self._distance_to_obstacle_center(obstacle)
            if distance <= self.obstacle_config.observation_distance_m:
                visible.append((distance, obstacle))
        visible.sort(key=lambda item: item[0])
        return tuple(o for _, o in visible[: self.obstacle_config.nearest_obstacles])

    def layout_signature(self) -> str:
        payload = {
            "layout_seed": self._layout_seed,
            "obstacles": [
                [round(o.center_x, 6), round(o.center_y, 6), round(o.width_m, 6)]
                for o in self._obstacles
            ],
            "observation_distance_m": self.obstacle_config.observation_distance_m,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def render(self) -> Room5RenderState:
        return Room5RenderState(
            x=float(self.pos[0]),
            y=float(self.pos[1]),
            vx=int(self.vel[0]),
            vy=int(self.vel[1]),
            step_count=self._step_count,
            simulated_time_s=self._step_count * self.motion.time_step_s,
            terminated=self._terminated,
            truncated=self._truncated,
            exit_center=self.motion.exit_center,
            exit_radius_m=self.motion.exit_radius_m,
            observation_distance_m=self.obstacle_config.observation_distance_m,
            obstacles=tuple(self._obstacles),
            visible_obstacles=tuple(self._last_visible),
            trajectory=tuple(self._trajectory),
        )

    def _observation(self) -> Room5Observation:
        width = self.motion.room_width_m
        height = self.motion.room_height_m
        ex, ey = self.motion.exit_center
        obs: list[float] = [
            float(self.pos[0] / width),
            float(self.pos[1] / height),
            float(self.vel[0]),
            float(self.vel[1]),
            float((ex - self.pos[0]) / width),
            float((ey - self.pos[1]) / height),
        ]
        visible = [
            (self._distance_to_obstacle_center(obstacle), obstacle)
            for obstacle in self._obstacles
            if self._distance_to_obstacle_center(obstacle) <= self.obstacle_config.observation_distance_m
        ]
        visible.sort(key=lambda item: item[0])
        max_d = self.obstacle_config.observation_distance_m
        for distance, obstacle in visible[: self.obstacle_config.nearest_obstacles]:
            obs.extend(
                [
                    1.0,
                    float((obstacle.center_x - self.pos[0]) / max_d),
                    float((obstacle.center_y - self.pos[1]) / max_d),
                    float(distance / max_d),
                ]
            )
        missing = self.obstacle_config.nearest_obstacles - min(
            len(visible), self.obstacle_config.nearest_obstacles
        )
        for _ in range(missing):
            obs.extend([0.0, 0.0, 0.0, 1.0])
        return tuple(float(v) for v in obs)

    def _make_layout(self, layout_seed: int) -> tuple[Obstacle, ...]:
        if self.obstacle_config.fixed_layout:
            return tuple(FIXED_ROOM5_OBSTACLES[: self.obstacle_config.max_obstacles])
        rng = np.random.default_rng(layout_seed)
        count = int(
            rng.integers(
                self.obstacle_config.min_obstacles,
                self.obstacle_config.max_obstacles + 1,
            )
        )
        obstacles: list[Obstacle] = []
        half = self.obstacle_config.obstacle_width_m / 2
        for _ in range(1000):
            if len(obstacles) >= count:
                break
            candidate = Obstacle(
                center_x=float(rng.uniform(half + 0.25, self.motion.room_width_m - half - 0.25)),
                center_y=float(rng.uniform(half + 0.25, self.motion.room_height_m - half - 0.25)),
                width_m=self.obstacle_config.obstacle_width_m,
            )
            if self._layout_candidate_is_valid(candidate, obstacles):
                obstacles.append(candidate)
        if len(obstacles) < count:
            raise RuntimeError("Could not generate a valid Room 5 obstacle layout")
        return tuple(obstacles)

    def _layout_candidate_is_valid(self, candidate: Obstacle, obstacles: list[Obstacle]) -> bool:
        start_x, start_y = self.motion.start_position
        exit_x, exit_y = self.motion.exit_center
        clearance = self.obstacle_config.obstacle_width_m + self.motion.exit_radius_m + 0.35
        if math.hypot(candidate.center_x - start_x, candidate.center_y - start_y) < clearance:
            return False
        if math.hypot(candidate.center_x - exit_x, candidate.center_y - exit_y) < clearance:
            return False
        for obstacle in obstacles:
            if (
                abs(candidate.center_x - obstacle.center_x) < self.obstacle_config.obstacle_width_m + 0.15
                and abs(candidate.center_y - obstacle.center_y) < self.obstacle_config.obstacle_width_m + 0.15
            ):
                return False
        return True

    def _validate_start(self, sx: float, sy: float, svx: int, svy: int) -> None:
        if not (0.0 <= sx <= self.motion.room_width_m and 0.0 <= sy <= self.motion.room_height_m):
            raise ValueError("start_state position outside room bounds")
        if (svx, svy) not in VELOCITY_BY_ACTION.values():
            raise ValueError("start_state velocity invalid")
        if math.hypot(sx - self.motion.exit_center[0], sy - self.motion.exit_center[1]) <= self.motion.exit_radius_m:
            raise ValueError("start_state is inside exit radius")
        half = self.obstacle_config.obstacle_width_m / 2
        for obstacle in self._obstacles:
            if abs(sx - obstacle.center_x) <= half and abs(sy - obstacle.center_y) <= half:
                raise ValueError("start_state intersects an obstacle")

    def _colliding_obstacle(self) -> Obstacle | None:
        half = self.obstacle_config.obstacle_width_m / 2
        for obstacle in self._obstacles:
            if abs(self.pos[0] - obstacle.center_x) <= half and abs(self.pos[1] - obstacle.center_y) <= half:
                return obstacle
        return None

    def _distance_to_exit(self) -> float:
        return math.hypot(
            float(self.pos[0]) - self.motion.exit_center[0],
            float(self.pos[1]) - self.motion.exit_center[1],
        )

    def _distance_to_obstacle_center(self, obstacle: Obstacle) -> float:
        return math.hypot(float(self.pos[0]) - obstacle.center_x, float(self.pos[1]) - obstacle.center_y)
