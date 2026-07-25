import math

import numpy as np

from core.types import (
    ContinuousRenderState,
    ContinuousRewardConfig,
    ContinuousState,
    Room4MotionConfig,
    StartMode,
    StepResult,
    VELOCITY_BY_ACTION,
    VelocityAction,
)
from environments.base_environment import BaseEnvironment


class Room4Continuous(BaseEnvironment):
    # Continuous 10x10m environment for Approximate SARSA.  State is
    # (x, y, vx, vy), and actions directly choose the next velocity.
    def __init__(
        self,
        motion_config: Room4MotionConfig | None = None,
        reward_config: ContinuousRewardConfig | None = None,
        max_steps: int = 750,
        start_mode: StartMode = StartMode.FIXED,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.motion = motion_config or Room4MotionConfig()
        self.rewards = reward_config or ContinuousRewardConfig()
        self._max_steps = max_steps
        self._start_mode = start_mode
        self.pos = np.zeros(2, dtype=float)
        self.vel = np.array([0, 0], dtype=int)
        self._step_count = 0
        self._collision_count = 0
        self._distance_travelled = 0.0
        self._trajectory: list[tuple[float, float]] = []
        self._terminated = False
        self._truncated = False

    @property
    def state(self) -> ContinuousState:
        return (float(self.pos[0]), float(self.pos[1]), int(self.vel[0]), int(self.vel[1]))

    @property
    def agent_position(self) -> tuple[float, float]:
        return (float(self.pos[0]), float(self.pos[1]))

    @property
    def velocity(self) -> tuple[int, int]:
        return (int(self.vel[0]), int(self.vel[1]))

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(int(a) for a in VelocityAction)

    @property
    def exit_center(self) -> tuple[float, float]:
        return self.motion.exit_center

    @property
    def exit_radius_m(self) -> float:
        return self.motion.exit_radius_m

    def _sample_start(self, rng: np.random.Generator) -> tuple[float, float]:
        # Different start modes are used to test whether tile coding
        # generalizes beyond a single fixed start point.
        if self._start_mode == StartMode.FIXED:
            return self.motion.start_position
        elif self._start_mode == StartMode.RANDOM_LOWER_LEFT:
            x = rng.uniform(0.25, 3.0)
            y = rng.uniform(0.25, 3.0)
            return (x, y)
        elif self._start_mode == StartMode.RANDOM_ROOM:
            ex, ey = self.motion.exit_center
            er = self.motion.exit_radius_m
            for _ in range(100):
                x = rng.uniform(0.0, self.motion.room_width_m)
                y = rng.uniform(0.0, self.motion.room_height_m)
                dx, dy = x - ex, y - ey
                if (dx * dx + dy * dy) > er * er:
                    return (x, y)
            return (0.5, 0.5)

    def reset(self, seed: int | None = None, *, start_state: ContinuousState | None = None) -> ContinuousState:
        # Explicit start_state is used during evaluation on fixed unseen starts;
        # otherwise the selected start mode samples a valid position.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start_state is not None:
            sx, sy, svx, svy = start_state
            if not (0.0 <= sx <= self.motion.room_width_m and 0.0 <= sy <= self.motion.room_height_m):
                raise ValueError(f"start_state position ({sx}, {sy}) outside room bounds")
            if (svx, svy) not in VELOCITY_BY_ACTION.values():
                raise ValueError(f"start_state velocity ({svx}, {svy}) invalid")
            ex, ey = self.motion.exit_center
            dx, dy = sx - ex, sy - ey
            if (dx * dx + dy * dy) <= self.motion.exit_radius_m * self.motion.exit_radius_m:
                raise ValueError(f"start_state ({sx}, {sy}) is inside exit radius")
            self.pos = np.array([sx, sy], dtype=float)
            self.vel = np.array([svx, svy], dtype=int)
        else:
            sx, sy = self._sample_start(self.rng)
            self.pos = np.array([sx, sy], dtype=float)
            self.vel = np.array(self.motion.start_velocity, dtype=int)
        self._step_count = 0
        self._collision_count = 0
        self._distance_travelled = 0.0
        self._trajectory = [(float(self.pos[0]), float(self.pos[1]))]
        self._terminated = False
        self._truncated = False
        return self.state

    def step(self, action: int) -> StepResult:
        # Euler-style motion: velocity is selected, position advances by
        # velocity * time_step, then rewards are assigned from the new position.
        if self.is_done:
            raise RuntimeError("Episode already terminated; call reset() first")
        self._step_count += 1

        action_enum = VelocityAction(action) if isinstance(action, int) else action
        vel = VELOCITY_BY_ACTION[action_enum]

        distance_before = math.sqrt(
            (self.pos[0] - self.motion.exit_center[0]) ** 2
            + (self.pos[1] - self.motion.exit_center[1]) ** 2
        )

        self.vel = np.array(vel, dtype=int)
        new_pos = self.pos + self.vel.astype(float) * self.motion.time_step_s

        # Boundary clipping
        clipped = np.clip(new_pos, 0.0, self.motion.room_width_m)
        collision_detected = bool(
            abs(clipped[0] - new_pos[0]) > 1e-12 or abs(clipped[1] - new_pos[1]) > 1e-12
        )
        if collision_detected:
            for i in range(2):
                if new_pos[i] < 0.0 or new_pos[i] > [self.motion.room_width_m, self.motion.room_height_m][i]:
                    self.vel[i] = 0
            self._collision_count += 1
        self.pos = clipped

        step_len = float(np.linalg.norm(self.vel.astype(float) * self.motion.time_step_s))
        self._distance_travelled += step_len
        self._trajectory.append((float(self.pos[0]), float(self.pos[1])))

        distance_after = math.sqrt(
            (self.pos[0] - self.motion.exit_center[0]) ** 2
            + (self.pos[1] - self.motion.exit_center[1]) ** 2
        )

        reward = self.rewards.step
        info: dict = {
            "collision": "boundary" if collision_detected else None,
            "event": None,
            "success": False,
            "distance_before": distance_before,
            "distance_after": distance_after,
            "distance_travelled_step": step_len,
            "distance_travelled_total": self._distance_travelled,
            "velocity": (int(self.vel[0]), int(self.vel[1])),
            "step_penalty": self.rewards.step,
            "boundary_penalty": 0.0,
            "exit_reward": 0.0,
            "timeout_penalty": 0.0,
            "progress_reward": 0.0,
        }

        if collision_detected:
            penalty = self.rewards.boundary_collision
            reward += penalty
            info["boundary_penalty"] = penalty

        if self.rewards.distance_progress_scale > 0:
            # Positive progress means the agent moved closer to the exit.
            progress = distance_before - distance_after
            progress_reward = self.rewards.distance_progress_scale * progress
            reward += progress_reward
            info["progress_reward"] = progress_reward

        terminated = False
        truncated = False

        if distance_after <= self.motion.exit_radius_m:
            exit_r = self.rewards.exit
            reward += exit_r
            terminated = True
            self._terminated = True
            info["event"] = "exit"
            info["success"] = True
            info["exit_reward"] = exit_r

        if self._step_count >= self._max_steps and not terminated:
            timeout_p = self.rewards.timeout
            reward += timeout_p
            truncated = True
            self._truncated = True
            info["event"] = "timeout"
            info["timeout_penalty"] = timeout_p

        return StepResult(
            next_state=self.state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def render(self) -> ContinuousRenderState:
        return ContinuousRenderState(
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
            trajectory=tuple(self._trajectory),
        )
