import numpy as np

from core.types import RewardConfig
from environments.base_environment import BaseEnvironment


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

ACTIONS = [UP, RIGHT, DOWN, LEFT, STAY]
ACTION_VELOCITIES = {
    UP: (0, 1),
    RIGHT: (1, 0),
    DOWN: (0, -1),
    LEFT: (-1, 0),
    STAY: (0, 0),
}


class Room4Continuous(BaseEnvironment):
    def __init__(
        self,
        exit_centre: tuple[float, float] = (9.5, 9.5),
        exit_radius: float = 0.3,
        max_steps: int = 500,
        dt: float = 0.02,
        rewards: RewardConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self.exit_centre = np.array(exit_centre, dtype=float)
        self.exit_radius = exit_radius
        self.max_steps = max_steps
        self.dt = dt
        self.rewards = rewards or RewardConfig(step_penalty=-0.1)

        self.pos = np.zeros(2, dtype=float)
        self.vel = np.zeros(2, dtype=int)
        self.step_count = 0
        self.collision_count = 0
        self.distance_travelled = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.array([0.5, 0.5], dtype=float)
        self.vel = np.array([0, 0], dtype=int)
        self.step_count = 0
        self.collision_count = 0
        self.distance_travelled = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], float(self.vel[0]), float(self.vel[1])])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        info = {"timeout": False, "collision": False, "distance": self.distance_travelled}

        if action in ACTION_VELOCITIES:
            self.vel = np.array(ACTION_VELOCITIES[action], dtype=int)

        new_pos = self.pos + self.vel.astype(float) * self.dt
        self.distance_travelled += float(np.linalg.norm(self.vel.astype(float) * self.dt))

        clipped = np.clip(new_pos, 0, 10)
        collision = not np.allclose(clipped, new_pos)
        if collision:
            for i in range(2):
                if new_pos[i] < 0 or new_pos[i] > 10:
                    new_pos[i] = np.clip(new_pos[i], 0, 10)
                    self.vel[i] = 0
            self.collision_count += 1
            info["collision"] = True

        self.pos = np.clip(new_pos, 0, 10)

        reward = self.rewards.step_penalty
        terminated = False

        if collision:
            reward += self.rewards.wall_penalty

        exit_dist = np.linalg.norm(self.pos - self.exit_centre)
        if exit_dist <= self.exit_radius:
            reward = self.rewards.compute_exit_reward(self.max_steps, self.step_count)
            terminated = True

        if self.step_count >= self.max_steps and not terminated:
            terminated = True
            info["timeout"] = True

        info["collision_count"] = self.collision_count
        info["exit_distance"] = float(exit_dist)

        return self._get_state(), reward, terminated, info

    def render(self) -> dict:
        return {
            "position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "exit_centre": self.exit_centre.copy(),
            "exit_radius": self.exit_radius,
            "collision_count": self.collision_count,
            "distance_travelled": self.distance_travelled,
        }
