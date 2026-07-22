import numpy as np

from core.types import CellType, Action, RewardConfig
from environments.grid_environment import GridEnvironment


ROOM3_GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 1, 0, 0, 0, 7, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 4, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 4, 0, 0, 0, 0, 8],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])


class Room3QLearning(GridEnvironment):
    def __init__(self, slip_prob: float = 0.2, max_steps: int = 300,
                 rewards: RewardConfig | None = None, seed: int | None = None):
        grid = ROOM3_GRID.copy()
        super().__init__(grid, slip_prob=slip_prob, max_steps=max_steps, rewards=rewards, seed=seed)
        self.has_key = False

    def reset(self, seed: int | None = None) -> tuple[int, int, bool]:
        state = super().reset(seed=seed)
        self.has_key = False
        return (*state, self.has_key)

    def step(self, action: int) -> tuple[tuple[int, int, bool], float, bool, dict]:
        self.step_count += 1
        actual_action = self._apply_slippery(action)

        dr, dc = ACTION_DELTAS[Action(actual_action)]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        r, c = new_pos
        info = {"timeout": False, "key_collected": False}

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            reward, terminated = self.rewards.wall_penalty, False
        else:
            cell = self.grid[r, c]
            if cell == CellType.WALL:
                reward, terminated = self.rewards.wall_penalty, False
                r, c = self.agent_pos
            elif cell == CellType.TRAP:
                reward, terminated = self.rewards.trap_penalty, False
                self.agent_pos = (r, c)
            elif cell == CellType.KEY:
                self.has_key = True
                self.grid[r, c] = CellType.EMPTY
                reward, terminated = self.rewards.key_reward, False
                self.agent_pos = (r, c)
                info["key_collected"] = True
            elif cell == CellType.LOCKED_EXIT:
                if self.has_key:
                    reward = self.rewards.compute_exit_reward(self.max_steps, self.step_count)
                    terminated = True
                    self.agent_pos = (r, c)
                else:
                    reward, terminated = self.rewards.locked_exit_penalty, False
            elif cell == CellType.EXIT:
                reward = self.rewards.compute_exit_reward(self.max_steps, self.step_count)
                terminated = True
                self.agent_pos = (r, c)
            else:
                reward, terminated = self.rewards.step_penalty, False
                self.agent_pos = (r, c)

        if self.step_count >= self.max_steps and not terminated:
            terminated = True
            info["timeout"] = True

        return (*self.agent_pos, self.has_key), reward, terminated, info
