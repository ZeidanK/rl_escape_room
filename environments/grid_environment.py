import numpy as np

from core.types import CellType, Action, RewardConfig
from environments.base_environment import BaseEnvironment


ACTION_DELTAS = {Action.UP: (-1, 0), Action.RIGHT: (0, 1), Action.DOWN: (1, 0), Action.LEFT: (0, -1)}


class GridEnvironment(BaseEnvironment):
    def __init__(self, grid: np.ndarray, slip_prob: float = 0.2, max_steps: int = 200,
                 rewards: RewardConfig | None = None, seed: int | None = None):
        super().__init__(seed=seed)
        self.grid = grid
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.rewards = rewards or RewardConfig()
        self.rows, self.cols = grid.shape

        start_positions = np.argwhere(grid == CellType.START)
        if len(start_positions) == 0:
            raise ValueError("Grid must have a START cell")
        self.start_pos = (int(start_positions[0][0]), int(start_positions[0][1]))

        self.agent_pos = self.start_pos
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> tuple[int, int]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agent_pos = self.start_pos
        self.step_count = 0
        return self.agent_pos

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict]:
        self.step_count += 1
        actual_action = self._apply_slippery(action)

        dr, dc = ACTION_DELTAS[Action(actual_action)]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        reward, terminated, info = self._process_move(new_pos)

        if not terminated:
            self.agent_pos = new_pos

        if self.step_count >= self.max_steps and not terminated:
            terminated = True
            info["timeout"] = True

        return self.agent_pos, reward, terminated, info

    def _apply_slippery(self, action: int) -> int:
        cell = self.grid[self.agent_pos]
        if cell != CellType.SLIPPERY:
            return action
        p = self._rng.random()
        if p < 0.8:
            return action
        elif p < 0.9:
            return (action - 1) % 4
        else:
            return (action + 1) % 4

    def _process_move(self, new_pos: tuple[int, int]) -> tuple[float, bool, dict]:
        r, c = new_pos
        info = {"timeout": False}

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return self.rewards.wall_penalty, False, info

        cell = self.grid[r, c]
        if cell == CellType.WALL:
            return self.rewards.wall_penalty, False, info
        elif cell == CellType.TRAP:
            return self.rewards.trap_penalty, False, info
        elif cell == CellType.EXIT:
            exit_reward = self.rewards.compute_exit_reward(self.max_steps, self.step_count)
            return exit_reward, True, info
        elif cell == CellType.REWARD_ITEM:
            self.grid[r, c] = CellType.EMPTY
            return self.rewards.key_reward, False, info
        elif cell == CellType.KEY:
            self.grid[r, c] = CellType.EMPTY
            return self.rewards.key_reward, False, {"key_collected": True, **info}
        elif cell == CellType.LOCKED_EXIT:
            return self.rewards.locked_exit_penalty, False, info
        else:
            return self.rewards.step_penalty, False, info

    def render(self) -> np.ndarray:
        return self.grid.copy()

    def get_transition_model(self) -> dict:
        transitions = {}
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r, c]
                if cell == CellType.WALL:
                    continue
                for a in Action:
                    outcomes = []
                    if cell == CellType.SLIPPERY:
                        action_probs = [(0.8, a), (0.1, (a - 1) % 4), (0.1, (a + 1) % 4)]
                    else:
                        action_probs = [(1.0, a)]

                    for prob, actual_a in action_probs:
                        dr, dc = ACTION_DELTAS[Action(actual_a)]
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                            nr, nc = r, c

                        cell2 = self.grid[nr, nc]
                        if cell2 == CellType.WALL:
                            nr, nc = r, c

                        rew = self.rewards.step_penalty
                        terminated = False
                        if cell2 == CellType.TRAP:
                            rew = self.rewards.trap_penalty
                        elif cell2 == CellType.EXIT:
                            rew = self.rewards.compute_exit_reward(self.max_steps, 0)
                            terminated = True
                        elif cell2 == CellType.WALL:
                            rew = self.rewards.wall_penalty

                        outcomes.append((prob, (nr, nc), rew, terminated))

                    merged = {}
                    for op, ns, rew, term in outcomes:
                        key = (ns, rew, term)
                        merged[key] = merged.get(key, 0) + op
                    merged_list = [(p, ns, rew, term) for (ns, rew, term), p in merged.items()]

                    transitions[(r, c, a)] = merged_list
        return transitions
