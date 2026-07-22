import numpy as np

from core.types import CellType, RewardConfig
from environments.grid_environment import GridEnvironment


ROOM1_GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 4, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 4, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 4, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])


class Room1DP(GridEnvironment):
    def __init__(self, slip_prob: float = 0.2, max_steps: int = 200,
                 rewards: RewardConfig | None = None, seed: int | None = None):
        grid = ROOM1_GRID.copy()
        super().__init__(grid, slip_prob=slip_prob, max_steps=max_steps, rewards=rewards, seed=seed)
