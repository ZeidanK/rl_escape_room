import numpy as np

from core.types import CellType, RewardConfig
from environments.grid_environment import GridEnvironment


ROOM2_GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 4, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 5, 0, 0, 4, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 4, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 4, 0, 0, 0, 0, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])


class Room2SARSA(GridEnvironment):
    def __init__(self, slip_prob: float = 0.2, max_steps: int = 200,
                 rewards: RewardConfig | None = None, seed: int | None = None):
        grid = ROOM2_GRID.copy()
        super().__init__(grid, slip_prob=slip_prob, max_steps=max_steps, rewards=rewards, seed=seed)
