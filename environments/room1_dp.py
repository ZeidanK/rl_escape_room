import numpy as np

from core.types import RewardConfig, SlipConfig
from environments.grid_environment import parse_grid_map, KnownModelGridEnvironment


ROOM1_MAP = [
    "##########",
    "S....I...#",
    "#.###.##.#",
    "#.#I.....#",
    "#.#.####.#",
    "#....I...#",
    "#####.##.#",
    "#...I....#",
    "#......E.#",
    "##########",
]

ROOM1_GRID = parse_grid_map(ROOM1_MAP)


class Room1DP(KnownModelGridEnvironment):
    def __init__(
        self,
        max_steps: int = 200,
        reward_config: RewardConfig | None = None,
        slip_config: SlipConfig | None = None,
        seed: int | None = None,
    ):
        grid = ROOM1_GRID.copy()
        super().__init__(
            grid=grid,
            reward_config=reward_config,
            max_steps=max_steps,
            slip_config=slip_config,
            seed=seed,
        )
