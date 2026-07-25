import numpy as np

from core.types import RewardConfig, SlipConfig
from environments.grid_environment import parse_grid_map, GridEnvironment


# Room 2 keeps the same grid mechanics but hides the transition model from the
# agent.  SARSA learns from sampled experience in this fixed layout.
ROOM2_MAP = [
    "##########",
    "#SI......#",
    "#.##.###.#",
    "#.#T..I#.#",
    "#....#...#",
    "####.#.#.#",
    "#I.....#.#",
    "#.####.#.#",
    "#...I....E",
    "##########",
]

ROOM2_GRID = parse_grid_map(ROOM2_MAP)


class Room2SARSA(GridEnvironment):
    # The generic GridEnvironment already supports slippery cells, traps, and
    # timeouts, so the room class only provides the map and default step limit.
    def __init__(
        self,
        max_steps: int = 200,
        reward_config: RewardConfig | None = None,
        slip_config: SlipConfig | None = None,
        seed: int | None = None,
    ):
        grid = ROOM2_GRID.copy()
        super().__init__(
            grid=grid,
            reward_config=reward_config,
            max_steps=max_steps,
            slip_config=slip_config,
            seed=seed,
        )
