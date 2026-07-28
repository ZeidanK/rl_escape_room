"""Room 1 fixed known-model grid used by the Value Iteration agent."""

import numpy as np

from core.types import RewardConfig, SlipConfig
from environments.grid_environment import parse_grid_map, KnownModelGridEnvironment


# Room 1 is the "known model" room.  The map is fixed and the agent can query
# exact transition probabilities, which is why Value Iteration is appropriate.
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
    # No custom behaviour is needed beyond the known-model grid base class.
    # The subclass mainly names the room and supplies its fixed layout.
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
