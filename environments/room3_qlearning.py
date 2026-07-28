"""Room 3 key-collection grid environment for tabular Q-Learning."""

from typing import Any

import numpy as np

from core.types import CellType, Position, RewardConfig, SlipConfig
from environments.grid_environment import parse_grid_map, GridEnvironment


# Room 3 adds memory to the tabular state: the same cell can mean something
# different depending on whether the key has already been collected.
ROOM3_MAP = [
    "##########",
    "#S..#...K#",
    "#.#.#.##.#",
    "#.#....#.#",
    "#...##.#.#",
    "###.#I...#",
    "#......#.#",
    "#.####.#.#",
    "#...I....L",
    "##########",
]

ROOM3_GRID = parse_grid_map(ROOM3_MAP)


class Room3QLearning(GridEnvironment):
    # Q-Learning sees states as (row, column, has_key), so this subclass
    # overrides state encoding and the special key/locked-exit cell behaviour.
    def __init__(
        self,
        max_steps: int = 300,
        reward_config: RewardConfig | None = None,
        slip_config: SlipConfig | None = None,
        seed: int | None = None,
    ):
        grid = ROOM3_GRID.copy()
        super().__init__(
            grid=grid,
            reward_config=reward_config,
            max_steps=max_steps,
            slip_config=slip_config,
            seed=seed,
        )
        self._key_collected = False

    def _terminal_cell_types(self) -> set[CellType]:
        return {CellType.EXIT, CellType.LOCKED_EXIT}

    def _encode_state(self) -> tuple[int, int, bool]:
        # The boolean key flag doubles the state space, allowing the policy to
        # choose different actions before and after the key is collected.
        r, c = self._agent_pos
        return (r, c, self._key_collected)

    @property
    def states(self) -> tuple[tuple[int, int, bool], ...]:
        positions = tuple(
            (r, c)
            for r in range(self._grid.shape[0])
            for c in range(self._grid.shape[1])
            if CellType(int(self._grid[r, c])) != CellType.WALL
        )
        return tuple(
            (r, c, has_key)
            for r, c in positions
            for has_key in (False, True)
        )

    @property
    def goal_position(self) -> Position | None:
        for r in range(self._grid.shape[0]):
            for c in range(self._grid.shape[1]):
                if CellType(int(self._grid[r, c])) == CellType.LOCKED_EXIT:
                    return (r, c)
        return None

    @property
    def key_position(self) -> Position | None:
        for r in range(self._grid.shape[0]):
            for c in range(self._grid.shape[1]):
                if CellType(int(self._grid[r, c])) == CellType.KEY:
                    return (r, c)
        return None

    def is_terminal_state(self, state: Any) -> bool:
        if not isinstance(state, tuple) or len(state) != 3:
            return False
        row, column, has_key = state
        if not has_key:
            return False
        goal = self.goal_position
        if goal is None:
            return False
        return (row, column) == goal

    def reset(self, seed: int | None = None) -> Any:
        self._key_collected = False
        return super().reset(seed=seed)

    def _on_enter_cell(self, position: Position, cell: CellType) -> tuple[float, bool, dict]:
        # The locked exit becomes terminal only after the key flag is true.
        if cell == CellType.KEY:
            if not self._key_collected:
                self._key_collected = True
                self._grid[position] = int(CellType.EMPTY)
                return self.reward_config.key_reward, False, {"event": "key", "key_collected": True}
            return 0.0, False, {"event": "key_already_collected"}
        elif cell == CellType.LOCKED_EXIT:
            if self._key_collected:
                exit_reward = self.reward_config.compute_exit_reward(self.max_steps, self._step_count)
                return exit_reward, True, {"event": "exit", "success": True}
            return self.reward_config.locked_exit_penalty, False, {"event": "locked_exit"}
        return super()._on_enter_cell(position, cell)
