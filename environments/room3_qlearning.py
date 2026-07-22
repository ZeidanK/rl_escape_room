from typing import Any

import numpy as np

from core.types import CellType, Position, RewardConfig, SlipConfig
from environments.grid_environment import parse_grid_map, GridEnvironment


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
        r, c = self._agent_pos
        return (r, c, self._key_collected)

    def reset(self, seed: int | None = None) -> Any:
        self._key_collected = False
        return super().reset(seed=seed)

    def _on_enter_cell(self, position: Position, cell: CellType) -> tuple[float, bool, dict]:
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
