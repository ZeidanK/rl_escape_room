"""Grid-world environment mechanics for the tabular escape-room tasks."""

from collections import deque
from typing import Any

import numpy as np

from core.types import (
    ACTION_DELTAS,
    TURN_LEFT,
    TURN_RIGHT,
    Action,
    CellType,
    GridRenderState,
    Position,
    RewardConfig,
    SlipConfig,
    StepResult,
    TransitionOutcome,
)
from environments.base_environment import BaseEnvironment


# Text-map symbols used by the 10x10 grid rooms.  Keeping maps as strings makes
# the room layouts easy to inspect in code while still converting to numeric
# arrays for fast environment logic.
CHAR_TO_CELL: dict[str, CellType] = {
    ".": CellType.EMPTY,
    "#": CellType.WALL,
    "S": CellType.START,
    "E": CellType.EXIT,
    "I": CellType.SLIPPERY,
    "T": CellType.TRAP,
    "K": CellType.KEY,
    "L": CellType.LOCKED_EXIT,
}

CELL_TO_CHAR: dict[CellType, str] = {v: k for k, v in CHAR_TO_CELL.items()}


def parse_grid_map(lines: list[str]) -> np.ndarray:
    # All required grid rooms are 10x10.  Validation happens here so room files
    # can define their maps declaratively and fail early if a symbol is wrong.
    if len(lines) != 10:
        raise ValueError(f"Grid must have exactly 10 rows; received {len(lines)}")
    for i, row in enumerate(lines):
        if len(row) != 10:
            raise ValueError(f"Row {i} must have exactly 10 characters; received {len(row)}")
        for j, ch in enumerate(row):
            if ch not in CHAR_TO_CELL:
                raise ValueError(f"Unknown map symbol {ch!r} at row {i}, column {j}")

    grid = np.zeros((10, 10), dtype=np.int32)
    for i, row in enumerate(lines):
        for j, ch in enumerate(row):
            grid[i, j] = int(CHAR_TO_CELL[ch])
    return grid


class GridEnvironment(BaseEnvironment):
    # Base implementation for Rooms 1-3: walls, slippery cells, traps, step
    # limits, rewards, and rendering are shared.  Room-specific subclasses only
    # override the pieces that differ, such as Room 3's key state.
    def __init__(
        self,
        grid: np.ndarray,
        reward_config: RewardConfig | None = None,
        max_steps: int = 200,
        slip_config: SlipConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self._grid = grid.copy()
        self._original_grid = grid.copy()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.slip_config = slip_config or SlipConfig()
        self.rows, self.cols = self._grid.shape
        self._agent_pos: Position = (0, 0)
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._validate_grid()
        start_positions = np.argwhere(self._grid == CellType.START)
        self._start_pos = (int(start_positions[0][0]), int(start_positions[0][1]))
        self._agent_pos = self._start_pos

    # --- Public properties ---

    @property
    def agent_position(self) -> Position:
        return self._agent_pos

    @property
    def state(self) -> Any:
        return self._encode_state()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def is_done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def start_position(self) -> Position:
        return self._start_pos

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    # --- Terminal cell type hook (override for Room 3) ---

    def _terminal_cell_types(self) -> set[CellType]:
        return {CellType.EXIT}

    @property
    def goal_position(self) -> Position | None:
        for cell_type in self._terminal_cell_types():
            positions = np.argwhere(self._grid == cell_type)
            if len(positions) > 0:
                return (int(positions[0][0]), int(positions[0][1]))
        return None

    @property
    def states(self) -> tuple[Position, ...]:
        rows, cols = self._grid.shape
        result = []
        for r in range(rows):
            for c in range(cols):
                if CellType(int(self._grid[r, c])) != CellType.WALL:
                    result.append((r, c))
        return tuple(result)

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(Action)

    def is_terminal_state(self, state: Position) -> bool:
        if not self._is_inside(state):
            return False
        return CellType(int(self._grid[state])) in self._terminal_cell_types()

    # --- Validation ---

    def _validate_grid(self) -> None:
        if self._grid.shape != (10, 10):
            raise ValueError(f"Grid must have shape (10, 10); received {self._grid.shape}")
        valid_types = set(int(t) for t in CellType)
        for r in range(self.rows):
            for c in range(self.cols):
                if int(self._grid[r, c]) not in valid_types:
                    raise ValueError(f"Unknown cell value {self._grid[r, c]} at ({r}, {c})")
        starts = int(np.sum(self._grid == CellType.START))
        if starts != 1:
            raise ValueError(f"Map must contain exactly one START cell; found {starts}")
        terminal_types = self._terminal_cell_types()
        goals_found = 0
        for cell_type in terminal_types:
            goals_found += int(np.sum(self._grid == cell_type))
        if goals_found < 1:
            names = ", ".join(t.name for t in terminal_types)
            raise ValueError(f"Map must contain at least one terminal cell type ({names})")
        self._check_reachability()

    def _check_reachability(self) -> None:
        # Breadth-first search confirms the assignment room is solvable from
        # START before training begins.
        start = tuple(np.argwhere(self._grid == CellType.START)[0])
        visited = set()
        queue = deque([start])
        terminal_types = self._terminal_cell_types()
        found_goal = False
        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            r, c = pos
            if CellType(self._grid[r, c]) in terminal_types:
                found_goal = True
                break
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cell = CellType(int(self._grid[nr, nc]))
                    if cell != CellType.WALL:
                        queue.append((nr, nc))
        if not found_goal:
            raise ValueError("No reachable path from START to any terminal cell")

    # --- State encoding hook (override for Room 3) ---

    def _encode_state(self) -> Position:
        return self._agent_pos

    # --- Core API ---

    def reset(self, seed: int | None = None) -> Any:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._grid = self._original_grid.copy()
        self._agent_pos = self._start_pos
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        return self._encode_state()

    def step(self, action: Action | int) -> StepResult:
        # Main grid transition: sample any slip, move if possible, apply reward
        # shaping, and mark terminal/truncated status.
        if self.is_done:
            raise RuntimeError("Cannot call step() after the episode has ended. Call reset() first.")
        action = Action(action)
        self._step_count += 1
        effective_action = self._sample_effective_action(action)
        dr, dc = ACTION_DELTAS[effective_action]
        candidate = (self._agent_pos[0] + dr, self._agent_pos[1] + dc)
        reward = self.reward_config.step_penalty
        collision: str | None = None
        info: dict[str, Any] = {
            "requested_action": int(action),
            "effective_action": int(effective_action),
            "slipped": effective_action != action,
        }
        if not self._is_inside(candidate):
            collision = "boundary"
            new_pos = self._agent_pos
        elif self._get_cell(candidate) == CellType.WALL:
            collision = "wall"
            new_pos = self._agent_pos
        else:
            new_pos = candidate
        if collision:
            reward += self.reward_config.wall_penalty
            info["collision"] = collision
            self._agent_pos = new_pos
        else:
            self._agent_pos = new_pos
            cell = self._get_cell(new_pos)
            add_reward, terminated, extra = self._on_enter_cell(new_pos, cell)
            reward += add_reward
            info.update(extra)
            if terminated:
                self._terminated = True
        if not self._terminated and self._step_count >= self.max_steps:
            reward += self.reward_config.step_limit_penalty
            self._truncated = True
            info["event"] = "timeout"
            info["success"] = False
        return StepResult(
            next_state=self._encode_state(),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            info=info,
        )

    def render(self) -> GridRenderState:
        return GridRenderState(
            grid=self._grid.copy(),
            agent_position=self._agent_pos,
            step_count=self._step_count,
            terminated=self._terminated,
            truncated=self._truncated,
        )

    def render_ansi(self) -> str:
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                if (r, c) == self._agent_pos:
                    row_chars.append("A")
                else:
                    row_chars.append(CELL_TO_CHAR.get(CellType(int(self._grid[r, c])), "?"))
            lines.append("".join(row_chars))
        return "\n".join(lines)

    # --- Internal helpers ---

    def _get_cell(self, position: Position) -> CellType:
        return CellType(int(self._grid[position]))

    def _is_inside(self, position: Position) -> bool:
        r, c = position
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_walkable(self, position: Position) -> bool:
        if not self._is_inside(position):
            return False
        return self._get_cell(position) != CellType.WALL

    def _sample_effective_action(self, action: Action) -> Action:
        # Slippery cells model stochastic dynamics.  The agent requests an
        # action, but the environment may rotate it left or right.
        cell = self._get_cell(self._agent_pos)
        if cell != CellType.SLIPPERY:
            return action
        p = self.rng.random()
        if p < self.slip_config.intended_probability:
            return action
        elif p < self.slip_config.intended_probability + self.slip_config.left_probability:
            return TURN_LEFT[action]
        else:
            return TURN_RIGHT[action]

    def _on_enter_cell(self, position: Position, cell: CellType) -> tuple[float, bool, dict]:
        # Hook used by subclasses to customize special cells.  The base grid
        # knows only traps and normal exits.
        info: dict[str, Any] = {}
        if cell == CellType.TRAP:
            info["event"] = "trap"
            return self.reward_config.trap_penalty, False, info
        elif cell == CellType.EXIT:
            info["event"] = "exit"
            info["success"] = True
            exit_reward = self.reward_config.compute_exit_reward(self.max_steps, self._step_count)
            return exit_reward, True, info
        return 0.0, False, info


class KnownModelGridEnvironment(GridEnvironment):
    # Room 1 uses Dynamic Programming, so the agent needs the full transition
    # distribution P(s'|s,a) instead of only sampled transitions from step().
    def get_transition_distribution(
        self,
        state: Position,
        action: Action | int,
    ) -> tuple[TransitionOutcome, ...]:
        action = Action(action)
        r, c = state
        cell = CellType(int(self._grid[r, c]))
        if cell == CellType.WALL:
            return (TransitionOutcome(1.0, state, 0.0, False, False),)
        if cell == CellType.SLIPPERY:
            action_probs: list[tuple[float, Action]] = [
                (self.slip_config.intended_probability, action),
                (self.slip_config.left_probability, TURN_LEFT[action]),
                (self.slip_config.right_probability, TURN_RIGHT[action]),
            ]
        else:
            action_probs = [(1.0, action)]
        outcomes: dict[tuple, TransitionOutcome] = {}
        for prob, effective_action in action_probs:
            # Multiple effective actions can collapse to the same outcome, for
            # example when both hit a wall.  Merge them by summing probability.
            dr, dc = ACTION_DELTAS[effective_action]
            nr, nc = r + dr, c + dc
            collision = False
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                nr, nc = r, c
                collision = True
            next_cell = CellType(int(self._grid[nr, nc]))
            if next_cell == CellType.WALL:
                nr, nc = r, c
                next_cell = cell
                collision = True
            rew = self.reward_config.step_penalty
            terminated = False
            if collision:
                rew += self.reward_config.wall_penalty
            elif next_cell == CellType.TRAP:
                rew += self.reward_config.trap_penalty
            elif next_cell in self._terminal_cell_types():
                rew += self.reward_config.compute_exit_reward(self.max_steps, 0)
                terminated = True
            next_pos = (nr, nc)
            key = (next_pos, rew, terminated)
            if key in outcomes:
                existing = outcomes[key]
                outcomes[key] = TransitionOutcome(
                    probability=existing.probability + prob,
                    next_state=existing.next_state,
                    reward=existing.reward,
                    terminated=existing.terminated,
                    truncated=False,
                )
            else:
                outcomes[key] = TransitionOutcome(
                    probability=prob,
                    next_state=next_pos,
                    reward=rew,
                    terminated=terminated,
                    truncated=False,
                )
        return tuple(outcomes.values())
