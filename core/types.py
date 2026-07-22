from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


Position = tuple[int, int]


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_DELTAS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

TURN_LEFT: dict[Action, Action] = {
    Action.UP: Action.LEFT,
    Action.RIGHT: Action.UP,
    Action.DOWN: Action.RIGHT,
    Action.LEFT: Action.DOWN,
}

TURN_RIGHT: dict[Action, Action] = {
    Action.UP: Action.RIGHT,
    Action.RIGHT: Action.DOWN,
    Action.DOWN: Action.LEFT,
    Action.LEFT: Action.UP,
}


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    EXIT = 3
    SLIPPERY = 4
    TRAP = 5
    KEY = 6
    LOCKED_EXIT = 7


class RoomKind(IntEnum):
    GRID_KNOWN = 1
    GRID_UNKNOWN_STOCHASTIC = 2
    GRID_UNKNOWN = 3
    CONTINUOUS = 4
    CONTINUOUS_OBSTACLES = 5


@dataclass(frozen=True)
class SlipConfig:
    intended_probability: float = 0.80
    left_probability: float = 0.10
    right_probability: float = 0.10

    def __post_init__(self):
        total = self.intended_probability + self.left_probability + self.right_probability
        if abs(total - 1.0) > 1e-10:
            raise ValueError(
                f"Slip probabilities must sum to 1.0; received {total}"
            )
        if not all(0.0 <= p <= 1.0 for p in (self.intended_probability, self.left_probability, self.right_probability)):
            raise ValueError("Each slip probability must be between 0 and 1")


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = -1.0
    exit_reward: float = 100.0
    wall_penalty: float = -3.0
    trap_penalty: float = -20.0
    key_reward: float = 10.0
    locked_exit_penalty: float = -5.0
    step_limit_penalty: float = -30.0
    time_bonus_scale: float = 0.0

    def compute_exit_reward(self, max_steps: int, steps_taken: int) -> float:
        bonus = max(0, max_steps - steps_taken) * self.time_bonus_scale
        return self.exit_reward + bonus


@dataclass(frozen=True)
class StepResult:
    next_state: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class TransitionOutcome:
    probability: float
    next_state: Position
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class GridRenderState:
    grid: np.ndarray
    agent_position: Position
    step_count: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    name: str
    kind: RoomKind
    algorithm: str
    state_description: str
    action_description: str
    grid_size: tuple[int, int] | None
    is_continuous: bool
    continuous_size: tuple[float, float] | None
    dt: float | None
    velocity_values: tuple[int, ...] | None
    rewards: RewardConfig = field(default_factory=RewardConfig)
    description: str = ""
