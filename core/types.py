from dataclasses import dataclass, field
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    EXIT = 3
    SLIPPERY = 4
    TRAP = 5
    REWARD_ITEM = 6
    KEY = 7
    LOCKED_EXIT = 8


class RoomKind(IntEnum):
    GRID_KNOWN = 1
    GRID_UNKNOWN_STOCHASTIC = 2
    GRID_UNKNOWN = 3
    CONTINUOUS = 4
    CONTINUOUS_OBSTACLES = 5


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
