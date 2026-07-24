from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from core.types import Action, Position


class AchievementId(Enum):
    FIRST_ESCAPE = auto()
    ICE_MASTER = auto()
    LASER_DODGER = auto()
    VAULT_EXPERT = auto()
    MOMENTUM_MASTER = auto()
    SPEED_RUNNER = auto()


@dataclass(frozen=True)
class Achievement:
    id: AchievementId
    name: str
    description: str
    emoji: str
    unlocked: bool = False


@dataclass(frozen=True)
class ReplayStep:
    step_index: int
    state: Any
    action: Action | None
    effective_action: Action | None
    reward: float
    next_state: Any
    slipped: bool
    collision: str | None
    event: str | None
    terminated: bool
    truncated: bool
    cumulative_reward: float
    epsilon_at_time: float | None = None
    q_values_at_time: dict[str, float] | None = None
    explanation: str | None = None


@dataclass(frozen=True)
class ReplayState:
    room_id: str
    steps: tuple[ReplayStep, ...]
    current_index: int
    playing: bool
    speed: float
    total_steps: int
    total_reward: float
    success: bool
    stage_label: str


@dataclass(frozen=True)
class RoomUnlockStatus:
    room_id: str
    unlocked: bool
    best_steps: int | None
    best_return: float | None
    best_time_s: float | None
    achievements: tuple[Achievement, ...] = ()


@dataclass(frozen=True)
class GameRoomState:
    room_id: str
    room_name: str
    room_index: int
    algorithm: str
    state_description: str
    challenge: str
    difficulty: int
    status: RoomUnlockStatus


@dataclass(frozen=True)
class RoomTransition:
    room_id: str
    success: bool
    steps: int
    total_reward: float
    new_best: bool
    message: str
    achievements_unlocked: tuple[Achievement, ...] = ()
