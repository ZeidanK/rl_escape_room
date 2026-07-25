from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
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


# ============================================================
# Phase 3 — Dynamic Programming types
# ============================================================


@dataclass(frozen=True)
class ValueIterationConfig:
    gamma: float = 0.95
    tolerance: float = 1e-6
    max_iterations: int = 10_000
    tie_tolerance: float = 1e-12

    def __post_init__(self):
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1); received {self.gamma}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive; received {self.tolerance}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive; received {self.max_iterations}")
        if self.tie_tolerance < 0:
            raise ValueError(f"tie_tolerance must be non-negative; received {self.tie_tolerance}")


@dataclass(frozen=True)
class ValueIterationResult:
    values: Mapping[Position, float]
    policy: Mapping[Position, Action | None]
    iterations: int
    converged: bool
    final_delta: float
    delta_history: tuple[float, ...]
    start_state_value: float


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    state: Position
    requested_action: Action
    effective_action: Action
    reward: float
    next_state: Position
    slipped: bool
    collision: str | None
    event: str | None
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class RolloutResult:
    steps: tuple[TrajectoryStep, ...]
    terminated: bool
    truncated: bool
    success: bool
    total_steps: int
    total_reward: float
    collisions: int
    slipped_actions: int
    trap_count: int = 0

    @property
    def traps(self) -> int:
        return self.trap_count


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    episodes: int
    successes: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    std_steps: float
    min_steps: int | None
    max_steps: int | None
    mean_successful_steps: float | None
    total_collisions: int
    total_slipped: int
    trajectories: tuple[tuple[Position, ...], ...]


# ============================================================
# Phase 4 — SARSA types
# ============================================================

Room2Factory = Callable[[], Any]


class EpsilonDecayKind(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class EpsilonScheduleConfig:
    kind: EpsilonDecayKind = EpsilonDecayKind.EXPONENTIAL
    start: float = 1.0
    minimum: float = 0.05
    decay: float = 0.995
    linear_decay_episodes: int = 4_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum <= self.start <= 1.0:
            raise ValueError(
                f"Epsilon must satisfy 0 <= minimum <= start <= 1; "
                f"got start={self.start}, minimum={self.minimum}"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1]; got {self.decay}")
        if self.linear_decay_episodes <= 0:
            raise ValueError(
                f"linear_decay_episodes must be positive; got {self.linear_decay_episodes}"
            )


@dataclass(frozen=True)
class SarsaConfig:
    episodes: int = 5_000
    alpha: float = 0.10
    gamma: float = 0.95
    max_steps: int = 500
    seed: int = 42
    epsilon: EpsilonScheduleConfig = field(default_factory=EpsilonScheduleConfig)
    snapshot_episodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive; got {self.episodes}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha}")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1); got {self.gamma}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {self.max_steps}")


@dataclass(frozen=True)
class TrainingEpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    epsilon: float
    collision_count: int
    slipped_action_count: int
    trap_count: int
    mean_abs_td_error: float = 0.0
    max_abs_td_error: float = 0.0


ProgressCallback = Callable[[int, int, TrainingEpisodeMetrics], None]


@dataclass(frozen=True)
class SarsaSnapshot:
    episode: int
    epsilon: float
    q_values: Mapping[Position, tuple[float, ...]]
    rollout: RolloutResult | None


@dataclass(frozen=True)
class SarsaTrainingResult:
    config: SarsaConfig
    q_values: Mapping[Position, tuple[float, ...]]
    metrics: tuple[TrainingEpisodeMetrics, ...]
    snapshots: Mapping[int, SarsaSnapshot]
    final_epsilon: float
    training_seed: int


@dataclass(frozen=True)
class SarsaEvaluationSummary:
    episodes: int
    successes: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    mean_successful_steps: float | None
    truncated_episodes: int
    total_collisions: int
    total_slipped_actions: int
    total_traps: int
    rollouts: tuple[RolloutResult, ...]


# ============================================================
# Phase 5 — Q-Learning types
# ============================================================

Room3State = tuple[int, int, bool]
Room3Factory = Callable[[], Any]


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 5_000
    alpha: float = 0.10
    gamma: float = 0.95
    max_steps: int = 500
    seed: int = 42
    epsilon: EpsilonScheduleConfig = field(default_factory=EpsilonScheduleConfig)
    snapshot_episodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive; got {self.episodes}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha}")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1); got {self.gamma}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {self.max_steps}")


@dataclass(frozen=True)
class QLearningEpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    epsilon: float
    key_collected: bool
    key_collection_step: int | None
    locked_exit_attempts: int
    collision_count: int
    slipped_action_count: int
    trap_count: int
    mean_abs_td_error: float = 0.0
    max_abs_td_error: float = 0.0


@dataclass(frozen=True)
class QLearningSnapshot:
    episode: int
    epsilon: float
    q_values: Mapping[Room3State, tuple[float, ...]]
    rollout: RolloutResult | None


@dataclass(frozen=True)
class QLearningTrainingResult:
    config: QLearningConfig
    q_values: Mapping[Room3State, tuple[float, ...]]
    metrics: tuple[QLearningEpisodeMetrics, ...]
    snapshots: Mapping[int, QLearningSnapshot]
    final_epsilon: float
    training_seed: int


@dataclass(frozen=True)
class QLearningEvaluationSummary:
    episodes: int
    successes: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    mean_successful_steps: float | None
    key_collection_rate: float
    mean_key_collection_step: float | None
    total_locked_exit_attempts: int
    truncated_episodes: int
    total_collisions: int
    total_slipped_actions: int
    total_traps: int
    rollouts: tuple[RolloutResult, ...]


# ============================================================
# Phase 6 — Function Approximation types
# ============================================================

ContinuousState = tuple[float, float, int, int]


class VelocityAction(IntEnum):
    STOP = 0
    NORTH = 1
    NORTH_EAST = 2
    EAST = 3
    SOUTH_EAST = 4
    SOUTH = 5
    SOUTH_WEST = 6
    WEST = 7
    NORTH_WEST = 8


VELOCITY_BY_ACTION: dict[VelocityAction, tuple[int, int]] = {
    VelocityAction.STOP: (0, 0),
    VelocityAction.NORTH: (0, 1),
    VelocityAction.NORTH_EAST: (1, 1),
    VelocityAction.EAST: (1, 0),
    VelocityAction.SOUTH_EAST: (1, -1),
    VelocityAction.SOUTH: (0, -1),
    VelocityAction.SOUTH_WEST: (-1, -1),
    VelocityAction.WEST: (-1, 0),
    VelocityAction.NORTH_WEST: (-1, 1),
}


class StartMode(str, Enum):
    FIXED = "fixed"
    RANDOM_LOWER_LEFT = "random_lower_left"
    RANDOM_ROOM = "random_room"


@dataclass(frozen=True)
class ContinuousRewardConfig:
    step: float = -0.01
    exit: float = 100.0
    boundary_collision: float = -1.0
    timeout: float = -25.0
    distance_progress_scale: float = 1.0


@dataclass(frozen=True)
class Room4MotionConfig:
    room_width_m: float = 10.0
    room_height_m: float = 10.0
    time_step_s: float = 0.02
    start_position: tuple[float, float] = (0.5, 0.5)
    start_velocity: tuple[int, int] = (0, 0)
    exit_center: tuple[float, float] = (9.5, 9.5)
    exit_radius_m: float = 0.35

    def __post_init__(self) -> None:
        if self.room_width_m <= 0:
            raise ValueError(f"room_width_m must be positive; got {self.room_width_m}")
        if self.room_height_m <= 0:
            raise ValueError(f"room_height_m must be positive; got {self.room_height_m}")
        if self.time_step_s <= 0:
            raise ValueError(f"time_step_s must be positive; got {self.time_step_s}")
        if self.exit_radius_m <= 0:
            raise ValueError(f"exit_radius_m must be positive; got {self.exit_radius_m}")
        sx, sy = self.start_position
        if not (0 <= sx <= self.room_width_m and 0 <= sy <= self.room_height_m):
            raise ValueError(f"start_position {self.start_position} outside room bounds")
        ex, ey = self.exit_center
        dx = sx - ex
        dy = sy - ey
        if (dx*dx + dy*dy) <= (self.exit_radius_m * self.exit_radius_m):
            raise ValueError(f"start_position {self.start_position} is inside exit radius {self.exit_radius_m}")
        if self.start_velocity not in VELOCITY_BY_ACTION.values():
            raise ValueError(f"start_velocity {self.start_velocity} invalid; must be in {set(VELOCITY_BY_ACTION.values())}")


@dataclass(frozen=True)
class TileCodingConfig:
    num_tilings: int = 8
    tiles_x: int = 10
    tiles_y: int = 10
    include_velocity: bool = True

    def __post_init__(self) -> None:
        if self.num_tilings <= 0:
            raise ValueError(f"num_tilings must be positive; got {self.num_tilings}")
        if self.tiles_x <= 0:
            raise ValueError(f"tiles_x must be positive; got {self.tiles_x}")
        if self.tiles_y <= 0:
            raise ValueError(f"tiles_y must be positive; got {self.tiles_y}")
        max_tiles = self.num_tilings * self.tiles_x * self.tiles_y * 3 * 3
        if max_tiles > 1_000_000:
            raise ValueError(f"Estimated feature count {max_tiles} exceeds 1,000,000 limit")


@dataclass(frozen=True)
class ApproximateSarsaConfig:
    episodes: int = 3_000
    alpha: float = 0.10
    gamma: float = 0.99
    max_steps: int = 750
    seed: int = 42
    epsilon: EpsilonScheduleConfig = field(
        default_factory=lambda: EpsilonScheduleConfig(
            start=1.0, minimum=0.02, decay=0.997,
        )
    )
    snapshot_episodes: tuple[int, ...] = ()
    tile_coding: TileCodingConfig = field(default_factory=TileCodingConfig)
    start_mode: StartMode = StartMode.RANDOM_LOWER_LEFT

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive; got {self.episodes}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha}")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1); got {self.gamma}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {self.max_steps}")


@dataclass(frozen=True)
class ApproximateEpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    simulated_time_s: float
    success: bool
    terminated: bool
    truncated: bool
    epsilon: float
    collision_count: int
    distance_travelled_m: float
    final_distance_to_exit_m: float
    mean_abs_td_error: float = 0.0
    max_abs_td_error: float = 0.0


ApproxProgressCallback = Callable[[int, int, ApproximateEpisodeMetrics], None]


@dataclass(frozen=True)
class ApproximateSarsaSnapshot:
    episode: int
    epsilon: float
    weights: np.ndarray
    rollout: "ContinuousRolloutResult | None"


@dataclass(frozen=True)
class ApproximateSarsaTrainingResult:
    config: ApproximateSarsaConfig
    weights: np.ndarray
    metrics: tuple[ApproximateEpisodeMetrics, ...]
    snapshots: Mapping[int, ApproximateSarsaSnapshot]
    final_epsilon: float
    training_seed: int


@dataclass(frozen=True)
class ContinuousTrajectoryStep:
    index: int
    state: ContinuousState
    requested_action: VelocityAction
    reward: float
    next_state: ContinuousState
    collision: str | None
    event: str | None
    terminated: bool
    truncated: bool
    distance_to_exit_m: float


@dataclass(frozen=True)
class ContinuousRolloutResult:
    seed: int
    start_state: ContinuousState
    final_state: ContinuousState
    total_reward: float
    steps: int
    simulated_time_s: float
    success: bool
    terminated: bool
    truncated: bool
    collision_count: int
    distance_travelled_m: float
    trajectory: tuple[ContinuousTrajectoryStep, ...]


@dataclass(frozen=True)
class ContinuousRenderState:
    x: float
    y: float
    vx: int
    vy: int
    step_count: int
    simulated_time_s: float
    terminated: bool
    truncated: bool
    exit_center: tuple[float, float]
    exit_radius_m: float
    trajectory: tuple[tuple[float, float], ...]


FIXED_UNSEEN_STARTS: tuple[tuple[float, float, float, float], ...] = (
    (0.5, 5.0, 0, 0),
    (5.0, 0.5, 0, 0),
    (2.0, 7.0, 0, 0),
    (7.0, 2.0, 0, 0),
    (5.0, 5.0, 0, 0),
)


@dataclass(frozen=True)
class ApproximateEvaluationSummary:
    n_episodes: int
    successes: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    mean_successful_steps: float | None
    truncated_count: int
    total_collisions: int
    mean_distance_travelled_m: float
    total_distance_travelled_m: float
    rollouts: tuple[ContinuousRolloutResult, ...]
    start_category: str = ""


Room4Factory = Callable[[], Any]


# ============================================================
# Optional Room 5 — Dynamic obstacles + DQN types
# ============================================================

Room5Observation = tuple[float, ...]
Room5Factory = Callable[[], Any]


@dataclass(frozen=True)
class Obstacle:
    center_x: float
    center_y: float
    width_m: float = 0.5

    def __post_init__(self) -> None:
        if self.width_m <= 0:
            raise ValueError(f"width_m must be positive; got {self.width_m}")


@dataclass(frozen=True)
class Room5ObstacleConfig:
    min_obstacles: int = 3
    max_obstacles: int = 5
    obstacle_width_m: float = 0.5
    observation_distance_m: float = 2.5
    nearest_obstacles: int = 4
    layout_seed: int = 42
    fixed_layout: bool = False

    def __post_init__(self) -> None:
        if self.min_obstacles < 0:
            raise ValueError("min_obstacles must be non-negative")
        if self.max_obstacles < self.min_obstacles:
            raise ValueError("max_obstacles must be >= min_obstacles")
        if self.obstacle_width_m <= 0:
            raise ValueError("obstacle_width_m must be positive")
        if abs(self.obstacle_width_m - 0.5) > 1e-12:
            raise ValueError("Room 5 obstacle width must be exactly 0.5 metres")
        if self.observation_distance_m <= 0:
            raise ValueError("observation_distance_m must be positive")
        if self.nearest_obstacles <= 0:
            raise ValueError("nearest_obstacles must be positive")


@dataclass(frozen=True)
class Room5RewardConfig:
    step: float = -0.01
    exit: float = 120.0
    boundary_collision: float = -1.0
    obstacle_collision: float = -60.0
    timeout: float = -25.0
    distance_progress_scale: float = 2.0


@dataclass(frozen=True)
class Room5TrajectoryStep:
    index: int
    observation: Room5Observation
    raw_state: ContinuousState
    requested_action: VelocityAction
    reward: float
    next_observation: Room5Observation
    next_raw_state: ContinuousState
    collision: str | None
    event: str | None
    terminated: bool
    truncated: bool
    cumulative_reward: float
    visible_obstacle_count: int
    distance_to_exit_m: float


@dataclass(frozen=True)
class Room5RolloutResult:
    seed: int
    layout_seed: int
    start_state: ContinuousState
    final_state: ContinuousState
    total_reward: float
    steps: int
    simulated_time_s: float
    success: bool
    terminated: bool
    truncated: bool
    boundary_collisions: int
    obstacle_collisions: int
    visible_obstacle_steps: int
    trajectory: tuple[Room5TrajectoryStep, ...]


@dataclass(frozen=True)
class Room5RenderState:
    x: float
    y: float
    vx: int
    vy: int
    step_count: int
    simulated_time_s: float
    terminated: bool
    truncated: bool
    exit_center: tuple[float, float]
    exit_radius_m: float
    observation_distance_m: float
    obstacles: tuple[Obstacle, ...]
    visible_obstacles: tuple[Obstacle, ...]
    trajectory: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 600
    learning_rate: float = 0.001
    gamma: float = 0.99
    max_steps: int = 260
    seed: int = 42
    epsilon: EpsilonScheduleConfig = field(
        default_factory=lambda: EpsilonScheduleConfig(
            start=1.0, minimum=0.05, decay=0.995,
        )
    )
    replay_capacity: int = 20_000
    batch_size: int = 64
    warmup_steps: int = 128
    target_update_interval: int = 100
    hidden_units: int = 64
    snapshot_episodes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")


@dataclass(frozen=True)
class DQNEpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    epsilon: float
    obstacle_collisions: int
    boundary_collisions: int
    visible_obstacle_steps: int
    mean_loss: float
    mean_abs_td_error: float
    max_abs_td_error: float


DQNProgressCallback = Callable[[int, int, DQNEpisodeMetrics], None]


@dataclass(frozen=True)
class DQNSnapshot:
    episode: int
    epsilon: float
    weights: Mapping[str, np.ndarray]
    rollout: Room5RolloutResult | None


@dataclass(frozen=True)
class DQNTrainingResult:
    config: DQNConfig
    weights: Mapping[str, np.ndarray]
    metrics: tuple[DQNEpisodeMetrics, ...]
    snapshots: Mapping[int, DQNSnapshot]
    final_epsilon: float
    training_seed: int
    input_dim: int
    action_count: int


@dataclass(frozen=True)
class DQNEvaluationSummary:
    n_episodes: int
    successes: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    mean_successful_steps: float | None
    truncated_count: int
    obstacle_collision_count: int
    boundary_collision_count: int
    rollouts: tuple[Room5RolloutResult, ...]
    category: str = ""
