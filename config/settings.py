from dataclasses import dataclass, field


@dataclass
class RLConfig:
    # General-purpose config shape kept for simple experiments and historical
    # compatibility.  The room-specific agents use richer config dataclasses in
    # core/types.py.
    episodes: int = 5000
    max_steps: int = 500
    gamma: float = 0.95
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42


@dataclass
class EpisodeRecord:
    # Lightweight record for a single training episode.
    episode_number: int
    total_reward: float
    steps: int
    success: bool
    epsilon: float | None
    trajectory: list
    actions: list
    rewards: list


@dataclass
class TrainingResult:
    # Container for experiment outputs that may include a saved model path.
    algorithm_name: str
    room_name: str
    config: dict
    episodes: list[EpisodeRecord] = field(default_factory=list)
    best_episode: int | None = None
    model_path: str | None = None
