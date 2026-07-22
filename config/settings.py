from dataclasses import dataclass, field


@dataclass
class RLConfig:
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
    algorithm_name: str
    room_name: str
    config: dict
    episodes: list[EpisodeRecord] = field(default_factory=list)
    best_episode: int | None = None
    model_path: str | None = None
