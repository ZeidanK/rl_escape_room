from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.types import Position


class BaseEnvironment(ABC):
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def agent_position(self) -> Position:
        pass

    @property
    @abstractmethod
    def state(self) -> Any:
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> Any:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> Any:
        pass
