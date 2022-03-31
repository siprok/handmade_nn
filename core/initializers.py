import numpy as np
from typing import Callable, Sequence
from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    @staticmethod
    def initialize(cur_neurons: int, next_neurons: int) -> np.ndarray:
        pass


class He(Initializer):
    @staticmethod
    def initialize(cur_neurons: int, next_neurons: int = 0) -> np.ndarray:
        assert cur_neurons > 0
        return np.random.normal(scale=2 / cur_neurons, size=cur_neurons)


class Xavier(Initializer):
@staticmethod
    def initialize(cur_neurons: int, next_neurons: int) -> np.ndarray:
        assert cur_neurons * next_neurons > 0
        sacle = 2 * np.sqrt(6) / (cur_neurons + next_neurons)
        return (np.random.random(size=cur_neurons) - 0.5) * scale