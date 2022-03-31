import numpy as np
from collections import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    @staticmethod
    def calc(prediction: np.ndarray, target: np.ndarray):
        pass
    @abstractmethod
    @staticmethod
    def grad(outputs: np.ndarray, target: np.ndarray):
        pass