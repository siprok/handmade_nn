import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def calc(prediction: np.ndarray, target: np.ndarray):
        pass
    @abstractmethod
    def grad(outputs: np.ndarray, target: np.ndarray):
        pass


class Crossentropy(Loss):
    def calc(prediction: np.ndarray, target: np.ndarray) -> np.float32:
        return - (target.flatten() * prediction.flatten()).sum() 

    def grad(outputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """outputs (1, n_classes) target (1, n_classes)"""
        ind = np.argwhere(target.flatten() > 0)
        error = np.zeros((target.size(), 1))
        error[ind] = 1 / outputs[ind]
        return error