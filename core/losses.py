import numpy as np
from abc import ABC, abstractmethod
import pdb


class Loss(ABC):
    @abstractmethod
    def calc(prediction: np.ndarray, target: np.ndarray):
        pass
    @abstractmethod
    def grad(outputs: np.ndarray, target: np.ndarray):
        pass


class Crossentropy(Loss):
    def calc(prediction: np.ndarray, target: np.ndarray) -> np.float32:
        ind = np.argwhere(target.flatten() > 0)[0,0]
        return -np.log(prediction[ind])

    def grad(outputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """outputs (1, n_classes) target (1, n_classes)"""
        # pdb.set_trace()
        ind = np.argwhere(target.flatten() > 0)[0,0]
        error = np.zeros((outputs.size, 1))
        error[ind] =  - 1 / outputs[ind]
        return error