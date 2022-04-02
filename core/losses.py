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
        """prediction.shape == (batch_size, layer_size)"""
        indxs = range(prediction.shape[0])
        return -np.log(prediction[indxs, target]).sum()

    def grad(outputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """outputs.shape == (batch_size, n_classes) target.shape == (batch_size, 1)
            return (batch_size, n_neurons)
        """
        indxs = np.arange(outputs.shape[0])
        error = np.zeros((outputs.shape[0], outputs.shape[1]))
        error[indxs, target] =  - 1 / outputs[indxs, target]
        return error