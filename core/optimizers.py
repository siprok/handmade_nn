import numpy as np
from abc import ABC, abstractmethod

    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        pass

class OptimizerWithState(Optimizer):
    @abstractmethod
    def reboot(self):
        pass


class GradDesc(Optimizer):
    def __init__(self, learning_rate: np.float32):
        self.lr = learning_rate

    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        return weights_matrix - self.lr * error_der_matrix


class Momentum(OptimizerWithState):
    def __init__(self, learning_rate: np.float32, momentum: np.float32):
        self.lr = learning_rate
        self.m = momentum
        self.change = 0

    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        self.change = self.m * self.change + self.lr * error_der_matrix
        return weights_matrix - self.change

    def reboot(self):
        self.change = 0


class Adagrad(OptimizerWithState):
    def __init__(self, learning_rate: np.float32, epsilon: np.float32 = 10**(-5)):
        self.lr = learning_rate
        self.eps = epsilon
        self.accum = 0
    
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        result = weights_matrix - self.lr / np.sqrt(self.accum + self.eps) * error_der_matrix
        self.accum += np.power(error_der_matrix, 2)
        return result

    def reboot(self):
        self.accum = np.zeros_like(self.accum)



    