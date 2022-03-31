import numpy as np
from abc import ABC, abstractmethod

    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        pass
    @abstractmethod
    def reboot(self):
        pass
