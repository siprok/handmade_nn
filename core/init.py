import numpy as np
from typing import Callable, Sequence
from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def initialize(self):
        pass
    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass
    
class Activation(ABC):
    @abstractmethod
    def calc(self):
        pass
    @abstractmethod
    def grad(self):
        pass
    
class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def back(self):
        pass
    
class Model(ABC):
    @abstractmethod
    def fit(train_samples: np.ndarray,
            train_labels: np.ndarray):
        pass
    @abstractmethod
    def predict(test_samples: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def predict_proba(test_samples: np.ndarray) -> np.ndarray:
        pass
    
    
class MNISTDense(Model):
    def __init__(self,
                 layers_sizes:Sequence[int],
                 loss: Callable[[np.ndarray, np.ndarray], np.float32],
                 initializer: Initializer,
                 optimizer: Optimizer):
        pass
    
    def fit(train_samples: np.ndarray,
            train_labels: np.ndarray):
        pass
    
    def predict(test_samples: np.ndarray) -> np.ndarray:
        pass
    
    def predict_proba(test_samples: np.ndarray) -> np.ndarray:
        pass