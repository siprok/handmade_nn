import numpy as np
from typing import Callable, Sequence
from abc import ABC, abstractmethod

    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self):
        pass
