import numpy as np
from typing import Callable, Sequence
from abc import ABC, abstractmethod

   
class Layer(ABC):
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def back(self):
        pass
