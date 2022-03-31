import numpy as np
from abc import ABC, abstractmethod
from .activations import Activation
from .optimizers import Optimizer
from .initializers import Initializer
   
class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def backward(self, error_grad: np.ndarray) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self,
                 size: int,
                 prev_size: int,
                 next_size: int,
                 initializer_class: type,
                 activation_class: type,
                 optimizer: Optimizer):
        assert issubclass(activation_class, Activation)
        assert issubclass(initializer_class, Initializer)
        self.size = size
        self.initializer = initializer_class
        self.activation = activation_class
        self.optimizer = optimizer
        self.outputs = np.empty((self.size, 1), dtype=np.float32)
        self.weights = self.initializer.initialize(prev_size, self.size, next_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray (1, n_input)"""
        assert len(inputs.shape) == 2 and inputs.shape[0] == 1
        self.outputs = self.activation((inputs * self.weights).sum(axis=0))
        return self.outputs

    def backward(self, error_grad: np.ndarray) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять веса"""
        # Найдем градиент ошибки по входам функции активации
        grad_out_by_summator = self.activation.error_back_prop(self.outputs, error_grad)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = self.weights.dot(grad_out_by_summator)
        # Найдем матрицу производных ошибки по весам нейронов текущего слоя
        error_der_matrix = self.outputs.T.dot(error_grad.T)
        # Произведем шаг оптимизации весов по найенным производным
        self.weights = self.optimizer(self.weights, error_der_matrix)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer