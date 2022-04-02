import numpy as np
from abc import ABC, abstractmethod
from .activations import Activation
from .optimizers import Optimizer
from .initializers import Initializer
import pdb
   
class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def backward(self, inputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self,
                 order_ind: int,
                 size: int,
                 prev_size: int,
                 next_size: int,
                 initializer_class: type,
                 activation_class: type,
                 optimizer: Optimizer):
        assert issubclass(activation_class, Activation)
        assert issubclass(initializer_class, Initializer)
        self.order_ind = order_ind
        self.size = size
        self.initializer = initializer_class
        self.activation = activation_class
        self.optimizer = optimizer
        self.weights = self.initializer.initialize(prev_size, self.size, next_size) #(n_inputs, n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray (batch_size, n_input)
            return shape == (batch_size, n_neurons)"""
        return self.activation.calc(inputs.dot(self.weights))

    def backward(self, inputs: np.ndarray, outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять веса
            inputs.shape == (batch_size, n_prev_neurons), error_grad.shape == (batch_size, n_neurons), return.shape == (batch_size, n_neurons)
        """
        # Найдем градиент ошибки по входам функции активации
        grad_out_by_summator = self.activation.error_back_prop(outputs[:, :-1], error_grad)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = grad_out_by_summator.dot(self.weights[:-1, :].T)
        # Найдем матрицу производных ошибки по весам нейронов текущего слоя
        error_der_matrix = inputs.T.dot(error_grad)
        # Произведем шаг оптимизации весов по найенным производным
        self.weights = self.optimizer.optimize(self.weights, error_der_matrix)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer