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
        self.outputs = np.empty((self.size, 1), dtype=np.float32)
        self.weights = self.initializer.initialize(prev_size, self.size, next_size) #(n_inputs, n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray (1, n_input)"""
        assert len(inputs.shape) == 2 and inputs.shape[0] == 1
        # pdb.set_trace()
        self.outputs[:] = self.activation.calc(inputs.dot(self.weights)).reshape((-1,1))
        # print(f" {self.order_ind} min output {self.outputs.min()} max output {self.outputs.max()}")
        return self.outputs[:, 0]
        

    def backward(self, inputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять веса"""
        assert len(inputs.shape) == 2 and inputs.shape[0] == 1
        # print(f" {self.order_ind} min error_grad {error_grad.min()} max error_grad {error_grad.max()}")
        # Найдем градиент ошибки по входам функции активации
        grad_out_by_summator = self.activation.error_back_prop(self.outputs, error_grad)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = self.weights[:-1, :].dot(grad_out_by_summator)
        # Найдем матрицу производных ошибки по весам нейронов текущего слоя
        error_der_matrix = inputs.T.dot(error_grad.T)
        # Произведем шаг оптимизации весов по найенным производным
        # pdb.set_trace()
        self.weights = self.optimizer.optimize(self.weights, error_der_matrix)
        # print(f" {self.order_ind} min self.weights {self.weights.min()} max self.weights {self.weights.max()}")
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer