import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from .activations import Activation
from .optimizers import Optimizer
from .initializers import Initializer

   
class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def backward(self, inputs: np.ndarray, error_grad_mat: np.ndarray) -> np.ndarray:
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
        self.weights = self.initializer.initialize(prev_size + 1, self.size, next_size) #(n_inputs, n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray (batch_size, n_input)
            return shape == (batch_size, n_neurons)"""
        # Добавим к входным векторам единицы для учета веса смещения
        w_inputs = add_unit_h(inputs)
        return self.activation.calc(w_inputs.dot(self.weights))

    def backward(self, inputs: np.ndarray, outputs: np.ndarray, error_grad_mat: np.ndarray, l1:np.float32 = 0.001, l2:np.float32 = 0.001) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять веса
            inputs.shape == (batch_size, n_prev_neurons),
            outputs.shape == (batch_size, n_neurons),
            error_grad_mat.shape == (batch_size, n_neurons),
            return.shape == (batch_size, n_neurons)
        """
        batch_size = inputs.shape[0]
        # Добавим к входным векторам единицы для учета веса смещения
        w_inputs = add_unit_h(inputs)
        # Найдем градиент ошибки по входам функции активации
        grad_out_by_summator = self.activation.error_back_prop(outputs, error_grad_mat)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = grad_out_by_summator.dot(self.weights[:-1, :].T)
        # Найдем матрицу производных ошибки по весам нейронов текущего слоя
        error_der_matrix = w_inputs.T.dot(error_grad_mat) / batch_size + 2 * l2 * self.weights + l1 * (self.weights > 0)
        # Произведем шаг оптимизации весов по найенным производным
        self.weights = self.optimizer.optimize(self.weights, error_der_matrix)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer


class BatchNormalizer(Layer):
    def __init__(self, size: np.int32, optimizer: Optimizer, epsilon: np.float32 = 10**(-8)):
        self.scale = np.ones((1, size), dtype=np.float32)
        self.bias = np.zeros((1, size), dtype=np.float32)        
        self.eps = epsilon
        self.optimizer_scale = optimizer
        self.optimizer_bias = deepcopy(optimizer)

    def forward(self, inputs: np.ndarray):
        """inputs: np.ndarray (batch_size, n_prev_neurons)
            return shape == (batch_size, n_neurons)"""
        self.mean = np.expand_dims(inputs.mean(axis=0), 0)
        self.norm = 1 / (np.expand_dims(inputs.std(axis=0), 0) + self.eps)
        return self.scale * self.norm *(inputs - self.mean)  + self.bias

    def backward(self, inputs: np.ndarray, outputs: np.ndarray, error_grad_mat: np.ndarray, l1:np.float32 = 0.001, l2:np.float32 = 0.001) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять параметры масштаба и смещения
            inputs.shape == (batch_size, n_neurons),
            outputs.shape == (batch_size, n_neurons),
            error_grad.shape == (batch_size, n_neurons),
            return.shape == (batch_size, n_neurons)
        """
        batch_size = inputs.shape[0]
        # Найдем градиент ошибки по нормализованным входам
        grad_by_normalized = error_grad_mat * self.scale
        # Найдем градиент ошибки по дисперсии
        grad_by_disp = -0.5 * np.power(self.norm, 3) * np.expand_dims((grad_by_normalized * (inputs - self.mean)).sum(axis=0), 0)
        # Найдем градиент ошибки по мат. ожиданию
        grad_by_mean = -self.norm * np.expand_dims((grad_by_normalized).sum(axis=0), 0) -2 * grad_by_disp * np.expand_dims((inputs - self.mean).sum(axis=0), 0) / (batch_size - 1)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = self.norm * grad_by_normalized + grad_by_disp * 2 * (inputs - self.mean) / (batch_size - 1) + grad_by_mean / batch_size
        # Найдем градиент ошибки по коэфициенту сдвига
        error_grad_by_bias = np.expand_dims(error_grad_mat.sum(axis=0), 0)
        # Произведем шаг оптимизации коэффициента сдвига
        self.bias = self.optimizer_bias.optimize(self.bias, error_grad_by_bias)
        # Найдем градиент ошибки по коэфициенту масштаба        
        error_grad_by_scale = np.expand_dims((self.norm * (inputs - self.mean) * error_grad_mat).sum(axis=0), 0)
        # Произведем шаг оптимизации коэффициента масштаба
        self.scale = self.optimizer_scale.optimize(self.scale, error_grad_by_scale)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer

def add_unit_h(source: np.ndarray):
    size = source.shape[0]
    return np.hstack((source, np.ones((size, 1))))