import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        pass
        

class ReLu(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0) * inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1 and np.array_equal(outputs.shape, error_grad.shape) 
        return (outputs > 0) * error_grad


class Tanh(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1 and np.array_equal(outputs.shape, error_grad.shape) 
        derivative = 1 - np.power(outputs, 2)
        return  derivative * error_grad


class Softmax(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        powered = np.exp(inputs)
        return powered / powered.sum()
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1 and np.array_equal(outputs.shape, error_grad.shape) 
        size = outputs.shape[0]
        der_matrix = np.empty(shape=(size, size), dtype=np.float32)
        der_matrix[range(size), range(size)] = outputs - np.power(outputs, 2)
        rows, columns = np.triu_indices(n=size, k=1)
        triag_der_values = - outputs[rows] * outputs[columns]
        der_matrix[rows, columns] = triag_der_values
        der_matrix[columns, rows] = triag_der_values
        return der_matrix.dot(error_grad)

class Linear(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1 and np.array_equal(outputs.shape, error_grad.shape) 
        return error_grad
        