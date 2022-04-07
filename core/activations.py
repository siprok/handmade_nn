import numpy as np
import pdb
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        pass
        

class ReLu(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0) * inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        return (outputs > 0) * error_grad


class Tanh(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        derivative = 1 - np.power(outputs, 2)
        return  derivative * error_grad


class Softmax(Activation):
    @staticmethod
    def calc(inputs: np.ndarray, epsilon: np.float32 = 10**(-8)) -> np.ndarray:
        """outputs.shape(batch_size, n_neurons)"""
        # powered = np.exp(inputs)
        # return powered / np.expand_dims(powered.sum(axis=1),1)
        maximum = np.expand_dims(inputs.max(axis=1),1)
        ln_of_sum = maximum + np.log(np.expand_dims(np.exp(inputs - maximum).sum(axis=1),1) + epsilon)
        ln_result = inputs - ln_of_sum
        return np.exp(ln_result)
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        """outputs.shape(batch_size, n_neurons)
        c  error_grad.shape == (batch_size, n_neurons)
           return shape == (batch_size, n_neurons)"""
        assert np.array_equal(outputs.shape, error_grad.shape) 
        batch_size, n_neurons = outputs.shape
        der_matrix = np.empty(shape=(batch_size, n_neurons, n_neurons), dtype=np.float32)  # накопитель для матрицы производных функции активации по выходам сумматора
        der_matrix[:, range(n_neurons), range(n_neurons)] = (outputs - np.power(outputs, 2))  # заполнение главной диагонали
        rows, columns = np.triu_indices(n=n_neurons, k=1)  # индекс строк и столбцов верхнетреугольным матриц для каждого элемента в пакете
        triag_der_values = - outputs[:, rows] * outputs[:, columns]  # внедиагональные значения производной
        der_matrix[:, rows, columns] = triag_der_values
        der_matrix[:, columns, rows] = triag_der_values
        return (der_matrix * np.expand_dims(error_grad,1)).sum(axis=2)

class Linear(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        return error_grad
        