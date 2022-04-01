import numpy as np
from copy import copy, deepcopy
from typing import Sequence
from abc import ABC, abstractmethod
from .optimizers import Optimizer
from .layers import Dense
from .losses import Loss
import pdb


class Model(ABC):
    @abstractmethod
    def fit(self, train_samples: np.ndarray, train_labels: np.ndarray):
        pass
    @abstractmethod
    def predict(self, test_samples: np.ndarray) -> np.ndarray:
        pass


class Classifier(Model):
    @abstractmethod
    def predict_proba(test_samples: np.ndarray) -> np.ndarray:
        pass
    
    
class MNISTDense(Model):
    def __init__(self,
                 input_size: int,
                 layers_sizes: Sequence[int],
                 initializers_classes: Sequence[type],
                 activations_classes: Sequence[type],
                 optimizer: Optimizer,
                 loss: type):
        assert len(layers_sizes) == len(initializers_classes) == len(activations_classes)
        assert issubclass(loss, Loss)
        self.input_size = input_size
        self.loss = loss
        self.layers_sizes = list(copy(layers_sizes))
        self.layers = []
        for i, (size, initializer, activator) in enumerate(zip(layers_sizes,
                                                            initializers_classes,
                                                            activations_classes)):
            self.layers.append(
                Dense(
                    order_ind=i,
                    size=size,
                    prev_size=layers_sizes[i-1] + 1 if i > 0 else input_size +1,
                    next_size=layers_sizes[i+1] if i + 1 < len(layers_sizes) else 0,
                    initializer_class=initializer,
                    activation_class=activator,
                    optimizer=deepcopy(optimizer)
                )
            )
    
    def fit(self, epoch: int, train_samples: np.ndarray, train_labels: np.ndarray):
        losses = np.empty((epoch, len(train_labels)), dtype=np.float32)
        layers_inputs = np.empty((max(self.layers_sizes + [self.input_size]) + 1, len(self.layers) + 1), dtype=np.float32)
        layers_inputs[self.layers_sizes, range(len(self.layers_sizes))] = 1  # значения входов для смещения
        for e in range(epoch):
            for i, (sample, target) in enumerate(zip(train_samples, train_labels)):    
                layers_inputs[:self.input_size, 0] = sample.reshape((1,-1))
                for j, (layer, input_size, output_size) in enumerate(zip(self.layers, [self.input_size] + self.layers_sizes[:-1], self.layers_sizes )): # распространение вперёд
                    layers_inputs[:output_size, j + 1] = layer.forward(layers_inputs[:input_size+1, j].reshape((1,-1)))
                output = layers_inputs[:self.layers_sizes[-1], -1]
                losses[e, i] = self.loss.calc(output, target)
                error_grad = self.loss.grad(output, target)
                for j, (layer, inp_size) in enumerate(zip(self.layers[::-1], self.layers_sizes[-2::-1] + [self.input_size])): # распространение назад
                    inputs = layers_inputs[: inp_size + 1, -(j+2)].reshape((1,-1))
                    error_grad = layer.backward(inputs, error_grad)
        return losses

    def predict(self, test_samples: np.ndarray) -> np.ndarray:
        for sample in enumerate(test_samples):
            output = np.block([sample.reshape((1,-1)), 1])
            for layer in self.layers: # распространение вперёд
                output = layer.forward(np.block([output, 1]))
        return output 
