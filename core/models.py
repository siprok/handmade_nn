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
    
    def fit(self, epoch: int, batch_size: int,  train_samples: np.ndarray, train_targets: np.ndarray):
        epoch_size = train_targets.shape[0]  # количество элементов из обучающей выборки наодной эпохе обучения
        losses = np.empty((epoch, train_targets.shape[0] // batch_size), dtype=np.float32)  #  накопитель значения функции потерь по пакетам на эпохах обучения
        layers_inputs = np.empty((batch_size,
                                  len(self.layers) + 1,
                                  max(self.layers_sizes + [self.input_size]) + 1),
                                 dtype=np.float32)  # накопитель значений входов слоев сети (batch_size, n_layers + 1, layer_size)
        layers_inputs[:, range(len(self.layers_sizes)), self.layers_sizes] = 1  # значения входов для смещения
        for e in range(epoch): # итерации по эпохам
            order = np.arange(epoch_size)
            np.random.shuffle(order) # перемещаем элементы обучающей выборки
            for i, start in enumerate(range(0, epoch_size, batch_size)):  # итерируемся по пакетам
                stop = start + batch_size 
                batch_indxs = order[start: stop]
                if batch_indxs.size < batch_size:
                    continue
                samples = train_samples[batch_indxs]  # входные значения пакета
                targets = train_targets[batch_indxs]  # целевые значения пакета
                layers_inputs[:, 0, :self.input_size] = samples  # установим входные значения для первого слоя
                for j, (layer, input_size, output_size)\
                    in enumerate(zip(self.layers, [self.input_size] + self.layers_sizes[:-1], self.layers_sizes )): # распространение вперёд
                    layers_inputs[:, j + 1, :output_size] = layer.forward(layers_inputs[:, j, :input_size + 1])
                output = layers_inputs[:, -1, :self.layers_sizes[-1]]
                losses[e, i] = self.loss.calc(output, targets)
                error_grad = self.loss.grad(output, targets)
                for j, (layer, inp_size, out_size)\
                     in enumerate(zip(self.layers[::-1], self.layers_sizes[-2::-1] + [self.input_size], self.layers_sizes[::-1])): # распространение назад
                    inputs = layers_inputs[:, -(j+2), :inp_size + 1]
                    outputs = layers_inputs[:, -(j+1), :out_size + 1]
                    error_grad = layer.backward(inputs=inputs, outputs=outputs, error_grad=error_grad)
        return losses

    def predict(self, test_samples: np.ndarray) -> np.ndarray:
        samples = np.vstack((test_samples, np.ones(test_samples.shape[0]))) 
        for layer in self.layers: # распространение вперёд
            output = layer.forward(samples)
        return output 
