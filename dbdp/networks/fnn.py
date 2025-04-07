import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


class FNN(nn.Module):
    """
    Feedforward Neural Network (FNN).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden: int,
        hidden_dim: int,
        activation: Callable = F.tanh,
    ):
        """
        Parameters
        ----------
        input_dim : int
            The number of input features.
        output_dim : int
            The dimension of the output.
        num_hidden : int
            The number of hidden layers.
        hidden_dim : int
            The number of neurons in each hidden layer.
        activation : Callable, default=F.tanh
            The activation function to use for the hidden layers.
        """
        super().__init__()

        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self._hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self._output_layer = nn.Linear(hidden_dim, output_dim)
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._activation(self._input_layer(x))
        for hidden_layer in self._hidden_layers:
            x = self._activation(hidden_layer(x))

        return self._output_layer(x)
