import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed Forward Network
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden: int,
        hidden_dim: int,
        activation: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
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
        activation : nn.Module, default=nn.Tanh()
            The activation function to use for the hidden layers.
        use_batch_norm : bool, default=True
            Whether to use batch normalization in each layer.
        """
        super().__init__()

        if use_batch_norm:
            input = [nn.BatchNorm1d(input_dim), nn.Linear(input_dim, hidden_dim), activation]
            hidden = [nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), activation] * (num_hidden - 1)
            output = [nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, output_dim)]
        else:
            input = [nn.Linear(input_dim, hidden_dim), activation]
            hidden = [nn.Linear(hidden_dim, hidden_dim), activation] * (num_hidden - 1)
            output = [nn.Linear(hidden_dim, output_dim)]

        layers = input + hidden + output
        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers(x)
        return x
