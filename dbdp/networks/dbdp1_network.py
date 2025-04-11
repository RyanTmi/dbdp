from .fnn import FNN
from ..models import DBDPModel

import torch
import torch.nn as nn

from typing import Callable


class DBDP1NetworkElement(nn.Module):
    """Represents a single time-step element in the deep backward dynamic programming scheme."""

    def __init__(self, model: DBDPModel, dt: torch.Tensor):
        """
        Parameters
        ----------
        model : DBDPModel
            The model defining the PDE to be solved.
        dt : torch.Tensor
            The time step size.
        """
        super().__init__()

        self._model = model
        self.dt = dt

        dim = model.dim
        self._u_network = FNN(
            input_dim=dim,
            output_dim=1,
            num_hidden=2,
            hidden_dim=dim + 10,
        )
        self._z_network = FNN(
            input_dim=dim,
            output_dim=dim,
            num_hidden=2,
            hidden_dim=dim + 10,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        y = self._u_network(x)
        z = self._z_network(x)
        return y - self._model.f(t, x, y, z) * self.dt + torch.matmul(z.mT, dw)

    @property
    def u_network(self) -> Callable:
        return self._u_network

    @property
    def z_network(self) -> Callable:
        return self._z_network
