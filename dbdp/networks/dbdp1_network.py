from .fnn import FNN

import torch
import torch.nn as nn

from typing import Callable


class DBDP1CellNetwork(nn.Module):
    """Represents a single time-step cell in the deep backward dynamic programming scheme."""

    def __init__(self, f: Callable, dt: float, dim: int):
        """
        Parameters
        ----------
        f : Callable
            A function f(t, x, u, z) defining the non-linearity in the PDE.
        dt : float
            The time increment used in the Euler update.
        dim : int
            The dimension of the state space.
        """
        super().__init__()

        self._f = f
        self._dt = dt
        self._dim = dim

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

    def forward(self, t: float, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        y = self._u_network(x)
        z = self._z_network(x)
        return y - self._f(t, x, y, z) * self._dt + torch.matmul(z.transpose(-2, -1), dw)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def u_network(self) -> Callable:
        return self._u_network

    @property
    def z_network(self) -> Callable:
        return self._z_network
