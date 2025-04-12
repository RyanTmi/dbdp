from .feed_forward import FeedForward
from ..models import DBDPModelDynamic

import torch
import torch.nn as nn

from typing import Callable


class DBDP1NetworkElement(nn.Module):
    """Represents a single time-step element in the deep backward dynamic programming scheme."""

    def __init__(self, model: DBDPModelDynamic):
        """
        Parameters
        ----------
        model : DBDPModelDynamic
            The model dynamic defining the PDE to be solved.
        """
        super().__init__()

        self._model = model

        dim = model.dim
        self._u_network = FeedForward(
            input_dim=dim,
            output_dim=1,
            num_hidden=2,
            hidden_dim=dim + 10,
        )
        self._z_network = FeedForward(
            input_dim=dim,
            output_dim=dim,
            num_hidden=2,
            hidden_dim=dim + 10,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        y = self._u_network(x)
        z = self._z_network(x)
        return y - self._model.f(t, x, y, z) * dt + torch.matmul(z.mT, dw)

    @property
    def u_network(self) -> Callable:
        return self._u_network

    @property
    def z_network(self) -> Callable:
        return self._z_network
