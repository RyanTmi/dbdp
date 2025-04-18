from .feed_forward import FeedForward
from ..models import DBDPModelDynamic

import torch
import torch.nn as nn

from typing import Callable


class DBDP2NetworkElement(nn.Module):
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

    def forward(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        # Only build a graph for x → y → y_grad
        with torch.enable_grad():
            x = x.requires_grad_(True)

            y = self._u_network(x)
            # Compute the gradient of `y` (`u`) w.r.t `x` via autograd.
            y_grad = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
            )[0]

        z = torch.matmul(y_grad, self._model.diffusion(t, x))
        return y - self._model.f(t, x, y, z) * dt + torch.einsum("ij,ij->i", z, dw).unsqueeze(1)

    @property
    def u_network(self) -> Callable:
        return self._u_network
