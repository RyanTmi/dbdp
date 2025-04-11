import numpy as np
import torch
import torch.nn as nn

from typing import Protocol


class DBDPModelDynamic(Protocol):
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...
    def f(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
    def g(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def dim(self) -> int: ...


class DBDPModel(nn.Module, DBDPModelDynamic):
    def __init__(self):
        super().__init__()

    def make_buffer(self, data: float | torch.Tensor) -> nn.Buffer:
        if isinstance(data, float):
            data = torch.tensor(data)
        return nn.Buffer(data, persistent=False)

    def generate_datas(
        self,
        x: torch.Tensor,
        dt: float,
        time_steps: int,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 1:
            raise ValueError(f"x should have dimension 1, but got {x.dim()} instead.")

        dw = np.sqrt(dt) * torch.randn((sample_count, time_steps, x.size(0)))
        x_paths = self._build_path(x, dt, dw)
        return x_paths, dw

    # TODO: Work only for 1d PDEs. Make this function work with d-dimensional PDEs.
    # Maybe do theses computations on GPU ?
    def _build_path(self, x: torch.Tensor, dt: float, dw: torch.Tensor) -> torch.Tensor:
        """
        Build a simulated path for the SDE using the Euler-Maruyama scheme.

        Parameters
        ----------
        x : torch.Tensor
            A 1D tensor representing the initial state of the SDE (of shape (d,)).
        dt : float
            The time increment for each Euler step.
        dw : torch.Tensor
            A 3D tensor containing the Brownian increments with shape (n, num_steps, d).

        Returns
        -------
        torch.Tensor
            A tensor of simulated SDE paths with shape (n, num_steps + 1, d).
        """

        paths = torch.zeros((dw.shape[0], dw.shape[1] + 1, dw.shape[2]))
        paths[:, 0] = x

        for i in range(dw.shape[1]):
            drift = self.drift(i * dt, paths[:, i])
            diffusion = self.diffusion(i * dt, paths[:, i])
            paths[:, i + 1] = paths[:, i] + drift * dt + diffusion * dw[:, i]

        return paths
