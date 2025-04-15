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
        if isinstance(data, (float, int)):
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

        dt = torch.tensor(dt)
        dw = torch.sqrt(dt) * torch.randn((sample_count, time_steps, x.size(0)))
        x_paths = self._build_path(x, dt, dw)
        return x_paths, dw

    def _build_path(self, x: torch.Tensor, dt: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        """
        Build a simulated path for the SDE using the Euler-Maruyama scheme.

        Parameters
        ----------
        x : torch.Tensor
            A 1D tensor representing the initial state of the SDE (of shape (d,)).
        dt : torch.Tensor
            The time increment for each Euler step.
        dw : torch.Tensor
            A 3D tensor containing the Brownian increments with shape (n, n_steps, d).

        Returns
        -------
        torch.Tensor
            A tensor of simulated SDE paths with shape (n, n_steps + 1, d).
        """
        n, n_steps, d = dw.shape

        paths = torch.zeros((n, n_steps + 1, d))
        paths[:, 0] = x

        for i in range(n_steps):
            drift = self.drift(i * dt, paths[:, i])
            diffusion = self.diffusion(i * dt, paths[:, i])
            paths[:, i + 1] = paths[:, i] + drift * dt + torch.matmul(dw[:, i], diffusion.mT)

        return paths
