import torch

from typing import Protocol


# TODO: Work only for scalar SDE
class SDE(Protocol):
    def drift(self, t: float, x: torch.Tensor) -> torch.Tensor: ...

    def diffusion(self, t: float, x: torch.Tensor) -> torch.Tensor: ...

    @property
    def dim(self) -> int: ...


def build_path(sde: SDE, x: torch.Tensor, dt: float, dw: torch.Tensor) -> torch.Tensor:
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

    Raises
    ------
    ValueError
        - If x is not a 1D tensor.
        - If dw is not a 3D tensor.
        - If the dimension of x does not match the last dimension of dw.
        - If the SDE dimension does not match the dimension of x.
    """
    if x.dim() != 1:
        raise ValueError(f"x0 should have dimension 1, but got {x.dim()} instead.")
    if dw.dim() != 3:
        raise ValueError(f"dw should have dimension 3, but got {dw.dim()} instead.")
    if x.shape[0] != dw.shape[2]:
        raise ValueError(f"Mismatch in dimension: x0 has shape {x.shape[0]} but dw's last dimension is {dw.shape[2]}.")
    if sde.dim != x.dim():
        raise ValueError(f"Mismatch in SDE dimension: SDE.dim is {sde.dim} but x0 has dimension {x.dim()}.")

    paths = torch.zeros((dw.shape[0], dw.shape[1] + 1, dw.shape[2]))
    paths[:, 0] = x

    for i in range(dw.shape[1]):
        drift = sde.drift(i * dt, paths[:, i])
        diffusion = sde.diffusion(i * dt, paths[:, i])
        paths[:, i + 1] = paths[:, i] + drift * dt + diffusion * dw[:, i]

    return paths
