import torch

from typing import Callable


# TODO: Work only for scalar SDE
class SDE:
    """
    drift: [0,T] x R^d -> R^d

    diffusion: [0,T] x R^d -> R^{dxd}
    """

    def __init__(self, drift: Callable, diffusion: Callable, dim: int) -> None:
        self._drift = drift
        self._diffusion = diffusion
        self._dim = dim

    def drift(self, t, x: torch.Tensor) -> torch.Tensor:
        return self._drift(t, x)

    def diffusion(self, t, x: torch.Tensor) -> torch.Tensor:
        return self._diffusion(t, x)

    @property
    def dim(self) -> int:
        return self._dim
