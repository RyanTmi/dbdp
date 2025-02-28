import torch

from typing import Callable

type SDEFunction = Callable[[float, torch.Tensor], torch.Tensor]


# TODO: Work only for scalar SDE
class SDE:
    """
    Class representing a Stochastic Differential Equation (SDE).

    The SDE is characterized by two functions:
        - drift: [0, T] x R^d → R^d,
        - diffusion: [0, T] x R^d → R^(d x d).
    """

    def __init__(self, drift: SDEFunction, diffusion: SDEFunction, dim: int):
        """
        Parameters
        ----------
        drift : SDEFunction
            The drift function of the SDE.
        diffusion : SDEFunction
            The diffusion function of the SDE.
        dim : int
            The dimension d of the state space R^d.
        """
        self._drift = drift
        self._diffusion = diffusion
        self._dim = dim

    def drift(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self._drift(t, x)

    def diffusion(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self._diffusion(t, x)

    @property
    def dim(self) -> int:
        return self._dim
