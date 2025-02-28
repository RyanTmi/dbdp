from .sde import SDE

import torch


# TODO: Work only for scalar SDE
class EulerScheme:
    """
    Class implementing the Euler-Maruyama scheme for a given SDE.

    The EulerScheme class uses a provided SDE object to simulate the
    SDE's path over a time grid and using Brownian motion increments.
    """

    def __init__(self, sde: SDE):
        """
        Parameters
        ----------
        sde : SDE
            An instance of the SDE class.
        """
        self._sde = sde

    def build_path(self, x0: torch.Tensor, dt: float, dw: torch.Tensor) -> torch.Tensor:
        """
        Build a simulated path for the SDE using the Euler-Maruyama scheme.

        Parameters
        ----------
        x0 : torch.Tensor
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
            - If x0 is not a 1D tensor.

            - If dw is not a 3D tensor.

            - If the dimension of x0 does not match the last dimension of dw.

            - If the SDE dimension does not match the dimension of x0.
        """
        if x0.dim() != 1:
            raise ValueError(f"x0 should have dimension 1, but got {x0.dim()} instead.")

        if dw.dim() != 3:
            raise ValueError(f"dw should have dimension 3, but got {dw.dim()} instead.")

        if x0.shape[0] != dw.shape[2]:
            raise ValueError(
                f"Mismatch in dimension: x0 has shape {x0.shape[0]} but dw's last dimension is {dw.shape[2]}."
            )

        if self._sde.dim != x0.dim():
            raise ValueError(f"Mismatch in SDE dimension: SDE.dim is {self._sde.dim} but x0 has dimension {x0.dim()}.")

        paths = torch.zeros((dw.shape[0], dw.shape[1] + 1, dw.shape[2]))
        paths[:, 0] = x0

        for i in range(dw.shape[1]):
            drift = self._sde.drift(i * dt, paths[:, i])
            diffusion = self._sde.diffusion(i * dt, paths[:, i])
            paths[:, i + 1] = paths[:, i] + drift * dt + diffusion * dw[:, i]

        return paths
