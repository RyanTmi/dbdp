from .sde import SDE

import torch


# TODO: Work only for scalar SDE
class EulerScheme:
    def __init__(self, sde: SDE) -> None:
        self._sde = sde

    def build_path(self, x: torch.Tensor, dt: float, dw: torch.Tensor) -> torch.Tensor:
        paths = torch.zeros((dw.shape[0], dw.shape[1] + 1))
        paths[:, 0] = x
        for i in range(dw.shape[1]):
            drift = self._sde.drift(i * dt, paths[:, i])
            diffusion = self._sde.diffusion(i * dt, paths[:, i])
            paths[:, i + 1] = paths[:, i] + drift * dt + diffusion * dw[:, i]

        return paths


# NOTE: Maybe a good idea to compare with Milstein schemek
