from ..networks.rdbdp_network import RDBDPNetworkElement
from ..models import DBDPModel
from .base_solver import DBDPSolver

import torch.nn as nn


class RDBDPSolver(DBDPSolver):
    """PDE solver using reflected Deep Backward Dynamic Programming scheme"""

    def __init__(self, model: DBDPModel, dt: float, time_steps: int):
        """
        Parameters
        ----------
        model : DBDPModel
            The model defining the PDE to be solved.
        dt : float
            The time step size.
        time_steps : int
            The number of time steps used to discretize the interval [0,T].
        """
        network = nn.ModuleList([RDBDPNetworkElement(model) for _ in range(time_steps)])
        super().__init__(model, dt, time_steps, network)
