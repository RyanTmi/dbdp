from ..networks.dbdp2_network import DBDP2NetworkElement
from ..models import DBDPModel
from .base_solver import DBDPSolver

import torch.nn as nn


class DBDP2Solver(DBDPSolver):
    """PDE solver using Deep Backward Dynamic Programming scheme 2"""

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
        network = nn.ModuleList([DBDP2NetworkElement(model) for _ in range(time_steps)])
        super().__init__(model, dt, time_steps, network)
