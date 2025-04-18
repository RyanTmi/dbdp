from ..models import DBDPModel

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import math


class DBDPSolver:
    """The base PDE solver using Deep Backward Dynamic Programming scheme"""

    def __init__(self, model: DBDPModel, dt: float, time_steps: int, network: nn.ModuleList):
        """
        Parameters
        ----------
        model : DBDPModel
            The model defining the PDE to be solved.
        dt : float
            The time step size.
        time_steps : int
            The number of time steps used to discretize the interval [0,T].
        network: nn.ModuleList
            The list of network element, can be either `DBDP1NetworkElement`,
            DBDP2NetworkElement or `RDBDPNetworkElement`.
        """
        self._model = model
        self._time_steps = time_steps
        self._dt = torch.tensor(dt)
        self._network = network

    @torch.no_grad()
    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Inference method for the DBDP scheme.

        For a given time t and state x, the corresponding u approximation using linear interpolation is returned.
        If t is the terminal time, the terminal condition g(x) is used.

        Parameters
        ----------
        t : float
            The time at which to evaluate the solution.
        x : torch.Tensor
            The state input.

        Returns
        -------
        torch.Tensor
            The approximated solution u(t,x).
        """
        dt = self._dt.item()
        n = self._time_steps

        i_lower = int(math.floor(t / dt))
        if i_lower >= n:
            return self._model.g(x)

        t_lower = i_lower * dt
        lbd = (t - t_lower) / dt

        if lbd == 0.0:
            # Exactly on a grid point
            net = self._network[i_lower]
            net.eval()
            return net.u_network(x)
        else:
            # Interpolate between i_lower and i_lower + 1
            net0 = self._network[i_lower]
            net1 = self._network[i_lower + 1]
            net0.eval()
            net1.eval()

            u0 = net0.u_network(x)
            u1 = net1.u_network(x)
            return (1.0 - lbd) * u0 + lbd * u1

    def train(
        self,
        datas: torch.Tensor,
        dw: torch.Tensor,
        *,
        num_epochs: int,
        batch_size: int,
        lr: float = 1e-2,
        device: torch.device = torch.device("cpu"),
        step: int = -1,  # For debug purposes
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the DBDP scheme using the provided data and Brownian increments.

        The training is performed backward in time (from terminal to initial time).
        A warm-start strategy is used by initializing the network for the current time step
        with the weights from the subsequent time step.

        Parameters
        ----------
        datas : torch.Tensor
            The tensor of state paths.
        dw : torch.Tensor
            The tensor of Brownian increments.
        num_epochs : int
            The number of epochs to train each time step.
        batch_size : int
            The batch size for the DataLoader.
        lr : float, default=1e-2
            The learning rate for the optimizer.
        device : torch.device, default=torch.device("cpu")
            The device on which to perform the training.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays: the training losses and the testing losses
        """
        trains_losses = np.zeros((self._time_steps, num_epochs))
        tests_losses = np.zeros((self._time_steps, num_epochs))

        # Theses are small enough to fit in GPU's memory, so we can move them to the target device.
        datas = datas.to(device)
        dw = dw.to(device)

        # Network elements (parameters) are moved to the target device.
        self._network.to(device)

        times = torch.linspace(0, self._time_steps * self._dt, self._time_steps)
        # Reverse loop for backward training.
        for time_idx in tqdm(range(self._time_steps - 1, -1, -1), desc="Training"):
            # HACK: We initialize the weights and bias of the neural network to the weights and bias
            # of the previous time step treated: this trick is commonly used in iterative solvers of PDE,
            # and allows us to start with a value close to the solution, hence avoiding local minima
            # which are too far away from the true solution. Besides the number of gradient iterations
            # to achieve is rather small after the first resolution step.
            if time_idx < self._time_steps - 1:
                self._network[time_idx].load_state_dict(self._network[time_idx + 1].state_dict())

            # Training one time step
            train_losses, test_losses = self._train_time_step(
                time_idx,
                times[time_idx],
                datas,
                dw,
                num_epochs,
                batch_size,
                lr,
            )
            trains_losses[time_idx] = train_losses
            tests_losses[time_idx] = test_losses

            # Decrease the learning rate
            lr = max(1e-5, lr * 3 / 4)

            # NOTE: For debug purposes, train only `step` time steps
            if step == self._time_steps - time_idx:
                break

        # Move the networks back to CPU
        self._network.cpu()

        return trains_losses, tests_losses

    def _train_time_step(
        self,
        time_idx: int,
        t: torch.Tensor,
        datas: torch.Tensor,
        dw: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train at a specific time step.

        Parameters
        ----------
        time_idx : int
            The index of the current time step.
        datas : torch.Tensor
            The tensor of state paths.
        dw : torch.Tensor
            The tensor of Brownian increments.
        num_epochs : int
            The number of training epochs for this time step.
        batch_size : int
            The batch size for the DataLoader.
        lr : float
            The learning rate for the optimizer.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays containing the training and testing losses for each epoch.

        Raises
        ------
        ValueError
            - If datas is not a 3D tensor.
            - If the dimension of the scheme does not match the last dimension of datas.
        """
        if datas.dim() != 3:
            raise ValueError(f"datas must be a 3D tensor, but got dimension {datas.dim()}.")
        if datas.shape[2] != self._model.dim:
            raise ValueError(f"Mismatch in state dimension: expected {self._model.dim}, got {datas.shape[2]}.")

        network_elt = self._network[time_idx]
        optimizer = Adam(network_elt.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader, test_loader = self._create_data_loaders(time_idx, datas, dw, batch_size)
        train_losses = np.zeros(num_epochs)
        test_losses = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            # Training
            network_elt.train()
            running_train_loss = 0.0
            for x_batch, y_batch, dw_batch in train_loader:
                optimizer.zero_grad()
                pred = network_elt(t, x_batch, self._dt, dw_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * x_batch.size(0)

            train_loss = running_train_loss / len(train_loader.dataset)
            train_losses[epoch] = train_loss

            # Testing
            network_elt.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch, dw_batch in test_loader:
                    pred = network_elt(t, x_batch, self._dt, dw_batch)
                    loss = criterion(pred, y_batch)

                    running_test_loss += loss.item() * x_batch.size(0)

            test_loss = running_test_loss / len(test_loader.dataset)
            test_losses[epoch] = test_loss

        return train_losses, test_losses

    def _create_data_loaders(
        self,
        time_idx: int,
        datas: torch.Tensor,
        dw: torch.Tensor,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create training and testing DataLoaders for a given time step.

        Parameters
        ----------
        time_idx : int
            The index of the current time step.
        datas : torch.Tensor
            The tensor of state paths.
        dw : torch.Tensor
            The tensor of Brownian increments.
        batch_size : int
            The batch size for the DataLoader.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            A tuple containing the training and testing DataLoaders.
        """
        n_samples = datas.size(0)
        train_size = int(n_samples * 0.8)

        x_train = datas[:train_size, time_idx]
        x_test = datas[train_size:, time_idx]

        u_next = self._model.g if time_idx == self._time_steps - 1 else self._network[time_idx + 1].u_network
        with torch.no_grad():
            y_train = u_next(datas[:train_size, time_idx + 1])
            y_test = u_next(datas[train_size:, time_idx + 1])

        dw_train = dw[:train_size, time_idx]
        dw_test = dw[train_size:, time_idx]

        train_dataset = TensorDataset(x_train, y_train, dw_train)
        test_dataset = TensorDataset(x_test, y_test, dw_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_loader, test_loader

    def save(self, filepath: str) -> None:
        """
        Save all models in the container to a single file.

        Parameters
        ----------
        filepath : str
            The file path to save the checkpoint.
        """
        checkpoint = {}
        for i, network_elt in enumerate(self._network):
            checkpoint[f"network_elt_{i}"] = network_elt.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load models from a checkpoint file.

        Parameters
        ----------
        filepath : str
            The file path from which to load the checkpoint.
        """
        checkpoint = torch.load(filepath, weights_only=True)
        for i, network_elt in enumerate(self._network):
            network_elt.load_state_dict(checkpoint[f"network_elt_{i}"])

        print(f"Model loaded from {filepath}")
