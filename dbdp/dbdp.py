from .fnn import FNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from typing import Callable


class DBDP1:
    """Deep Backward Dynamic Programming scheme 1"""

    def __init__(self, f: Callable, g: Callable, maturity: float, time_steps: int, dim: int):
        self._time_steps = time_steps
        self._dt = maturity / time_steps
        self._dim = dim
        self._f = f

        self._u_approximators = [FNN(dim, 1, 2, dim + 10) for _ in range(time_steps)] + [g]
        self._z_approximators = [FNN(dim, dim, 2, dim + 10) for _ in range(time_steps)] + [None]

        self._scalers = [None] * time_steps

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        time_idx = int(round(t / self._dt)) + 1

        # If this is the terminal time, just call g(x) (no scaling needed).
        if time_idx == self._time_steps:
            return self._u_approximators[-1](x)

        x_norm = self._scale_input(x, time_idx)

        u_val = self._u_approximators[time_idx](x_norm)
        return u_val

    def train(
        self,
        datas: torch.Tensor,
        dw: torch.Tensor,
        num_epochs: int,
        batch_size: int = 512,
    ) -> tuple[list, list]:
        trains_losses = []
        tests_losses = []

        for time_idx in tqdm(range(self._time_steps - 1, -1, -1), desc="Training"):
            # HACK: We initialize the weights and bias of the neural network to the weights and bias
            # of the previous time step treated: this trick is commonly used in iterative solvers of PDE,
            # and allows us to start with a value close to the solution, hence avoiding local minima
            # which are too far away from the true solution. Besides the number of gradient iterations
            # to achieve is rather small after the first resolution step.
            if time_idx < self._time_steps - 1:
                self._u_approximators[time_idx].load_state_dict(self._u_approximators[time_idx + 1].state_dict())
                self._z_approximators[time_idx].load_state_dict(self._z_approximators[time_idx + 1].state_dict())

            losses = self._train_time_step(time_idx, datas, dw, num_epochs, batch_size)
            trains_losses.append(losses[0])
            tests_losses.append(losses[1])

        return trains_losses[::-1], tests_losses[::-1]

    def save(self, filepath: str) -> None:
        """
        Save all models in the container to a single file.

        Parameters
        ----------
        filepath : str
            The file path to save the checkpoint.
        """

        checkpoint = {}
        checkpoint["scalers"] = self._scalers
        for i in range(self._time_steps):
            checkpoint[f"u_{i}"] = self._u_approximators[i].state_dict()
            checkpoint[f"z_{i}"] = self._z_approximators[i].state_dict()

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
        self._scalers = checkpoint["scalers"]
        for i in range(self._time_steps):
            self._u_approximators[i].load_state_dict(checkpoint[f"u_{i}"])
            self._z_approximators[i].load_state_dict(checkpoint[f"z_{i}"])

        print(f"Model loaded from {filepath}")

    def _create_data_loaders(
        self,
        time_idx: int,
        datas: torch.Tensor,
        dw: torch.Tensor,
        u_next: Callable,
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Create training and testing DataLoaders for a given time step. Uses an 80-20% train-test split."""

        n_samples = datas.size(0)
        train_size = int(n_samples * 0.8)

        # TODO: Add min-max data normalization.
        x_train = datas[:train_size, time_idx].unsqueeze(1)
        x_test = datas[train_size:, time_idx].unsqueeze(1)

        dw_train = dw[:train_size, time_idx].unsqueeze(1)
        dw_test = dw[train_size:, time_idx].unsqueeze(1)

        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std = x_train.std(dim=0, keepdim=True)
        self._scalers[time_idx] = (x_mean, x_std)

        x_train_norm = self._scale_input(x_train, time_idx)
        x_test_norm = self._scale_input(x_test, time_idx)

        with torch.no_grad():
            if time_idx < self._time_steps - 1:
                y_train = u_next(self._scale_input(datas[:train_size, time_idx + 1].unsqueeze(1), time_idx + 1))
                y_test = u_next(self._scale_input(datas[train_size:, time_idx + 1].unsqueeze(1), time_idx + 1))
            else:
                y_train = u_next(datas[:train_size, time_idx + 1].unsqueeze(1))
                y_test = u_next(datas[train_size:, time_idx + 1].unsqueeze(1))

        train_dataset = TensorDataset(x_train, x_train_norm, y_train, dw_train)
        test_dataset = TensorDataset(x_test, x_test_norm, y_test, dw_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def _train_time_step(
        self,
        time_idx: int,
        datas: torch.Tensor,
        dw: torch.Tensor,
        num_epochs: int,
        batch_size: int,
    ) -> None:
        assert datas.dim() - 1 == self._dim, "Mismatch in state dimension."

        def F(t, x, y, z, h, d):
            return y - self._f(t, x, y, z) * h + torch.matmul(z.transpose(-2, -1), d)

        u_model = self._u_approximators[time_idx]
        z_model = self._z_approximators[time_idx]

        u_optimizer = torch.optim.Adam(u_model.parameters(), lr=0.001)
        z_optimizer = torch.optim.Adam(z_model.parameters(), lr=0.001)

        next_u = self._u_approximators[time_idx + 1]

        train_loader, test_loader = self._create_data_loaders(time_idx, datas, dw, next_u, batch_size)

        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            # --- Training Phase ---
            running_train_loss = 0.0

            for x_batch, x_batch_norm, y_batch, dw_batch in train_loader:
                u_optimizer.zero_grad(set_to_none=True)
                z_optimizer.zero_grad(set_to_none=True)

                u_output = u_model(x_batch_norm)
                z_output = z_model(x_batch_norm)

                pred = F(time_idx * self._dt, x_batch, u_output, z_output, self._dt, dw_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                u_optimizer.step()
                z_optimizer.step()

                running_train_loss += loss.item() * x_batch.size(0)

            train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # --- Evaluation Phase ---
            running_test_loss = 0.0
            with torch.no_grad():
                for x_batch, x_batch_norm, y_batch, dw_batch in test_loader:
                    u_output = u_model(x_batch_norm)
                    z_output = z_model(x_batch_norm)
                    pred = F(time_idx * self._dt, x_batch, u_output, z_output, self._dt, dw_batch)
                    loss = criterion(pred, y_batch)

                    running_test_loss += loss.item() * x_batch.size(0)

            test_loss = running_test_loss / len(test_loader.dataset)
            test_losses.append(test_loss)

        #     if (epoch + 1) % 50 == 0:
        #         print(f"Epoch {epoch + 1}/{num_epochs} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")

            if train_loss < 0.005:
                # print(f"Took {epoch + 1} epochs")
                break
        # else:
        #     print("Not converged")

        return train_losses, test_losses

    def _scale_input(self, x: torch.Tensor, time_idx: int) -> torch.Tensor:
        """
        Scale x using the stored (x_min, x_max) for the given time index.
        """
        x_mean, x_std = self._scalers[time_idx]
        return (x - x_mean) / x_std
