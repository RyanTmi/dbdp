import torch
import torch.nn as nn


class FNN(nn.Module):
    """Feedforward Neural Network"""

    def __init__(self, input_dim: int, output_dim: int, num_hidden: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.tanh(hidden_layer(x))

        return self.output_layer(x)
