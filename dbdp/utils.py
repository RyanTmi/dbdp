import torch

from typing import Literal


def get_device(device_type: Literal["cpu", "gpu"] = "cpu") -> torch.device:
    if device_type == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            print("GPU not available, using CPU instead.")
            return torch.device("cpu")
    elif device_type == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError("Invalid device type. Use 'cpu' or 'gpu'.")
