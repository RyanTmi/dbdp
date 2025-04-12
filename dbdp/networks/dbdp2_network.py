from .feed_forward import FeedForward
from ..models import DBDPModelDynamic

import torch
import torch.nn as nn

from typing import Callable


# TODO: Implement this class
class DBDP2CellNetwork(nn.Module):
    """Represents a single time-step cell in the deep backward dynamic programming scheme."""

    ...
