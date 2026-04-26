"""Reproducibility utilities for setting random seeds."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic mode (slower but reproducible).
            Sets cudnn.deterministic=True and cudnn.benchmark=False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
