"""Reproducibility utilities for random seed management."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Additional deterministic settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng(seed: int) -> np.random.Generator:
    """
    Create a NumPy random generator with fixed seed.

    Args:
        seed: Random seed value

    Returns:
        NumPy random generator
    """
    return np.random.default_rng(seed)
