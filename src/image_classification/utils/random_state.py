import logging
import os
import random
import numpy as np
import torch


def init_random_state(seed: int = 42, force_deterministic: bool = False) -> int:
    """
    Initialize the random state for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation.
        force_deterministic (bool): If True, Set python's hash seed and torch's cudnn settings for deterministic behavior.
                                 Defaults to False.

    Returns:
        int: The seed value used for initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if force_deterministic:
        # Set the environment variable for Python's hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Set torch's cudnn settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f"Random state initialized with seed:\n\t{seed}")
    return seed
