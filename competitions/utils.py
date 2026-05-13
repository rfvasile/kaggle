import os
import random
from typing import Any

import numpy as np
import torch


def get_reproducible_dataloader_kwargs(seed: int = 0) -> dict[str, Any]:
    """
    Returns a dictionary of arguments to pass to a DataLoader
    to ensure worker-level reproducibility.
    """

    def seed_worker(_: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return {
        "worker_init_fn": seed_worker,
        "generator": g,
    }


def make_reproducible(seed: int = 0) -> None:
    """
    Sets all seeds and configuration flags for reproducibility.
    See: https://docs.pytorch.org/docs/2.11/notes/randomness.html
    """
    # 1. Basic Python and NumPy seeding
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. PyTorch seeding (CPU and all GPUs)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 3. Algorithm Determinism
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 4. CuDNN Backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Reproducibility set with seed: {seed}")
