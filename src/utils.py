import random

import numpy as np
import torch


def fix_seed(seed: int = 123321):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Seed {seed} fixed")
