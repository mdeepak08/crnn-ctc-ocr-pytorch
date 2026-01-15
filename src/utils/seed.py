from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


def seed_everything(cfg: SeedConfig) -> None:
    """
    Best-effort determinism across CPU/CUDA. Some ops can still be nondeterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If a nondeterministic op is used, this will error (good for reproducibility).
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this fully.
            pass

