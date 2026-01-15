from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    prefer: str = "auto"  # auto|cpu|cuda|mps


def get_device(cfg: DeviceConfig) -> torch.device:
    if cfg.prefer == "cpu":
        return torch.device("cpu")
    if cfg.prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.prefer == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    # torch.cuda.amp is well supported; MPS AMP support varies by version.
    return device.type == "cuda"

