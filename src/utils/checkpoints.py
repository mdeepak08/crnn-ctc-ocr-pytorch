from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Checkpoint:
    model: dict[str, Any]
    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None
    epoch: int = 0
    step: int = 0
    best_metric: float | None = None
    config: dict[str, Any] | None = None
    vocab: dict[str, Any] | None = None


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(ckpt), path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

