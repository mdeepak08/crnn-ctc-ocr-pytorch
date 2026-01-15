from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CollateConfig:
    cnn_downsample_factor: int = 4  # must match model's width reduction


def collate_ocr_batch(batch: list[dict[str, Any]], cfg: CollateConfig) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,1,H,W]

    target_lengths = torch.tensor([len(b["target"]) for b in batch], dtype=torch.long)
    targets = torch.tensor([t for b in batch for t in b["target"]], dtype=torch.long)

    # Estimate per-sample input length (time steps) based on valid (unpadded) width.
    valid_ws = torch.tensor([int(b["valid_w"]) for b in batch], dtype=torch.long)
    input_lengths = torch.clamp(valid_ws // cfg.cnn_downsample_factor, min=1)

    texts = [b["text"] for b in batch]
    paths = [b["path"] for b in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "texts": texts,
        "paths": paths,
    }

