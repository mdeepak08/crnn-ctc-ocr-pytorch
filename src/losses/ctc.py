from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CTCLossConfig:
    blank_idx: int = 0
    zero_infinity: bool = True


class CTCLoss(nn.Module):
    def __init__(self, cfg: CTCLossConfig):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.CTCLoss(blank=cfg.blank_idx, zero_infinity=cfg.zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,  # [T,B,C]
        targets: torch.Tensor,  # [sum(target_lengths)]
        input_lengths: torch.Tensor,  # [B]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        return self.loss(log_probs, targets, input_lengths, target_lengths)

