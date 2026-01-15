from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class GreedyDecoderConfig:
    blank_idx: int = 0


def ctc_collapse(indices: list[int], *, blank_idx: int) -> list[int]:
    out: list[int] = []
    prev = None
    for idx in indices:
        if idx == blank_idx:
            prev = idx
            continue
        if prev == idx:
            continue
        out.append(idx)
        prev = idx
    return out


def greedy_decode(
    log_probs: torch.Tensor,  # [T,B,C]
    idx2char: Sequence[str],
    cfg: GreedyDecoderConfig = GreedyDecoderConfig(),
) -> list[str]:
    """
    Greedy CTC decode: argmax per timestep, then collapse repeats and blanks.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected log_probs [T,B,C], got {tuple(log_probs.shape)}")

    preds = torch.argmax(log_probs, dim=-1)  # [T,B]
    T, B = preds.shape
    results: list[str] = []

    for b in range(B):
        seq = preds[:, b].tolist()
        collapsed = ctc_collapse(seq, blank_idx=cfg.blank_idx)
        text = "".join(idx2char[i] for i in collapsed)
        results.append(text)

    return results

