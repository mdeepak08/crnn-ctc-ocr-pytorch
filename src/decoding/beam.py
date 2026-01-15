from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class BeamDecoderConfig:
    blank_idx: int = 0
    beam_width: int = 5


NEG_INF = -1e9


def _log_add(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _topk_beams(beams: dict[tuple[int, ...], tuple[float, float]], k: int) -> dict[tuple[int, ...], tuple[float, float]]:
    # Sort by total probability (log-space)
    items = sorted(
        beams.items(),
        key=lambda kv: _log_add(kv[1][0], kv[1][1]),
        reverse=True,
    )
    return dict(items[:k])


def _ctc_prefix_beam_search_one(
    log_probs_t_c: torch.Tensor,  # [T,C] (log-probs)
    cfg: BeamDecoderConfig,
) -> tuple[int, ...]:
    T, C = log_probs_t_c.shape

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}  # prefix -> (p_blank, p_nonblank)

    for t in range(T):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        lp = log_probs_t_c[t]  # [C]

        beams = _topk_beams(beams, cfg.beam_width)

        for prefix, (p_b, p_nb) in beams.items():
            for c in range(C):
                p = float(lp[c].item())

                if c == cfg.blank_idx:
                    nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (_log_add(nb[0], _log_add(p_b + p, p_nb + p)), nb[1])
                    continue

                end = prefix[-1] if len(prefix) > 0 else None
                new_prefix = prefix + (c,)

                # Extend with non-blank
                nb_new = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                if c == end:
                    # If same as last char, only transitions from blank to nonblank count for extension
                    p_nb_new = _log_add(nb_new[1], p_b + p)
                else:
                    p_nb_new = _log_add(nb_new[1], _log_add(p_b + p, p_nb + p))
                next_beams[new_prefix] = (nb_new[0], p_nb_new)

                # Also allow staying at same prefix if repeated char comes without blank
                if c == end:
                    nb_same = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (nb_same[0], _log_add(nb_same[1], p_nb + p))

        beams = next_beams

    best = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))[0]
    return best


def beam_decode(
    log_probs: torch.Tensor,  # [T,B,C]
    idx2char: Sequence[str],
    cfg: BeamDecoderConfig = BeamDecoderConfig(),
) -> list[str]:
    """
    Simple CTC prefix beam search. Beam width defaults to 5.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected log_probs [T,B,C], got {tuple(log_probs.shape)}")

    T, B, C = log_probs.shape
    out: list[str] = []
    for b in range(B):
        best = _ctc_prefix_beam_search_one(log_probs[:, b, :], cfg)
        out.append("".join(idx2char[i] for i in best))
    return out

