from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class CERStats:
    total_edits: int
    total_chars: int

    @property
    def cer(self) -> float:
        denom = max(1, self.total_chars)
        return float(self.total_edits) / float(denom)


def cer_one(pred: str, gt: str) -> CERStats:
    edits = int(Levenshtein.distance(pred, gt))
    chars = len(gt)
    return CERStats(total_edits=edits, total_chars=chars)


def cer_corpus(preds: Iterable[str], gts: Iterable[str]) -> CERStats:
    total_edits = 0
    total_chars = 0
    for p, g in zip(preds, gts):
        s = cer_one(p, g)
        total_edits += s.total_edits
        total_chars += s.total_chars
    return CERStats(total_edits=total_edits, total_chars=total_chars)

