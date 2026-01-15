from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WordAccStats:
    correct: int
    total: int

    @property
    def acc(self) -> float:
        denom = max(1, self.total)
        return float(self.correct) / float(denom)


def word_acc_corpus(preds: Iterable[str], gts: Iterable[str]) -> WordAccStats:
    correct = 0
    total = 0
    for p, g in zip(preds, gts):
        total += 1
        if p == g:
            correct += 1
    return WordAccStats(correct=correct, total=total)

