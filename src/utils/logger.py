from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


@dataclass
class TBLogger:
    log_dir: str

    def __post_init__(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(self.log_dir)

    def add_scalars(self, main_tag: str, scalars: dict[str, float], step: int) -> None:
        self._writer.add_scalars(main_tag, scalars, global_step=step)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, global_step=step)

    def add_text(self, tag: str, text_string: str, step: int = 0) -> None:
        self._writer.add_text(tag, text_string, global_step=step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def print_once(msg: str) -> None:
    # Simple wrapper for consistent formatting; can be expanded later.
    print(msg, flush=True)

