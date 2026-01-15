from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .transforms import OCRPreprocess, OCRTransformConfig


def load_vocab(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    vocab = json.loads(path.read_text())
    if "char2idx" not in vocab or "idx2char" not in vocab:
        raise ValueError(f"Invalid vocab.json at {path}")
    if vocab.get("blank_idx", 0) != 0:
        raise ValueError("This project assumes CTC blank_idx=0.")
    return vocab


@dataclass(frozen=True)
class OCRCsvDatasetConfig:
    csv_path: str
    vocab_path: str
    max_len: int = 25
    lowercase: bool = True
    strict_vocab: bool = True  # drop samples containing chars not in vocab
    image_base_dir: str | None = None  # optional prefix to join with relative paths
    transform: OCRTransformConfig = OCRTransformConfig()


class OCRCsvDataset(Dataset):
    """
    Expects CSV with header containing at least: path,text
    """

    def __init__(self, cfg: OCRCsvDatasetConfig):
        self.cfg = cfg
        self.vocab = load_vocab(cfg.vocab_path)
        self.char2idx: dict[str, int] = {str(k): int(v) for k, v in self.vocab["char2idx"].items()}
        self.blank_idx: int = int(self.vocab.get("blank_idx", 0))

        self.transform = OCRPreprocess(cfg.transform)
        self.samples: list[tuple[str, str]] = []

        self._load_csv()

    def _norm_text(self, s: str) -> str:
        s = s.strip()
        if self.cfg.lowercase:
            s = s.lower()
        return s

    def _load_csv(self) -> None:
        p = Path(self.cfg.csv_path)
        if not p.exists():
            raise FileNotFoundError(p)

        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "path" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(f"CSV must have columns path,text. Found: {reader.fieldnames}")

            for row in reader:
                rel = row["path"]
                text = self._norm_text(row["text"])

                if len(text) == 0 or len(text) > self.cfg.max_len:
                    continue

                if self.cfg.strict_vocab:
                    ok = all(ch in self.char2idx for ch in text)
                    if not ok:
                        continue

                img_path = Path(rel)
                if not img_path.is_absolute():
                    if self.cfg.image_base_dir is not None:
                        img_path = Path(self.cfg.image_base_dir) / img_path
                    else:
                        img_path = p.parent / img_path

                self.samples.append((str(img_path), text))

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples loaded from {p}")

    def __len__(self) -> int:
        return len(self.samples)

    def encode(self, text: str) -> list[int]:
        # blank is reserved at idx=0; char2idx should map chars to >=1
        return [self.char2idx[ch] for ch in text]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, text = self.samples[idx]
        img = Image.open(path).convert("RGB")  # robust to palette/etc
        image_t, valid_w = self.transform(img)
        target = self.encode(text)
        return {
            "image": image_t,
            "text": text,
            "target": target,
            "valid_w": valid_w,
            "path": path,
        }

