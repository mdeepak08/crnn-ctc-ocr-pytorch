from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class OCRTransformConfig:
    img_h: int = 32
    img_w: int = 128
    pad_value: int = 0
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


def _to_grayscale_pil(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img


def resize_keep_ratio_pad(
    img: Image.Image, *, img_h: int, img_w: int, pad_value: int = 0
) -> Tuple[Image.Image, int]:
    """
    Resize to fixed height while keeping aspect ratio, then right-pad to img_w.
    Returns (padded_img, valid_width_px).
    """
    img = _to_grayscale_pil(img)

    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: {img.size}")

    new_w = int(round(w * (img_h / float(h))))
    new_w = max(1, min(new_w, img_w))

    img_resized = img.resize((new_w, img_h), resample=Image.BILINEAR)

    canvas = Image.new("L", (img_w, img_h), color=int(pad_value))
    canvas.paste(img_resized, (0, 0))
    return canvas, new_w


def pil_to_normalized_tensor(img: Image.Image, *, mean: float, std: float) -> torch.Tensor:
    """
    Returns float tensor in shape [1, H, W], normalized.
    """
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W] in [0,1]
    t = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
    t = (t - mean) / std
    return t


class OCRPreprocess:
    def __init__(self, cfg: OCRTransformConfig):
        self.cfg = cfg

    def __call__(self, img: Image.Image) -> tuple[torch.Tensor, int]:
        padded, valid_w = resize_keep_ratio_pad(
            img, img_h=self.cfg.img_h, img_w=self.cfg.img_w, pad_value=self.cfg.pad_value
        )
        tensor = pil_to_normalized_tensor(padded, mean=self.cfg.normalize_mean, std=self.cfg.normalize_std)
        return tensor, valid_w

