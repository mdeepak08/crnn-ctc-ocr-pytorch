from __future__ import annotations

from dataclasses import dataclass
import io
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from torchvision import transforms as T


@dataclass(frozen=True)
class OCRTransformConfig:
    img_h: int = 32
    img_w: int = 128
    pad_value: int = 0
    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    # Train-time augmentation (kept off by default; enable via config for experiments)
    augment_enabled: bool = False
    aug_perspective_p: float = 0.15
    aug_perspective_distortion: float = 0.25
    aug_affine_p: float = 0.35
    aug_affine_degrees: float = 2.0
    aug_affine_translate: float = 0.02  # fraction of image size
    aug_affine_scale_min: float = 0.9
    aug_affine_scale_max: float = 1.1
    aug_photometric_p: float = 0.35
    aug_brightness: float = 0.25  # factor range [1-b, 1+b]
    aug_contrast: float = 0.25
    aug_blur_p: float = 0.15
    aug_blur_radius_max: float = 1.2
    aug_jpeg_p: float = 0.15
    aug_jpeg_quality_min: int = 30
    aug_jpeg_quality_max: int = 85


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

        # Torchvision augmentors (use torch RNG → controlled by torch.manual_seed + worker seeds)
        self._perspective = T.RandomPerspective(distortion_scale=float(cfg.aug_perspective_distortion), p=1.0)
        self._affine = T.RandomAffine(
            degrees=float(cfg.aug_affine_degrees),
            translate=(float(cfg.aug_affine_translate), float(cfg.aug_affine_translate)),
            scale=(float(cfg.aug_affine_scale_min), float(cfg.aug_affine_scale_max)),
            shear=None,
            fill=int(cfg.pad_value),
        )

    def _maybe_geometric_aug(self, img_l: Image.Image) -> Image.Image:
        if not self.cfg.augment_enabled:
            return img_l

        # Perspective + affine BEFORE resize/pad (helps with real-world viewpoint + mild rotations)
        if torch.rand(1).item() < float(self.cfg.aug_perspective_p):
            # torchvision's RandomPerspective can rarely sample a degenerate transform
            # that makes lstsq fail; skip augmentation in that case instead of crashing.
            try:
                img_l = self._perspective(img_l)
            except Exception:
                pass
        if torch.rand(1).item() < float(self.cfg.aug_affine_p):
            try:
                img_l = self._affine(img_l)
            except Exception:
                pass
        return img_l

    def _maybe_photometric_aug(self, img_l: Image.Image) -> Image.Image:
        if not self.cfg.augment_enabled:
            return img_l

        # Photometric + corruption AFTER pad (keeps output shape fixed)
        if random.random() < float(self.cfg.aug_photometric_p):
            b = float(self.cfg.aug_brightness)
            c = float(self.cfg.aug_contrast)
            if b > 0:
                img_l = ImageEnhance.Brightness(img_l).enhance(random.uniform(1.0 - b, 1.0 + b))
            if c > 0:
                img_l = ImageEnhance.Contrast(img_l).enhance(random.uniform(1.0 - c, 1.0 + c))

        if random.random() < float(self.cfg.aug_blur_p):
            rmax = float(self.cfg.aug_blur_radius_max)
            if rmax > 0:
                img_l = img_l.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, rmax)))

        if random.random() < float(self.cfg.aug_jpeg_p):
            qmin = int(self.cfg.aug_jpeg_quality_min)
            qmax = int(self.cfg.aug_jpeg_quality_max)
            q = int(random.randint(min(qmin, qmax), max(qmin, qmax)))
            buf = io.BytesIO()
            img_l.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img_l = Image.open(buf).convert("L")

        return img_l

    def __call__(self, img: Image.Image) -> tuple[torch.Tensor, int]:
        img = _to_grayscale_pil(img)
        img = self._maybe_geometric_aug(img)

        padded, valid_w = resize_keep_ratio_pad(
            img, img_h=self.cfg.img_h, img_w=self.cfg.img_w, pad_value=self.cfg.pad_value
        )
        padded = self._maybe_photometric_aug(padded)
        tensor = pil_to_normalized_tensor(padded, mean=self.cfg.normalize_mean, std=self.cfg.normalize_std)
        return tensor, valid_w

