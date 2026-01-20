from __future__ import annotations

from dataclasses import dataclass
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TPSConfig:
    """
    Thin Plate Spline rectifier config.

    We implement the classic STR TPS rectifier:
    - predict K fiducial points
    - compute TPS transform to map predicted -> canonical grid
    - sample input with grid_sample
    """

    num_fiducial: int = 20  # typically 20 (10 top + 10 bottom)
    margin_x: float = 0.05  # canonical point x margin in normalized coords
    margin_y: float = 0.05  # canonical point y margin in normalized coords
    loc_channels: int = 32


def _build_canonical_fiducials(k: int, margin_x: float, margin_y: float, device: torch.device) -> torch.Tensor:
    """
    Return canonical fiducial points in normalized coords [-1,1], shape [K,2].
    Arranged as (top row left->right) then (bottom row left->right).
    """
    if k % 2 != 0:
        raise ValueError("num_fiducial must be even (top+bottom rows).")
    k2 = k // 2

    xs = torch.linspace(-1.0 + margin_x * 2.0, 1.0 - margin_x * 2.0, steps=k2, device=device)
    y_top = torch.full((k2,), -1.0 + margin_y * 2.0, device=device)
    y_bot = torch.full((k2,), 1.0 - margin_y * 2.0, device=device)

    top = torch.stack([xs, y_top], dim=1)  # [k2,2]
    bot = torch.stack([xs, y_bot], dim=1)  # [k2,2]
    return torch.cat([top, bot], dim=0)  # [K,2]


class TPSGridGen(nn.Module):
    """
    Generates a sampling grid for TPS transformation.
    """

    def __init__(self, out_h: int, out_w: int, cfg: TPSConfig):
        super().__init__()
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.cfg = cfg

        # Precompute target grid points (canonical grid) in normalized coords [-1,1].
        # Shape: [H*W, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.out_h),
            torch.linspace(-1.0, 1.0, self.out_w),
            indexing="ij",
        )
        self.register_buffer("_grid", torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1), persistent=False)

    def forward(self, source_fiducials: torch.Tensor) -> torch.Tensor:
        """
        source_fiducials: [B, K, 2] in normalized coords [-1,1]
        Returns grid: [B, H, W, 2] suitable for grid_sample.
        """
        B, K, _ = source_fiducials.shape
        device = source_fiducials.device
        # IMPORTANT: TPS uses linalg.solve; under AMP autocast this can mix dtypes (fp16/fp32)
        # and crash. Force TPS grid computation to fp32 for stability.
        dtype = torch.float32

        # Canonical fiducials (target) are fixed.
        target = _build_canonical_fiducials(K, self.cfg.margin_x, self.cfg.margin_y, device).to(dtype)  # [K,2]

        # Build TPS system matrix for target fiducials.
        # We solve for mapping f(.) such that f(source) ~= target for each batch.
        # Classic TPS uses:
        #  [ K  P ] [ w ] = [ target ]
        #  [ P^T 0] [ a ]   [  0     ]
        #
        # where K_ij = U(||p_i - p_j||), P=[1, x, y], w are non-linear weights, a affine coeffs.

        def U(r2: torch.Tensor) -> torch.Tensor:
            # U(r) = r^2 * log(r^2), with U(0)=0
            eps = 1e-6
            return r2 * torch.log(r2 + eps)

        # Compute pairwise distances between source fiducials (for each batch).
        # source: [B,K,2]
        src = source_fiducials.to(dtype)
        diff = src[:, :, None, :] - src[:, None, :, :]  # [B,K,K,2]
        r2 = (diff * diff).sum(dim=-1)  # [B,K,K]
        Kmat = U(r2)  # [B,K,K]

        ones = torch.ones((B, K, 1), device=device, dtype=dtype)
        P = torch.cat([ones, src], dim=2)  # [B,K,3]

        # Assemble L: [B, K+3, K+3]
        zeros_3_3 = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        upper = torch.cat([Kmat, P], dim=2)  # [B,K,K+3]
        lower = torch.cat([P.transpose(1, 2), zeros_3_3], dim=2)  # [B,3,K+3]
        L = torch.cat([upper, lower], dim=1)  # [B,K+3,K+3]

        # Y: [B, K+3, 2]
        Y = torch.zeros((B, K + 3, 2), device=device, dtype=dtype)
        Y[:, :K, :] = target.unsqueeze(0).expand(B, -1, -1)  # map source -> canonical

        # Solve L * coeff = Y
        # coeff: [B, K+3, 2] where first K rows are w, last 3 are affine a.
        coeff = torch.linalg.solve(L, Y)  # [B,K+3,2]
        w = coeff[:, :K, :]  # [B,K,2]
        a = coeff[:, K:, :]  # [B,3,2]

        # Now compute mapping for each grid point g: f(g) = a0 + a1*x + a2*y + sum_i w_i * U(||g - p_i||)
        grid = self._grid.to(device=device, dtype=dtype)  # [HW,2]
        HW = grid.size(0)
        grid_exp = grid.unsqueeze(0).expand(B, -1, -1)  # [B,HW,2]

        # Affine part
        ones_hw = torch.ones((B, HW, 1), device=device, dtype=dtype)
        P_g = torch.cat([ones_hw, grid_exp], dim=2)  # [B,HW,3]
        affine = torch.bmm(P_g, a)  # [B,HW,2]

        # Non-linear part
        diff_g = grid_exp[:, :, None, :] - src[:, None, :, :]  # [B,HW,K,2]
        r2_g = (diff_g * diff_g).sum(dim=-1)  # [B,HW,K]
        U_g = U(r2_g)  # [B,HW,K]
        nonlin = torch.bmm(U_g, w)  # [B,HW,2]

        out = affine + nonlin  # [B,HW,2] in normalized coords
        return out.view(B, self.out_h, self.out_w, 2)


class TPSRectifier(nn.Module):
    """
    TPS rectifier module:
    - predicts source fiducials from the input image
    - generates TPS sampling grid
    - samples the image into a rectified output of the same size
    """

    def __init__(self, in_channels: int, img_h: int, img_w: int, cfg: TPSConfig):
        super().__init__()
        self.cfg = cfg
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        K = int(cfg.num_fiducial)

        # Localization network
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, cfg.loc_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(cfg.loc_channels, cfg.loc_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        self.loc_pool = nn.AdaptiveAvgPool2d((4, 8))
        self.fc = nn.Sequential(
            nn.Linear(cfg.loc_channels * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, K * 2),
        )

        # Initialize to canonical fiducials (identity rectification)
        with torch.no_grad():
            canonical = _build_canonical_fiducials(K, cfg.margin_x, cfg.margin_y, device=torch.device("cpu"))
            nn.init.zeros_(self.fc[-1].weight)
            self.fc[-1].bias.copy_(canonical.reshape(-1))

        self.grid_gen = TPSGridGen(out_h=self.img_h, out_w=self.img_w, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] where H/W match preprocess (e.g., 32/128)
        orig_dtype = x.dtype
        autocast_off = (
            torch.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
        )
        with autocast_off:
            x32 = x.float()
            xs = self.loc(x32)
            xs = self.loc_pool(xs).reshape(xs.size(0), -1)
            fid = self.fc(xs).view(xs.size(0), self.cfg.num_fiducial, 2)  # [B,K,2] fp32

            # Generate grid mapping from canonical grid -> source (so sampling picks pixels from x)
            grid = self.grid_gen(fid)  # [B,H,W,2] fp32
            x_warp = F.grid_sample(x32, grid, mode="bilinear", padding_mode="border", align_corners=False)

        return x_warp.to(orig_dtype)

