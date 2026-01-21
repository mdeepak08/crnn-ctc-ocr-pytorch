from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tps import TPSConfig, TPSRectifier


@dataclass(frozen=True)
class CRNNConfig:
    img_h: int = 32
    img_w: int = 128
    num_channels: int = 1
    num_classes: int = 38  # includes blank at 0
    cnn_out_channels: int = 256
    # Visual backbone (CNN). "simple" is the original CRNN CNN; "resnet" is a stronger ResNet-style backbone.
    cnn_backbone: str = "simple"  # simple|resnet
    resnet_base_channels: int = 64
    # ResNet blocks per stage (like ResNet-18: [2,2,2]).
    # NOTE: This is a small OCR-tailored backbone; we keep width downsample factor at 4.
    resnet_layers: tuple[int, int, int] = (2, 2, 2)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_type: str = "lstm"  # lstm|gru
    dropout: float = 0.1
    # Option A (rectifier): Spatial Transformer Network (affine) before the CNN.
    # NOTE: TPS is stronger but more complex; affine STN is a solid first rectifier.
    stn_enabled: bool = False
    stn_localization_channels: int = 32
    # Option A (rectifier): TPS (Thin Plate Spline) rectifier before the CNN.
    # If enabled, TPS is used (and affine STN is ignored).
    tps_enabled: bool = False
    tps_num_fiducial: int = 20
    tps_margin_x: float = 0.05
    tps_margin_y: float = 0.05
    tps_localization_channels: int = 32


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out


class _ResNetOCRBackbone(nn.Module):
    """
    A small ResNet-style backbone tuned for CRNN:
    - Keeps the CRNN time axis over width.
    - Width downsample factor is fixed at 4 (via two 2x2 pools).
    - Collapses height to 1 via adaptive pooling at the end.
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int, layers: tuple[int, int, int]):
        super().__init__()
        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 4)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, layers[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2

        self.stage2 = self._make_stage(c1, c2, layers[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4

        self.stage3 = self._make_stage(c2, c3, layers[2])

        # Project to the CRNN feature dimension expected by the RNN.
        self.proj = nn.Sequential(
            nn.Conv2d(c3, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
        blocks: list[nn.Module] = []
        blocks.append(_ResBlock(in_ch, out_ch))
        for _ in range(max(0, int(num_blocks) - 1)):
            blocks.append(_ResBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.proj(x)
        # Collapse height to 1 while keeping width (time axis) unchanged.
        if x.size(2) != 1:
            x = F.adaptive_avg_pool2d(x, output_size=(1, x.size(3)))
        return x  # [B,C,1,W/4]


class _STNRectifier(nn.Module):
    """
    A lightweight affine Spatial Transformer Network (STN) that learns to warp the
    input word image to be more fronto-parallel before feature extraction.

    Input:  [B,1,32,W]
    Output: [B,1,32,W] (same shape)
    """

    def __init__(self, in_channels: int, loc_channels: int):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, loc_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16, W -> W/2
            nn.Conv2d(loc_channels, loc_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8, W/2 -> W/4
        )

        # Use adaptive pooling so we don't care about exact W.
        self.loc_pool = nn.AdaptiveAvgPool2d((4, 8))  # [B,C,4,8]
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_channels * 4 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),  # affine 2x3
        )

        # Initialize to identity transform.
        nn.init.zeros_(self.fc_loc[-1].weight)
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        xs = self.localization(x)
        xs = self.loc_pool(xs)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        x_warp = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return x_warp


class CRNN(nn.Module):
    """
    CNN -> (time steps over width) -> BiRNN -> Linear -> log_softmax for CTC

    Input:  images [B, C=1, H=32, W]
    Output: log_probs [T, B, num_classes]
    """

    cnn_downsample_factor_w: int = 4

    def __init__(self, cfg: CRNNConfig):
        super().__init__()
        self.cfg = cfg

        self.stn: nn.Module | None = None
        self.tps: nn.Module | None = None
        if bool(cfg.tps_enabled):
            self.tps = TPSRectifier(
                in_channels=cfg.num_channels,
                img_h=cfg.img_h,
                img_w=cfg.img_w,
                cfg=TPSConfig(
                    num_fiducial=int(cfg.tps_num_fiducial),
                    margin_x=float(cfg.tps_margin_x),
                    margin_y=float(cfg.tps_margin_y),
                    loc_channels=int(cfg.tps_localization_channels),
                ),
            )
        elif bool(cfg.stn_enabled):
            self.stn = _STNRectifier(cfg.num_channels, int(cfg.stn_localization_channels))

        backbone = str(getattr(cfg, "cnn_backbone", "simple")).lower()
        if backbone == "simple":
            if cfg.img_h != 32:
                # The original pooling stack below assumes H=32 for exact height squeezing.
                raise ValueError("CRNN expects img_h=32 when model.cnn_backbone=simple.")
            self.cnn = nn.Sequential(
                # [B,1,32,W]
                nn.Conv2d(cfg.num_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # [B,64,16,W/2]

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # [B,128,8,W/4]

                nn.Conv2d(128, cfg.cnn_out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # Reduce height only: 8 -> 2
                nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # [B,C,2,W/4]
                # Reduce height only: 2 -> 1
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [B,C,1,W/4]
            )
        elif backbone == "resnet":
            self.cnn = _ResNetOCRBackbone(
                in_channels=int(cfg.num_channels),
                out_channels=int(cfg.cnn_out_channels),
                base_channels=int(getattr(cfg, "resnet_base_channels", 64)),
                layers=tuple(getattr(cfg, "resnet_layers", (2, 2, 2))),
            )
        else:
            raise ValueError(f"Unknown cnn_backbone={cfg.cnn_backbone}. Use simple|resnet.")

        rnn_in = cfg.cnn_out_channels
        rnn_out = cfg.rnn_hidden
        bidirectional = True

        if cfg.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_in,
                hidden_size=rnn_out,
                num_layers=cfg.rnn_layers,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif cfg.rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_in,
                hidden_size=rnn_out,
                num_layers=cfg.rnn_layers,
                dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown rnn_type={cfg.rnn_type}. Use lstm|gru.")

        self.proj = nn.Linear(rnn_out * 2, cfg.num_classes)

    def rectifier_parameters(self):
        if self.tps is not None:
            return self.tps.parameters()
        if self.stn is not None:
            return self.stn.parameters()
        return []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,1,32,W]
        if self.tps is not None:
            images = self.tps(images)  # [B,1,32,W]
        elif self.stn is not None:
            images = self.stn(images)  # [B,1,32,W]
        feats = self.cnn(images)  # [B,C,1,W']
        feats = feats.squeeze(2)  # [B,C,W']
        feats = feats.permute(2, 0, 1).contiguous()  # [T=W',B,C]

        seq, _ = self.rnn(feats)  # [T,B,2*hidden]
        logits = self.proj(seq)  # [T,B,num_classes]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(images)

