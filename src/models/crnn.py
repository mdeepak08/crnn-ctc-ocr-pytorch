from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CRNNConfig:
    img_h: int = 32
    num_channels: int = 1
    num_classes: int = 38  # includes blank at 0
    cnn_out_channels: int = 256
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_type: str = "lstm"  # lstm|gru
    dropout: float = 0.1
    # Option A (rectifier): Spatial Transformer Network (affine) before the CNN.
    # NOTE: TPS is stronger but more complex; affine STN is a solid first rectifier.
    stn_enabled: bool = False
    stn_localization_channels: int = 32


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

        if cfg.img_h != 32:
            # The CNN pooling stack below assumes H=32 for exact height squeezing.
            # You can change img_h but you must adjust the pooling.
            raise ValueError("CRNN currently expects img_h=32 for the default CNN stack.")

        self.stn: nn.Module | None = None
        if bool(cfg.stn_enabled):
            self.stn = _STNRectifier(cfg.num_channels, int(cfg.stn_localization_channels))

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,1,32,W]
        if self.stn is not None:
            images = self.stn(images)  # [B,1,32,W] rectified
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

