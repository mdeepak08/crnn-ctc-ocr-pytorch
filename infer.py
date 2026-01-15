from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.dataloaders.transforms import OCRPreprocess, OCRTransformConfig
from src.decoding.beam import BeamDecoderConfig, beam_decode
from src.decoding.greedy import GreedyDecoderConfig, greedy_decode
from src.models.crnn import CRNN, CRNNConfig
from src.utils.device import DeviceConfig, get_device


REPO_ROOT = Path(__file__).resolve().parent


def load_vocab(path: str | Path) -> dict:
    vocab = json.loads(Path(path).read_text())
    if vocab.get("blank_idx", 0) != 0:
        raise ValueError("Expected blank_idx=0")
    return vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--vocab", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--decoder", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    args = ap.parse_args()

    device = get_device(DeviceConfig(prefer=args.device))
    vocab = load_vocab(args.vocab)
    idx2char = list(vocab["idx2char"])
    blank_idx = int(vocab.get("blank_idx", 0))
    num_classes = len(idx2char)

    # Model config is stored in checkpoint when trained with this repo
    state = torch.load(args.ckpt, map_location="cpu")
    cfg = state.get("config", {})
    model_cfg = cfg.get("model", {})
    mcfg = CRNNConfig(
        img_h=int(model_cfg.get("img_h", args.img_h)),
        num_channels=int(model_cfg.get("num_channels", 1)),
        num_classes=num_classes,
        cnn_out_channels=int(model_cfg.get("cnn_out_channels", 256)),
        rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
        rnn_layers=int(model_cfg.get("rnn_layers", 2)),
        rnn_type=str(model_cfg.get("rnn_type", "lstm")),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    model = CRNN(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    tfm = OCRPreprocess(OCRTransformConfig(img_h=args.img_h, img_w=args.img_w))
    img = Image.open(args.image).convert("RGB")
    x, _ = tfm(img)
    x = x.unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        log_probs = model(x)  # [T,1,C]

    if args.decoder == "beam":
        pred = beam_decode(log_probs, idx2char, BeamDecoderConfig(blank_idx=blank_idx, beam_width=args.beam_width))[0]
    else:
        pred = greedy_decode(log_probs, idx2char, GreedyDecoderConfig(blank_idx=blank_idx))[0]

    print(pred)


if __name__ == "__main__":
    main()

