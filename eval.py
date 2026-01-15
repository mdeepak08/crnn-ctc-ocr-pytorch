from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders.collate import CollateConfig, collate_ocr_batch
from src.dataloaders.datasets import OCRCsvDataset, OCRCsvDatasetConfig, load_vocab
from src.dataloaders.transforms import OCRTransformConfig
from src.decoding.beam import BeamDecoderConfig, beam_decode
from src.decoding.greedy import GreedyDecoderConfig, greedy_decode
from src.metrics.cer import cer_corpus
from src.metrics.word_acc import word_acc_corpus
from src.models.crnn import CRNN, CRNNConfig
from src.utils.device import DeviceConfig, get_device


REPO_ROOT = Path(__file__).resolve().parent


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict[str, Any]:
    p = (REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    cfg = yaml.safe_load(p.read_text())
    if isinstance(cfg, dict) and "base" in cfg:
        base = load_config(cfg["base"])
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base"})
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["", "iiit5k", "svt", "mjsynth_val"],
        help="Convenience selector (uses standard processed CSV names).",
    )
    ap.add_argument("--dataset_csv", type=str, default="", help="Explicit CSV with columns path,text (overrides --dataset)")
    ap.add_argument("--decoder", type=str, default="", choices=["", "greedy", "beam"])
    ap.add_argument("--out_dir", type=str, default="eval_outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(DeviceConfig(**cfg["device"]))

    vocab = load_vocab(REPO_ROOT / cfg["vocab"]["path"])
    idx2char = list(vocab["idx2char"])
    blank_idx = int(vocab.get("blank_idx", 0))
    num_classes = len(idx2char)

    tcfg = OCRTransformConfig(
        img_h=int(cfg["data"]["img_h"]),
        img_w=int(cfg["data"]["img_w"]),
    )
    if args.dataset_csv:
        dataset_csv = args.dataset_csv
    else:
        if args.dataset == "iiit5k":
            dataset_csv = "data/processed/iiit5k_test.csv"
        elif args.dataset == "svt":
            dataset_csv = "data/processed/svt.csv"
        elif args.dataset == "mjsynth_val":
            dataset_csv = str(cfg.get("datasets", {}).get("val_csv", "data/processed/mjsynth_val.csv"))
        else:
            raise ValueError("Provide --dataset_csv or set --dataset.")

    ds = OCRCsvDataset(
        OCRCsvDatasetConfig(
            csv_path=str((REPO_ROOT / dataset_csv).resolve() if not Path(dataset_csv).is_absolute() else dataset_csv),
            vocab_path=str(REPO_ROOT / cfg["vocab"]["path"]),
            max_len=int(cfg["data"]["max_len"]),
            lowercase=bool(cfg["data"]["lowercase"]),
            strict_vocab=bool(cfg["data"]["strict_vocab"]),
            image_base_dir=str(REPO_ROOT),
            transform=tcfg,
        )
    )

    mcfg = CRNNConfig(
        img_h=int(cfg["model"]["img_h"]),
        num_channels=int(cfg["model"]["num_channels"]),
        num_classes=num_classes,
        cnn_out_channels=int(cfg["model"]["cnn_out_channels"]),
        rnn_hidden=int(cfg["model"]["rnn_hidden"]),
        rnn_layers=int(cfg["model"]["rnn_layers"]),
        rnn_type=str(cfg["model"]["rnn_type"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = CRNN(mcfg).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    collate_cfg = CollateConfig(cnn_downsample_factor=model.cnn_downsample_factor_w)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["eval"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_ocr_batch, cfg=collate_cfg),
    )

    decoder = args.decoder or str(cfg["eval"]["decoder"])
    beam_width = int(cfg["eval"].get("beam_width", 5))

    preds_all: list[str] = []
    gts_all: list[str] = []
    paths_all: list[str] = []

    for batch in tqdm(loader, desc="eval"):
        images = batch["images"].to(device)
        with torch.no_grad():
            log_probs = model(images)  # [T,B,C]

        if decoder == "beam":
            preds = beam_decode(log_probs, idx2char, BeamDecoderConfig(blank_idx=blank_idx, beam_width=beam_width))
        else:
            preds = greedy_decode(log_probs, idx2char, GreedyDecoderConfig(blank_idx=blank_idx))

        preds_all.extend(preds)
        gts_all.extend(batch["texts"])
        paths_all.extend(batch["paths"])

    cer = cer_corpus(preds_all, gts_all).cer
    acc = word_acc_corpus(preds_all, gts_all).acc

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "cer": cer,
        "word_acc": acc,
        "decoder": decoder,
        "num_samples": len(ds),
        "dataset_csv": dataset_csv,
        "ckpt": args.ckpt,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "gt", "pred", "correct"])
        for p, gt, pr in zip(paths_all, gts_all, preds_all):
            w.writerow([p, gt, pr, int(gt == pr)])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

