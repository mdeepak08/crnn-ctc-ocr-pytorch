from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataloaders.collate import CollateConfig, collate_ocr_batch
from src.dataloaders.datasets import OCRCsvDataset, OCRCsvDatasetConfig, load_vocab
from src.dataloaders.transforms import OCRTransformConfig
from src.decoding.beam import BeamDecoderConfig, beam_decode
from src.decoding.greedy import GreedyDecoderConfig, greedy_decode
from src.losses.ctc import CTCLoss, CTCLossConfig
from src.metrics.cer import cer_corpus
from src.metrics.word_acc import word_acc_corpus
from src.models.crnn import CRNN, CRNNConfig
from src.utils.checkpoints import Checkpoint, save_checkpoint
from src.utils.device import DeviceConfig, get_device, supports_amp
from src.utils.logger import TBLogger, print_once
from src.utils.seed import SeedConfig, seed_everything


REPO_ROOT = Path(__file__).resolve().parent


def _seed_worker(worker_id: int) -> None:
    # Ensure numpy/random are deterministically seeded in each DataLoader worker.
    # PyTorch sets a unique (but deterministic) base seed per worker; we mirror it.
    worker_seed = torch.initial_seed() % (2**32)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception:
        pass
    try:
        import random

        random.seed(worker_seed)
    except Exception:
        pass


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


@torch.no_grad()
def run_eval(
    model: CRNN,
    loader: DataLoader,
    idx2char: list[str],
    device: torch.device,
    decoder: str,
    blank_idx: int,
    beam_width: int,
) -> dict[str, float]:
    model.eval()
    preds_all: list[str] = []
    gts_all: list[str] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["images"].to(device)
        log_probs = model(images)  # [T,B,C]

        if decoder == "beam":
            pred = beam_decode(log_probs, idx2char, BeamDecoderConfig(blank_idx=blank_idx, beam_width=beam_width))
        else:
            pred = greedy_decode(log_probs, idx2char, GreedyDecoderConfig(blank_idx=blank_idx))

        preds_all.extend(pred)
        gts_all.extend(batch["texts"])

    cer = cer_corpus(preds_all, gts_all).cer
    acc = word_acc_corpus(preds_all, gts_all).acc
    return {"cer": cer, "word_acc": acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (relative to repo root ok).")
    ap.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    seed_everything(SeedConfig(**cfg["seed"]))
    device = get_device(DeviceConfig(**cfg["device"]))
    print_once(f"Device: {device}")

    # NOTE: As of recent PyTorch builds, CTCLoss isn't implemented on MPS.
    # This env var allows missing MPS ops to fall back to CPU (slower, but works for sanity checks).
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print_once("MPS detected: enabling CPU fallback for missing ops (PYTORCH_ENABLE_MPS_FALLBACK=1).")

    vocab = load_vocab(REPO_ROOT / cfg["vocab"]["path"])
    idx2char = list(vocab["idx2char"])
    blank_idx = int(vocab.get("blank_idx", 0))
    num_classes = len(idx2char)

    # Datasets
    base_tcfg = dict(
        img_h=int(cfg["data"]["img_h"]),
        img_w=int(cfg["data"]["img_w"]),
    )
    aug_cfg = cfg.get("augment", {}) if isinstance(cfg, dict) else {}
    train_tcfg = OCRTransformConfig(
        **base_tcfg,
        augment_enabled=bool(aug_cfg.get("enabled", False)),
        aug_perspective_p=float(aug_cfg.get("perspective_p", 0.15)),
        aug_perspective_distortion=float(aug_cfg.get("perspective_distortion", 0.25)),
        aug_affine_p=float(aug_cfg.get("affine_p", 0.35)),
        aug_affine_degrees=float(aug_cfg.get("affine_degrees", 2.0)),
        aug_affine_translate=float(aug_cfg.get("affine_translate", 0.02)),
        aug_affine_scale_min=float(aug_cfg.get("affine_scale_min", 0.9)),
        aug_affine_scale_max=float(aug_cfg.get("affine_scale_max", 1.1)),
        aug_photometric_p=float(aug_cfg.get("photometric_p", 0.35)),
        aug_brightness=float(aug_cfg.get("brightness", 0.25)),
        aug_contrast=float(aug_cfg.get("contrast", 0.25)),
        aug_blur_p=float(aug_cfg.get("blur_p", 0.15)),
        aug_blur_radius_max=float(aug_cfg.get("blur_radius_max", 1.2)),
        aug_jpeg_p=float(aug_cfg.get("jpeg_p", 0.15)),
        aug_jpeg_quality_min=int(aug_cfg.get("jpeg_quality_min", 30)),
        aug_jpeg_quality_max=int(aug_cfg.get("jpeg_quality_max", 85)),
    )
    val_tcfg = OCRTransformConfig(**base_tcfg, augment_enabled=False)

    train_ds = OCRCsvDataset(
        OCRCsvDatasetConfig(
            csv_path=str(REPO_ROOT / cfg["datasets"]["train_csv"]),
            vocab_path=str(REPO_ROOT / cfg["vocab"]["path"]),
            max_len=int(cfg["data"]["max_len"]),
            lowercase=bool(cfg["data"]["lowercase"]),
            strict_vocab=bool(cfg["data"]["strict_vocab"]),
            image_base_dir=str(REPO_ROOT),
            transform=train_tcfg,
        )
    )
    val_ds = OCRCsvDataset(
        OCRCsvDatasetConfig(
            csv_path=str(REPO_ROOT / cfg["datasets"]["val_csv"]),
            vocab_path=str(REPO_ROOT / cfg["vocab"]["path"]),
            max_len=int(cfg["data"]["max_len"]),
            lowercase=bool(cfg["data"]["lowercase"]),
            strict_vocab=bool(cfg["data"]["strict_vocab"]),
            image_base_dir=str(REPO_ROOT),
            transform=val_tcfg,
        )
    )

    # Model
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

    # Loss/opt
    criterion = CTCLoss(CTCLossConfig(blank_idx=blank_idx, zero_infinity=True))
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    steps_per_epoch = math.ceil(len(train_ds) / int(cfg["train"]["batch_size"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["train"]["epochs"]) * steps_per_epoch))

    collate_cfg = CollateConfig(cnn_downsample_factor=model.cnn_downsample_factor_w)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(int(cfg["seed"]["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_ocr_batch, cfg=collate_cfg),
        worker_init_fn=_seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_ocr_batch, cfg=collate_cfg),
        worker_init_fn=_seed_worker,
        generator=dl_generator,
    )

    ckpt_dir = (REPO_ROOT / str(cfg["train"]["ckpt_dir"])).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = TBLogger(str(REPO_ROOT / str(cfg["train"]["log_dir"])))
    logger.add_text("config", "```json\n" + json.dumps(cfg, indent=2) + "\n```", step=0)

    start_epoch = 0
    global_step = 0
    best_cer = float("inf")
    resumed_best_cer: float | None = None

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        global_step = int(state.get("step", 0))
        best_cer = float(state.get("best_metric", best_cer))
        resumed_best_cer = best_cer
        print_once(f"Resumed from {args.resume} @ epoch={start_epoch} step={global_step} best_cer={best_cer:.4f}")

    use_amp = bool(cfg["train"]["amp"]) and supports_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    patience = int(cfg["train"]["early_stop_patience"])
    patience_left = patience

    # If we're fine-tuning on a new dataset, we want early stopping + best ckpt
    # selection to be based on the *new* validation set, not the previous run.
    if bool(cfg["train"].get("reset_best_on_resume", False)) and args.resume:
        best_cer = float("inf")
        patience_left = patience
        if resumed_best_cer is not None:
            print_once(
                f"Resetting best CER on resume (was {resumed_best_cer:.4f}) so this run can define a new best on its val set."
            )

    decoder_for_val = str(cfg["train"]["decoder_for_val"])
    beam_width = int(cfg["train"].get("beam_width", 5))

    max_epochs = int(cfg["train"]["epochs"])
    if start_epoch >= max_epochs:
        print_once(
            "Nothing to train: resumed start_epoch "
            f"({start_epoch}) >= config train.epochs ({max_epochs}). "
            "Either increase train.epochs in your config or run without --resume."
        )
        logger.close()
        return

    for epoch in range(start_epoch, max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        running_loss = 0.0

        for batch in pbar:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    log_probs = model(images)
                    T = log_probs.size(0)
                    input_lengths = torch.clamp(input_lengths, max=T)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                if float(cfg["train"]["grad_clip"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(images)
                T = log_probs.size(0)
                input_lengths = torch.clamp(input_lengths, max=T)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                if float(cfg["train"]["grad_clip"]) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                optimizer.step()

            scheduler.step()

            global_step += 1
            running_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

            if global_step % 50 == 0:
                logger.add_scalar("train/loss", float(loss.item()), global_step)
                logger.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        epoch_loss = running_loss / max(1, len(train_loader))
        logger.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # Validate
        metrics = run_eval(
            model=model,
            loader=val_loader,
            idx2char=idx2char,
            device=device,
            decoder=decoder_for_val,
            blank_idx=blank_idx,
            beam_width=beam_width,
        )
        logger.add_scalars("val", metrics, epoch)
        logger.flush()

        cer = float(metrics["cer"])
        print_once(f"epoch={epoch} loss={epoch_loss:.4f} val_cer={cer:.4f} val_word_acc={metrics['word_acc']:.4f}")

        # Save last
        save_checkpoint(
            ckpt_dir / "last.pt",
            Checkpoint(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                step=global_step,
                best_metric=best_cer,
                config=cfg,
                vocab=vocab,
            ),
        )

        # Best by CER
        if cer < best_cer:
            best_cer = cer
            patience_left = patience
            save_checkpoint(
                ckpt_dir / "best_by_cer.pt",
                Checkpoint(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    step=global_step,
                    best_metric=best_cer,
                    config=cfg,
                    vocab=vocab,
                ),
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print_once("Early stopping triggered.")
                break

    logger.close()


if __name__ == "__main__":
    main()

