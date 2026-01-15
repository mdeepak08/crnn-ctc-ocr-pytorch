from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ALLOWED = "0123456789abcdefghijklmnopqrstuvwxyz"


def write_csv(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(rows)


def make_vocab(allowed: str, out_path: Path) -> None:
    allowed = list(dict.fromkeys(list(allowed)))
    idx2char = [""] + allowed
    char2idx = {ch: i for i, ch in enumerate(idx2char) if i > 0}
    vocab = {"idx2char": idx2char, "char2idx": char2idx, "blank_idx": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="samples/toy", help="Where to write images and CSVs")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allowed", type=str, default=DEFAULT_ALLOWED)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--img_h", type=int, default=32)
    ap.add_argument("--img_w", type=int, default=128)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir).resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    rows: list[tuple[str, str]] = []
    for i in range(int(args.num_samples)):
        L = rng.randint(int(args.min_len), int(args.max_len))
        text = "".join(rng.choice(args.allowed) for _ in range(L))

        img = Image.new("L", (int(args.img_w), int(args.img_h)), color=255)
        draw = ImageDraw.Draw(img)
        # simple left padding + vertical centering
        draw.text((5, 6), text, font=font, fill=0)

        path = img_dir / f"{i:05d}.png"
        img.save(path)
        rows.append((str(path), text))

    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * float(args.val_ratio))))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_csv(train_rows, out_dir / "train.csv")
    write_csv(val_rows, out_dir / "val.csv")
    make_vocab(args.allowed, out_dir / "vocab.json")

    print(f"Wrote toy dataset to: {out_dir}")
    print(f"train={len(train_rows)} val={len(val_rows)} vocab={out_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()

