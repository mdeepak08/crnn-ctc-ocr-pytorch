from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from urllib.parse import unquote


DEFAULT_ALLOWED = "0123456789abcdefghijklmnopqrstuvwxyz"


def parse_label_from_path(p: Path) -> str:
    """
    MJSynth label is embedded in filename, commonly like:
      <id>_<label>_<...>.jpg
    """
    name = p.name
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename (no label token): {name}")
    label = parts[1]
    label = unquote(label)
    return label


def is_valid_label(s: str, *, allowed: set[str], max_len: int, lowercase: bool) -> bool:
    if lowercase:
        s = s.lower()
    if len(s) == 0 or len(s) > max_len:
        return False
    return all(ch in allowed for ch in s)


def iter_image_paths(raw_root: Path) -> list[Path]:
    """
    Prefer annotation files if present (much faster than scanning millions of images).
    Falls back to recursive glob.
    """
    ann = raw_root / "annotation.txt"
    if ann.exists():
        paths: list[Path] = []
        with ann.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # annotation.txt usually stores relative paths
                paths.append((raw_root / line).resolve())
        return paths

    # Fallback: scan
    return [p.resolve() for p in raw_root.rglob("*.jpg")]


def write_csv(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="Root folder containing MJSynth images")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for processed CSVs")
    ap.add_argument("--allowed", type=str, default=DEFAULT_ALLOWED)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--lowercase", action="store_true", help="Lowercase labels (recommended).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only keep this many samples (for sanity runs).")
    ap.add_argument(
        "--relative_to",
        type=str,
        default="",
        help="If set, store paths in CSV relative to this folder (recommended: repo root).",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(args.allowed)
    rng = random.Random(args.seed)

    img_paths = iter_image_paths(raw_root)
    rng.shuffle(img_paths)

    rows: list[tuple[str, str]] = []
    bad = 0
    for p in img_paths:
        try:
            label = parse_label_from_path(p)
        except Exception:
            bad += 1
            continue
        if args.lowercase:
            label = label.lower()
        if not is_valid_label(label, allowed=allowed, max_len=int(args.max_len), lowercase=False):
            bad += 1
            continue

        if args.relative_to:
            rel_root = Path(args.relative_to).resolve()
            try:
                rel = str(p.relative_to(rel_root))
            except Exception:
                rel = str(p)
        else:
            rel = str(p)

        rows.append((rel, label))
        if args.limit and len(rows) >= int(args.limit):
            break

    if len(rows) == 0:
        raise RuntimeError("No valid samples found. Check raw_root and allowed/max_len settings.")

    n_val = max(1, int(round(len(rows) * float(args.val_ratio))))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_csv(train_rows, out_dir / "mjsynth_train.csv")
    write_csv(val_rows, out_dir / "mjsynth_val.csv")

    print(f"Prepared MJSynth: train={len(train_rows)} val={len(val_rows)} bad/skipped={bad}")
    print(f"Wrote: {out_dir / 'mjsynth_train.csv'}")
    print(f"Wrote: {out_dir / 'mjsynth_val.csv'}")


if __name__ == "__main__":
    main()

