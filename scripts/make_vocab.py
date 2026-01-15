from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ALLOWED = "0123456789abcdefghijklmnopqrstuvwxyz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_paths", type=str, nargs="+", required=True, help="One or more CSVs with columns path,text")
    ap.add_argument("--out_path", type=str, required=True, help="Where to write vocab.json")
    ap.add_argument("--allowed", type=str, default=DEFAULT_ALLOWED, help="Allowed characters (order defines indices).")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before collecting characters.")
    ap.add_argument(
        "--use_seen_only",
        action="store_true",
        help="If set, keep only characters that appear in the provided CSVs (still ordered by --allowed).",
    )
    args = ap.parse_args()

    allowed = list(dict.fromkeys(list(args.allowed)))  # stable unique

    seen: set[str] = set()
    for csv_path in args.csv_paths:
        with Path(csv_path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"].strip()
                if args.lowercase:
                    text = text.lower()
                seen.update(text)

    if args.use_seen_only:
        allowed = [ch for ch in allowed if ch in seen]

    # CTC blank is at 0. Real characters start from 1.
    idx2char = [""] + allowed
    char2idx = {ch: i for i, ch in enumerate(idx2char) if i > 0}

    # Validate: report how many texts contain OOV chars (from the chosen vocab)
    oov = 0
    total = 0
    for csv_path in args.csv_paths:
        with Path(csv_path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                text = row["text"].strip()
                if args.lowercase:
                    text = text.lower()
                if any(c not in char2idx for c in text):
                    oov += 1

    vocab = {
        "idx2char": idx2char,
        "char2idx": char2idx,
        "blank_idx": 0,
    }

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    print(f"Wrote vocab to {out} (num_classes={len(idx2char)}). OOV rows vs vocab: {oov}/{total}")


if __name__ == "__main__":
    main()

