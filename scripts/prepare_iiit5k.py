from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scipy.io import loadmat


DEFAULT_ALLOWED = "0123456789abcdefghijklmnopqrstuvwxyz"


def _as_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    # scipy loadmat often gives numpy arrays of dtype '<U' or objects
    try:
        return str(x)
    except Exception:
        return ""


def _extract_records(mat_path: Path) -> list[tuple[str, str]]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Common keys: 'traindata', 'testdata'
    key = None
    for k in ["traindata", "testdata", "data"]:
        if k in mat:
            key = k
            break
    if key is None:
        # fallback: pick first key ending with 'data'
        for k in mat.keys():
            if k.lower().endswith("data"):
                key = k
                break
    if key is None:
        raise KeyError(f"Could not find data key in {mat_path}. Keys: {list(mat.keys())}")

    data = mat[key]
    records: list[tuple[str, str]] = []

    # data may be a list/ndarray of structs
    if not hasattr(data, "__len__"):
        data = [data]

    for item in data:
        # struct fields: ImgName, GroundTruth
        img_name = getattr(item, "ImgName", None)
        gt = getattr(item, "GroundTruth", None)
        if img_name is None or gt is None:
            continue
        records.append((_as_str(img_name).strip(), _as_str(gt).strip()))

    return records


def write_csv(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="IIIT-5K root folder (contains *.mat + images)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for processed CSVs")
    ap.add_argument("--allowed", type=str, default=DEFAULT_ALLOWED)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--lowercase", action="store_true")
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
    rel_root = Path(args.relative_to).resolve() if args.relative_to else None

    train_mat = raw_root / "traindata.mat"
    test_mat = raw_root / "testdata.mat"
    if not train_mat.exists() or not test_mat.exists():
        raise FileNotFoundError(
            f"Expected traindata.mat and testdata.mat in {raw_root}. Found: {train_mat.exists()} {test_mat.exists()}"
        )

    train_recs = _extract_records(train_mat)
    test_recs = _extract_records(test_mat)

    def norm_rows(recs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for img_name, text in recs:
            text = text.strip()
            if args.lowercase:
                text = text.lower()
            if len(text) == 0 or len(text) > int(args.max_len):
                continue
            if any(ch not in allowed for ch in text):
                continue

            img_path = (raw_root / img_name).resolve()
            if not img_path.exists():
                # some releases keep images under "train/" or "test/"
                if (raw_root / "train" / img_name).exists():
                    img_path = (raw_root / "train" / img_name).resolve()
                elif (raw_root / "test" / img_name).exists():
                    img_path = (raw_root / "test" / img_name).resolve()

            if rel_root is not None:
                try:
                    rel = str(img_path.relative_to(rel_root))
                except Exception:
                    rel = str(img_path)
            else:
                rel = str(img_path)

            out.append((rel, text))
        return out

    train_rows = norm_rows(train_recs)
    test_rows = norm_rows(test_recs)

    write_csv(train_rows, out_dir / "iiit5k_train.csv")
    write_csv(test_rows, out_dir / "iiit5k_test.csv")

    print(f"Prepared IIIT-5K: train={len(train_rows)} test={len(test_rows)}")
    print(f"Wrote: {out_dir / 'iiit5k_train.csv'}")
    print(f"Wrote: {out_dir / 'iiit5k_test.csv'}")


if __name__ == "__main__":
    main()

