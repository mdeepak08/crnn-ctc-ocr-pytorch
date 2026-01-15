from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


DEFAULT_ALLOWED = "0123456789abcdefghijklmnopqrstuvwxyz"


def write_csv(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="SVT root folder (contains xml + images)")
    ap.add_argument(
        "--xml_path",
        type=str,
        default="",
        help="Path to SVT XML annotation (if empty, tries common filenames inside raw_root).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (will write crops + CSV)")
    ap.add_argument("--allowed", type=str, default=DEFAULT_ALLOWED)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--relative_to", type=str, default="", help="If set, store crop paths relative to this folder.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "svt_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    allowed = set(args.allowed)
    rel_root = Path(args.relative_to).resolve() if args.relative_to else None

    xml_candidates = [
        raw_root / "test.xml",
        raw_root / "svt1.xml",
        raw_root / "svt.xml",
    ]
    xml_path = Path(args.xml_path).resolve() if args.xml_path else None
    if xml_path is None or not xml_path.exists():
        xml_path = next((p for p in xml_candidates if p.exists()), None)
    if xml_path is None:
        raise FileNotFoundError("Could not find SVT XML. Pass --xml_path explicitly.")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    rows: list[tuple[str, str]] = []
    n_skipped = 0

    # SVT XML commonly has structure:
    # <image file="img/xxx.jpg"> <taggedRect x= y= width= height=> <tag>WORD</tag> ...
    for img_el in root.iter("image"):
        file_attr = img_el.attrib.get("file", "") or img_el.attrib.get("name", "")
        if not file_attr:
            continue

        full_img_path = (raw_root / file_attr).resolve()
        if not full_img_path.exists():
            # some distributions put images under "img/"
            if (raw_root / "img" / Path(file_attr).name).exists():
                full_img_path = (raw_root / "img" / Path(file_attr).name).resolve()
            else:
                n_skipped += 1
                continue

        try:
            full = Image.open(full_img_path).convert("RGB")
        except Exception:
            n_skipped += 1
            continue

        for rect in img_el.iter("taggedRect"):
            tag_el = rect.find("tag")
            if tag_el is None or tag_el.text is None:
                continue
            text = tag_el.text.strip()
            if args.lowercase:
                text = text.lower()

            if len(text) == 0 or len(text) > int(args.max_len) or any(ch not in allowed for ch in text):
                n_skipped += 1
                continue

            try:
                x = int(float(rect.attrib.get("x", "0")))
                y = int(float(rect.attrib.get("y", "0")))
                w = int(float(rect.attrib.get("width", "0")))
                h = int(float(rect.attrib.get("height", "0")))
            except Exception:
                n_skipped += 1
                continue

            if w <= 1 or h <= 1:
                n_skipped += 1
                continue

            crop = full.crop((x, y, x + w, y + h))
            crop_name = f"{full_img_path.stem}_{x}_{y}_{w}_{h}.jpg"
            crop_path = (crops_dir / crop_name).resolve()
            try:
                crop.save(crop_path, quality=95)
            except Exception:
                n_skipped += 1
                continue

            if rel_root is not None:
                try:
                    rel = str(crop_path.relative_to(rel_root))
                except Exception:
                    rel = str(crop_path)
            else:
                rel = str(crop_path)

            rows.append((rel, text))

    write_csv(rows, out_dir / "svt.csv")
    print(f"Prepared SVT crops: n={len(rows)} skipped={n_skipped}")
    print(f"Wrote crops to: {crops_dir}")
    print(f"Wrote CSV to: {out_dir / 'svt.csv'}")


if __name__ == "__main__":
    main()

