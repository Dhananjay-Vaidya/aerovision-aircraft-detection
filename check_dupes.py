#!/usr/bin/env python3
"""Make one image showing random removed-duplicate pairs side by side.

Usage:  python check_dupes.py            -> writes dupes_sample.jpg (20 pairs)
        python check_dupes.py --n 30 --seed 7
Left = image that was removed, right = image it was judged a duplicate of.
Each row also shows the two hash distances (lower = more similar).
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--report", default="cvs_dataset_dedupe_report.csv")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="dupes_sample.jpg")
args = ap.parse_args()

with open(args.report, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["type"] == "near"]
random.Random(args.seed).shuffle(rows)
rows = rows[: args.n]

T = 240  # thumbnail size
sheet = Image.new("RGB", (2 * T + 20, len(rows) * (T + 24)), "white")
draw = ImageDraw.Draw(sheet)


def thumb(path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((T, T))
            return im
    except Exception:
        return Image.new("RGB", (T, T), (200, 0, 0))


for i, r in enumerate(rows):
    y = i * (T + 24)
    sheet.paste(thumb(r["removed_image"]), (0, y + 20))
    sheet.paste(thumb(r["kept_duplicate_of"]), (T + 20, y + 20))
    draw.text((4, y + 4),
              f"#{i + 1}  full dist {r['full_hash_dist']}  crop dist {r['crop_hash_dist']}",
              fill="black")

sheet.save(args.out, quality=85)
print(f"Wrote {args.out} with {len(rows)} pairs out of {sum(1 for _ in open(args.report)) - 1} removed")