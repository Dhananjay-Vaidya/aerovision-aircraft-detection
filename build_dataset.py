#!/usr/bin/env python3
"""
Build the merged CVS aircraft dataset (YOLO format) from three Roboflow sources:

  1. dhananjay/aircraft-detection-rtt9x  v2   (your main project; augmented x3 -> collapsed back to 1 per original)
  2. dhananjay/fyp-uagdf                 v32  (your FYP project; clean, unaugmented)
  3. superradarosint/merge_dataset-9nrmd v1   (Universe, CC BY 4.0 -- credit it in the paper)

Steps
  - download   : pulls each source in YOLO format (needs ROBOFLOW_API_KEY in the environment)
  - harmonise  : maps ~250 source class names onto 35 target classes + "other_aircraft"
  - dedupe     : exact (MD5) + near-duplicate (perceptual hash) removal across all sources
  - balance    : caps images per class so F-16 / C-130 don't swamp the rare classes
  - split      : fresh 70 / 20 / 10 train / val / test split, stratified by each image's rarest class
  - write      : YOLO folders, data.yaml, class_affiliation.json, and stats CSVs for the paper
  - leakage    : val vs train and test vs train+val, invariant to hflip / vflip / rot180
                 -> val_clean.txt, test_clean.txt, data_clean.yaml; counts in build_report.json

Usage
  pip install roboflow imagehash pillow pyyaml
  export ROBOFLOW_API_KEY=xxxx            # keep it in a gitignored .env, never in code
  python build_dataset.py --download      # first run: download + build
  python build_dataset.py                 # later runs: rebuild from the local downloads

  Options: --raw-dir raw --out-dir cvs_dataset --cap 1000 --other-cap 3000 --phash-dist 4 --seed 42
"""

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image

# --------------------------------------------------------------------------------------
# Sources, in priority order: when two sources hold the same picture, the earlier one wins.
# --------------------------------------------------------------------------------------
SOURCES = [
    {"tag": "own", "workspace": "dhananjay", "project": "aircraft-detection-rtt9x", "version": 2,
     "collapse_augmented": True},
    {"tag": "fyp", "workspace": "dhananjay", "project": "fyp-uagdf", "version": 32,
     "collapse_augmented": False},
    {"tag": "merge", "workspace": "superradarosint", "project": "merge_dataset-9nrmd", "version": 1,
     "collapse_augmented": False},
]

# --------------------------------------------------------------------------------------
# Target classes and the affiliation each maps to (used later by the abstention layer).
# Order here = class id order in data.yaml.
# --------------------------------------------------------------------------------------
TARGET_CLASSES = {
    # Allied (India)
    "Su-30MKI": "allied", "Rafale": "allied", "Tejas": "allied", "Mirage-2000": "allied",
    "MiG-29": "allied", "Jaguar": "allied", "C-17": "allied", "C-130J": "allied",
    "C-295": "allied", "An-32": "allied", "AH-64": "allied", "CH-47": "allied",
    "HAL-Prachand": "allied", "HAL-Dhruv": "allied", "HAL-Rudra": "allied", "HAL-LUH": "allied",
    "HAL-Chetak": "allied", "Mi-26": "allied",
    # Adversary (China)
    "J-20": "adversary", "J-10": "adversary", "J-16": "adversary", "J-11": "adversary",
    "H-6": "adversary", "Y-20": "adversary", "JH-7": "adversary", "KJ-500": "adversary",
    "Z-10": "adversary", "WZ-7": "adversary",
    # Adversary (Pakistan)
    "JF-17": "adversary", "F-16": "adversary", "Mirage-III-V": "adversary",
    "Saab-2000-Erieye": "adversary",
    # Flown by more than one side -> the system must answer "uncertain"
    "Mi-8-17": "ambiguous", "Il-76-78": "ambiguous", "Mi-24-35": "ambiguous",
    # Every other aircraft type (non-theatre military, civil, trainers, drones)
    "other_aircraft": "unknown",
}
OTHER = "other_aircraft"
DROP = None  # box is deleted (junk labels)

# Source class name -> target class. Keys are normalised (lowercase, letters+digits only),
# so "EADSCASA C-295", "EADSCASAC-295" and "EADS-CASA-C-295" all hit the same entry.
# Anything not listed becomes "other_aircraft".
ALIASES = {
    # India
    "sukhoisu30": "Su-30MKI", "su30mki": "Su-30MKI", "su30": "Su-30MKI",
    "rafale": "Rafale", "rafael": "Rafale",
    "tejas": "Tejas", "haltejas": "Tejas",
    "mirage2000": "Mirage-2000",
    "mig29": "MiG-29",
    "jaguar": "Jaguar",
    "c17": "C-17", "boeingc17": "C-17",
    "c130jsuperhercules": "C-130J", "c130j": "C-130J",
    "eadscasac295": "C-295", "c295": "C-295",
    "antonovan32": "An-32", "an32": "An-32",
    "ah64": "AH-64", "boeingah64apache": "AH-64",
    "ch47": "CH-47", "ch47chinook": "CH-47",
    "halprachand": "HAL-Prachand",
    "haldhruv": "HAL-Dhruv",
    "halrudra": "HAL-Rudra",
    "hallightutilityhelicopter": "HAL-LUH",
    "halchetak": "HAL-Chetak",
    "mi26": "Mi-26",
    # China
    "j20": "J-20",
    "j10": "J-10", "j10b": "J-10", "j10vigorousdragon": "J-10",
    "j16": "J-16", "j16d": "J-16",
    "j11": "J-11", "j11b": "J-11", "j11d": "J-11",
    "h6": "H-6",
    "y20": "Y-20", "yy20": "Y-20",
    "jh7": "JH-7",
    "kj500": "KJ-500",
    "z10": "Z-10",
    "wz7": "WZ-7",
    # Pakistan
    "jf17": "JF-17", "jf17thunder": "JF-17", "fc1": "JF-17",
    "f16": "F-16", "f16fightingfalcon": "F-16",
    "mirageiii": "Mirage-III-V", "mirage5": "Mirage-III-V",
    "saab2000erieye": "Saab-2000-Erieye", "saab2000": "Saab-2000-Erieye",
    # Ambiguous
    "mi8": "Mi-8-17", "milmi8": "Mi-8-17", "milmi17": "Mi-8-17", "milmi817": "Mi-8-17",
    "mi17pak": "Mi-8-17", "mi8china": "Mi-8-17", "mi171china": "Mi-8-17",
    "il76": "Il-76-78", "ilyushinil76": "Il-76-78", "lyushinil76": "Il-76-78",
    "ilyushinil78": "Il-76-78", "il78pak": "Il-76-78", "ii78china": "Il-76-78",
    "mi24": "Mi-24-35", "milmi24": "Mi-24-35",
    # Junk labels found in the FYP project
    "": DROP, "aircraft": DROP,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.20, 0.10)
AUG_NAME = re.compile(r"^(?P<stem>.+?)_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-fA-F]+$")


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def map_class(name: str):
    key = norm(name)
    if key in ALIASES:
        return ALIASES[key]
    return OTHER


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def download_sources(raw_dir: Path):
    from roboflow import Roboflow  # imported here so the build step works without it

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in your environment (or a gitignored .env) first.")
    rf = Roboflow(api_key=api_key)
    for src in SOURCES:
        dest = raw_dir / src["tag"]
        if dest.exists() and any(dest.iterdir()):
            print(f"[download] {src['tag']}: already present at {dest}, skipping")
            continue
        print(f"[download] {src['workspace']}/{src['project']} v{src['version']} -> {dest}")
        rf.workspace(src["workspace"]).project(src["project"]).version(src["version"]).download(
            "yolov8", location=str(dest), overwrite=True)


# --------------------------------------------------------------------------------------
# Load one YOLO export
# --------------------------------------------------------------------------------------
def load_names(export_dir: Path):
    data_yaml = export_dir / "data.yaml"
    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):  # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    return names


def find_split_dirs(export_dir: Path):
    out = []
    for split in ("train", "valid", "val", "test"):
        img_dir = export_dir / split / "images"
        if img_dir.is_dir():
            out.append((split, img_dir, export_dir / split / "labels"))
    return out


def read_labels(label_path: Path, names, stats):
    """Return list of (target_class, 'cx cy w h') for one label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id < 0 or cls_id >= len(names):
            stats["bad_class_id"] += 1
            continue
        target = map_class(names[cls_id])
        if target is DROP:
            stats["dropped_junk_boxes"] += 1
            continue
        if len(parts) > 5:
            # polygon line: convert to its bounding box
            pts = list(map(float, parts[1:]))
            xs, ys = pts[0::2], pts[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            coords = f"{(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
        else:
            coords = " ".join(parts[1:5])
        boxes.append((target, coords))
    return boxes


def load_source(src, raw_dir: Path, stats):
    """Yield candidate records for one source, collapsing augmented copies if asked."""
    export_dir = raw_dir / src["tag"]
    if not (export_dir / "data.yaml").exists():
        raise SystemExit(f"Missing {export_dir}/data.yaml. Run with --download first.")
    names = load_names(export_dir)

    records = []
    for split, img_dir, lbl_dir in find_split_dirs(export_dir):
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            boxes = read_labels(lbl_dir / (img_path.stem + ".txt"), names, stats)
            stats[f"{src['tag']}_files_seen"] += 1
            if not boxes:
                stats[f"{src['tag']}_unlabelled_skipped"] += 1
                continue
            m = AUG_NAME.match(img_path.stem)
            group = m.group("stem") if m else img_path.stem
            records.append({"source": src["tag"], "orig_split": split, "path": img_path,
                            "boxes": boxes, "group": group})

    if src["collapse_augmented"]:
        # Roboflow only augments the train split, so prefer a valid/test copy of each original.
        best = {}
        for r in records:
            rank = 0 if r["orig_split"] in ("valid", "val", "test") else 1
            if r["group"] not in best or rank < best[r["group"]][0]:
                best[r["group"]] = (rank, r)
        stats[f"{src['tag']}_augmented_copies_removed"] += len(records) - len(best)
        records = [v[1] for v in best.values()]
    return records


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------
def _popcount(x: int) -> int:
    return bin(x).count("1")


def _decode_for_hash(data: bytes) -> Image.Image:
    """Memory-light decode used for every pHash: JPEGs are decoded at reduced scale via draft()
    (pHash works on 32x32 anyway)."""
    with Image.open(io.BytesIO(data)) as im:
        try:
            im.draft("RGB", (512, 512))
        except Exception:
            pass
        if im.width * im.height > 25_000_000:  # huge PNG/TIFF: shrink before converting
            im.thumbnail((2048, 2048))
        return im.convert("RGB")


def _phash_int(im: Image.Image) -> int:
    return int(str(imagehash.phash(im)), 16)


def _image_hashes(path: Path, boxes):
    """Return (md5_hex, full_phash_int, crop_phash_int)."""
    data = path.read_bytes()
    digest = hashlib.md5(data).hexdigest()
    im = _decode_for_hash(data)
    full = _phash_int(im)
    crop = _phash_int(_largest_box_crop(im, boxes))
    del data
    return digest, full, crop


class HashCache:
    """Tab-separated cache so a crash or re-run resumes instead of re-hashing 60k images."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) == 6:
                        self.data[(p[0], p[1], p[2])] = (p[3], int(p[4]), int(p[5]))
        self.fh = open(path, "a", encoding="utf-8")
        self.pending = 0

    @staticmethod
    def key(path: Path):
        st = path.stat()
        return (str(path), str(st.st_size), str(st.st_mtime_ns))

    def get(self, path: Path):
        return self.data.get(self.key(path))

    def put(self, path: Path, value):
        k = self.key(path)
        self.data[k] = value
        self.fh.write("\t".join([*k, value[0], str(value[1]), str(value[2])]) + "\n")
        self.pending += 1
        if self.pending >= 500:
            self.fh.flush()
            self.pending = 0

    def close(self):
        self.fh.close()


def _largest_box_crop(im: Image.Image, boxes):
    """Crop the largest labelled box (+10% margin). Aircraft photos are mostly plain sky,
    so a whole-image hash alone calls different photos 'duplicates'; the crop hash
    compares the aircraft itself."""
    W, H = im.size
    best, best_area = None, -1.0
    for _, coords in boxes:
        cx, cy, w, h = map(float, coords.split()[:4])
        if w * h > best_area:
            best, best_area = (cx, cy, w, h), w * h
    cx, cy, w, h = best
    w, h = w * 1.1, h * 1.1
    x0, x1 = max(0, (cx - w / 2) * W), min(W, (cx + w / 2) * W)
    y0, y1 = max(0, (cy - h / 2) * H), min(H, (cy + h / 2) * H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return im
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


def dedupe(records, max_dist, stats, report_path: Path = None, cache_path: Path = None):
    """Keep the first occurrence (records are already in source-priority order).

    An image counts as a near-duplicate only if BOTH its whole-image pHash AND the pHash
    of its main aircraft crop are within max_dist of an image already kept.
    Hashes are stored as plain ints and cached on disk, so memory stays low and
    an interrupted run resumes where it stopped.
    """
    cache = HashCache(cache_path) if cache_path else None
    seen_md5 = {}
    bands = [defaultdict(list) for _ in range(4)]  # banded on the crop hash (more distinctive)
    kept = []
    removed_rows = []
    removed_per_class = Counter()
    total = len(records)
    try:
        for n, r in enumerate(records, 1):
            if n % 2000 == 0:
                print(f"[dedupe] {n}/{total} checked, {len(kept)} kept", flush=True)
            cached = cache.get(r["path"]) if cache else None
            if cached:
                digest, full, crop = cached
            else:
                try:
                    digest, full, crop = _image_hashes(r["path"], r["boxes"])
                except (Exception, MemoryError):
                    stats["unreadable_images_skipped"] += 1
                    gc.collect()
                    continue
                if cache:
                    cache.put(r["path"], (digest, full, crop))
            if n % 5000 == 0:
                gc.collect()

            if digest in seen_md5:
                stats["exact_duplicates_removed"] += 1
                removed_rows.append(["exact", str(r["path"]), seen_md5[digest], 0, 0])
                for c in {b[0] for b in r["boxes"]}:
                    removed_per_class[c] += 1
                continue
            keys = [(crop >> (16 * i)) & 0xFFFF for i in range(4)]
            match = None
            for i, k in enumerate(keys):
                for o_full, o_crop, o_path in bands[i].get(k, ()):
                    d_crop = _popcount(crop ^ o_crop)
                    if d_crop <= max_dist:
                        d_full = _popcount(full ^ o_full)
                        if d_full <= max_dist:
                            match = (o_path, d_full, d_crop)
                            break
                if match:
                    break
            if match:
                o_path, d_full, d_crop = match
                stats["near_duplicates_removed"] += 1
                removed_rows.append(["near", str(r["path"]), o_path, d_full, d_crop])
                for c in {b[0] for b in r["boxes"]}:
                    removed_per_class[c] += 1
                continue
            path_str = str(r["path"])
            seen_md5[digest] = path_str
            entry = (full, crop, path_str)
            for i, k in enumerate(keys):
                bands[i][k].append(entry)
            kept.append(r)
    finally:
        if cache:
            cache.close()

    for c, n in removed_per_class.items():
        stats[f"dedupe_removed__{c}"] = n
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["type", "removed_image", "kept_duplicate_of", "full_hash_dist", "crop_hash_dist"])
            w.writerows(removed_rows)
    return kept


# --------------------------------------------------------------------------------------
# Balancing and splitting
# --------------------------------------------------------------------------------------
def balance(records, cap, other_cap, rng, stats):
    """Keep every image of rare classes; stop adding images once all their classes are full."""
    freq = Counter(c for r in records for c in {b[0] for b in r["boxes"]})

    def rarity(r):
        targets = [c for c in {b[0] for b in r["boxes"]} if c != OTHER]
        return min((freq[c] for c in targets), default=10 ** 9)

    rng.shuffle(records)
    records.sort(key=rarity)  # stable sort: random order within equal rarity
    counts = Counter()
    kept = []
    for r in records:
        classes = {b[0] for b in r["boxes"]}
        targets = classes - {OTHER}
        if targets:
            if not any(counts[c] < cap for c in targets):
                stats["dropped_by_class_cap"] += 1
                continue
        elif counts[OTHER] >= other_cap:
            stats["dropped_by_other_cap"] += 1
            continue
        counts.update(classes)
        kept.append(r)
    return kept


def split_records(records, rng):
    freq = Counter(c for r in records for c in {b[0] for b in r["boxes"]})
    by_key = defaultdict(list)
    for r in records:
        classes = {b[0] for b in r["boxes"]}
        targets = classes - {OTHER} or classes
        key = min(targets, key=lambda c: (freq[c], c))  # rarest class in the image
        by_key[key].append(r)

    out = {s: [] for s in SPLITS}
    for key in sorted(by_key):
        group = by_key[key]
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * SPLIT_RATIOS[2])) if n >= 3 else 0
        n_val = max(1, round(n * SPLIT_RATIOS[1])) if n >= 3 else 0
        out["test"] += group[:n_test]
        out["val"] += group[n_test:n_test + n_val]
        out["train"] += group[n_test + n_val:]
    return out


# --------------------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------------------
def write_dataset(splits, out_dir: Path, stats):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    class_names = list(TARGET_CLASSES)
    class_id = {c: i for i, c in enumerate(class_names)}

    per_class = {c: Counter() for c in class_names}       # images per class per split
    per_class_boxes = {c: Counter() for c in class_names}  # boxes per class per split
    per_source = defaultdict(Counter)
    manifest_rows = []

    for split, recs in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(recs):
            name = f"{r['source']}_{split}_{i:06d}"
            dst_img = out_dir / "images" / split / (name + r["path"].suffix.lower())
            shutil.copy2(r["path"], dst_img)
            lines = [f"{class_id[c]} {coords}" for c, coords in r["boxes"]]
            (out_dir / "labels" / split / (name + ".txt")).write_text("\n".join(lines) + "\n")
            for c in {b[0] for b in r["boxes"]}:
                per_class[c][split] += 1
            for c, _ in r["boxes"]:
                per_class_boxes[c][split] += 1
            per_source[r["source"]][split] += 1
            manifest_rows.append([dst_img.name, split, r["source"], str(r["path"])])

    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with open(out_dir / "class_affiliation.json", "w") as f:
        json.dump(TARGET_CLASSES, f, indent=2)

    with open(out_dir / "stats_per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "affiliation", "train_images", "val_images", "test_images",
                    "total_images", "total_boxes"])
        for c in class_names:
            imgs = per_class[c]
            w.writerow([c, TARGET_CLASSES[c], imgs["train"], imgs["val"], imgs["test"],
                        sum(imgs.values()), sum(per_class_boxes[c].values())])

    with open(out_dir / "stats_per_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "train", "val", "test", "total"])
        for s in [src["tag"] for src in SOURCES]:
            cnt = per_source[s]
            w.writerow([s, cnt["train"], cnt["val"], cnt["test"], sum(cnt.values())])

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "source", "original_path"])
        w.writerows(manifest_rows)

    with open(out_dir / "build_report.json", "w") as f:
        json.dump(dict(sorted(stats.items())), f, indent=2)

    return per_class


# --------------------------------------------------------------------------------------
# Final step: flip/rotation-invariant leakage check on the written split
# --------------------------------------------------------------------------------------
# dedupe() compares images as they are; Roboflow exports also contain flipped / rotated copies of the
# same photo, which survive it and can land in different splits. Here every val image (vs train) and
# every test image (vs train + val) is hashed as-is and as hflip / vflip / rot180; it is flagged when, for
# some variant, BOTH the whole-image and the largest-box-crop pHash are within `max_dist` of one image.
# Flagged images are only left out of <split>_clean.txt; no file is moved or deleted.
LEAK_VARIANTS = {
    "none": None,
    "hflip": Image.Transpose.FLIP_LEFT_RIGHT,
    "vflip": Image.Transpose.FLIP_TOP_BOTTOM,
    "rot180": Image.Transpose.ROTATE_180,
}


def _variant_hashes(path: Path, boxes):
    """{variant: (full, crop)}; the crop is cut from the original and then transformed (= the crop of
    the transformed image), decoded exactly as in _image_hashes."""
    im = _decode_for_hash(path.read_bytes())
    crop = _largest_box_crop(im, boxes)
    return {v: (_phash_int(im if t is None else im.transpose(t)),
                _phash_int(crop if t is None else crop.transpose(t))) for v, t in LEAK_VARIANTS.items()}


def _hamming(arr, h: int):
    """Hamming distance between a uint64 array and one 64-bit hash."""
    import numpy as np
    x = arr ^ np.uint64(h)
    if hasattr(np, "bitwise_count"):  # numpy >= 2
        return np.bitwise_count(x).astype(np.int16)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).astype(np.int16)


def _split_boxes(out_dir: Path, split: str, img: Path):
    lbl = out_dir / "labels" / split / (img.stem + ".txt")
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                boxes.append((int(p[0]), " ".join(p[1:5])))
    return boxes


def leakage_check(out_dir: Path, max_dist: int = 6, cache_path: Path = None, log=print):
    """Returns one dict per val and test image: split, file, source, class_ids, flagged, and for a flagged
    image the best match (variant, match_file, match_split, d_full, d_crop) plus n_matched / variants_matched.
    Train hashes come from the hash cache when the raw file is still there (cvs files are copy2 copies)."""
    import numpy as np
    out_dir = Path(out_dir)
    with open(out_dir / "manifest.csv", encoding="utf-8") as f:
        manifest = {r["file"]: r for r in csv.DictReader(f)}
    cache = HashCache(Path(cache_path)) if cache_path and Path(cache_path).exists() else None

    def cached(img: Path):
        if cache is None or img.name not in manifest:
            return None
        raw = Path(manifest[img.name]["original_path"])
        if not raw.exists() or raw.stat().st_size != img.stat().st_size:
            return None
        v = cache.get(raw)
        return (v[1], v[2]) if v else None

    def load(split, variants):
        recs, fresh = [], 0
        for img in sorted((out_dir / "images" / split).iterdir()):
            boxes = _split_boxes(out_dir, split, img)
            rec = {"split": split, "file": img.name, "source": manifest.get(img.name, {}).get("source", ""),
                   "class_ids": sorted({b[0] for b in boxes}), "vh": None}
            if boxes:
                ident = None if variants else cached(img)
                if ident is None:
                    rec["vh"] = _variant_hashes(img, boxes)
                    fresh += 1
                else:
                    rec["vh"] = {"none": ident}
            recs.append(rec)
        log(f"[leakage] {split}: {len(recs)} images ({fresh} hashed fresh, {len(recs) - fresh} from cache)")
        return recs

    train, val, test = load("train", False), load("val", True), load("test", True)
    if cache is not None:
        cache.close()

    def check(queries, refs):
        refs = [r for r in refs if r["vh"]]
        R_full = np.array([r["vh"]["none"][0] for r in refs], dtype=np.uint64)
        R_crop = np.array([r["vh"]["none"][1] for r in refs], dtype=np.uint64)
        for q in queries:
            q.update(flagged=0, variant="", match_file="", match_split="", d_full="", d_crop="",
                     n_matched=0, variants_matched="")
            if not q["vh"]:
                continue
            hits = []
            for v, (full, crop) in q["vh"].items():
                df, dc = _hamming(R_full, full), _hamming(R_crop, crop)
                hits += [(int(df[i]) + int(dc[i]), list(LEAK_VARIANTS).index(v), v, i, int(df[i]), int(dc[i]))
                         for i in np.flatnonzero((df <= max_dist) & (dc <= max_dist))]
            if hits:
                _, _, v, i, d_full, d_crop = min(hits)  # closest match; ties -> untransformed first
                q.update(flagged=1, variant=v, match_file=refs[i]["file"], match_split=refs[i]["split"],
                         d_full=d_full, d_crop=d_crop, n_matched=len({h[3] for h in hits}),
                         variants_matched=";".join(sorted({h[2] for h in hits})))

    check(val, train)
    check(test, train + val)  # all val images, flagged or not: conservative
    rows = val + test
    for r in rows:
        r.pop("vh")
    return rows


# ---- label-only geometry rules for the clean lists (no model predictions, no manual decisions) ----
GT_OVERLAP_IOU = 0.5     # rule (i)
SHARED_X_TOL = 0.002     # rule (ii), in normalised image width


def gt_boxes_xyxy(out_dir: Path, split: str, file: str):
    """GT boxes of an image as normalised (x0, y0, x1, y1), from its YOLO label file."""
    import numpy as np
    bs = []
    for _, coords in _split_boxes(Path(out_dir), split, Path(file)):
        xc, yc, w, h = map(float, coords.split())
        bs.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return np.array(bs).reshape(-1, 4)


def gt_overlap(boxes, thr: float = GT_OVERLAP_IOU) -> bool:
    """Rule (i): some two GT boxes of the image have IoU >= thr."""
    import numpy as np
    if len(boxes) < 2:
        return False
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-12)
    np.fill_diagonal(iou, 0)
    return bool((iou >= thr).any())


def shared_x_groups(boxes, tol: float = SHARED_X_TOL):
    """Rule (ii), "stacked boxes on one x-range": groups (lists of box indices, size >= 2) of GT boxes whose left
    edges x0 AND right edges x1 both agree within `tol` (normalised image width) with the first box of the group.
    Boxes are taken in label-file order; each box belongs to at most one group. This is the signature of an export
    defect in which several boxes of an image were given the horizontal extent of one aircraft (each keeping its
    own y0 / y1); it also catches exact duplicate boxes and genuinely aligned aircraft (in-trail columns)."""
    groups, seen = [], set()
    for i in range(len(boxes)):
        if i in seen:
            continue
        g = [j for j in range(len(boxes)) if j not in seen
             and abs(boxes[j, 0] - boxes[i, 0]) <= tol and abs(boxes[j, 2] - boxes[i, 2]) <= tol]
        if len(g) >= 2:
            groups.append(g)
        seen.update(g)
    return groups


CLEAN_RULES = ("leak", "gt_overlap", "shared_x")


def write_clean_splits(out_dir: Path, rows, stats=None, max_dist: int = 6):
    """Write <out>/val_clean.txt, <out>/test_clean.txt and <out>/data_clean.yaml (train as in data.yaml; val/test ->
    the lists). An image is left out of a clean list if
      (leak)       it is flagged by the flip/rotation leakage check, or
      (gt_overlap) two of its GT boxes overlap at IoU >= 0.5 (rule i), or
      (shared_x)   two of its GT boxes share one x-range: |dx0| <= 0.002 and |dx1| <= 0.002 (rule ii).
    Only the labels and the image hashes are used. Sets r["gt_overlap"], r["shared_x"], r["in_clean"] and
    r["removed_by"] (first rule that applies, in the order above) on every row; adds counts to `stats`."""
    out_dir = Path(out_dir).resolve()
    d = yaml.safe_load((out_dir / "data.yaml").read_text())
    for r in rows:
        b = gt_boxes_xyxy(out_dir, r["split"], r["file"])
        r["gt_overlap"] = int(gt_overlap(b))
        r["shared_x"] = int(bool(shared_x_groups(b)))
        r["removed_by"] = next((k for k, v in (("leak", r["flagged"]), ("gt_overlap", r["gt_overlap"]),
                                               ("shared_x", r["shared_x"])) if v), "")
        r["in_clean"] = int(not r["removed_by"])
    counts = {}
    for split in ("val", "test"):
        sr = [r for r in rows if r["split"] == split]
        keep = [(out_dir / "images" / split / r["file"]).as_posix() for r in sr if r["in_clean"]]
        (out_dir / f"{split}_clean.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
        counts[split] = (len(sr), sum(1 for r in sr if r["flagged"]), len(keep))
        if stats is not None:
            for k in CLEAN_RULES:  # removed by this rule and not by an earlier one
                stats[f"clean_removed_{k}_{split}"] = sum(1 for r in sr if r["removed_by"] == k)
            stats[f"gt_overlap_{split}_images"] = sum(r["gt_overlap"] for r in sr)  # regardless of other rules
            stats[f"shared_x_{split}_images"] = sum(r["shared_x"] for r in sr)
    clean = {"path": out_dir.as_posix(), "train": d["train"], "val": "val_clean.txt", "test": "test_clean.txt",
             "nc": d["nc"], "names": d["names"]}
    with open(out_dir / "data_clean.yaml", "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
    if stats is not None:
        stats["leakage_max_dist"] = max_dist
        stats["clean_rule_gt_overlap_iou"] = GT_OVERLAP_IOU
        stats["clean_rule_shared_x_tol"] = SHARED_X_TOL
        for split, (n, fl, kept) in counts.items():
            stats[f"leakage_{split}_images"] = n
            stats[f"leakage_{split}_flagged"] = fl
            stats[f"leakage_{split}_clean"] = kept
        for v in LEAK_VARIANTS:
            stats[f"leakage_flagged_variant_{v}"] = sum(1 for r in rows if r["flagged"] and r["variant"] == v)
    return counts


def update_build_report(out_dir: Path, stats):
    """Merge `stats` into <out>/build_report.json (keys kept sorted)."""
    p = Path(out_dir) / "build_report.json"
    rep = json.loads(p.read_text()) if p.exists() else {}
    rep.update(dict(stats))
    p.write_text(json.dumps(dict(sorted(rep.items())), indent=2))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="download the Roboflow exports first")
    ap.add_argument("--raw-dir", default="raw", type=Path)
    ap.add_argument("--out-dir", default="cvs_dataset", type=Path)
    ap.add_argument("--cap", type=int, default=1000, help="max images per target class")
    ap.add_argument("--other-cap", type=int, default=3000, help="max images holding only other_aircraft")
    ap.add_argument("--phash-dist", type=int, default=4, help="near-duplicate Hamming threshold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak-dist", type=int, default=6,
                    help="flip/rotation leakage check: Hamming threshold on whole AND crop pHash")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = Counter()

    if args.download:
        download_sources(args.raw_dir)

    records = []
    for src in SOURCES:
        recs = load_source(src, args.raw_dir, stats)
        print(f"[load] {src['tag']}: {len(recs)} labelled originals")
        records.extend(recs)

    report = args.out_dir.parent / f"{args.out_dir.name}_dedupe_report.csv"
    records = dedupe(records, args.phash_dist, stats, report, args.raw_dir / "hash_cache.tsv")
    print(f"[dedupe] removed pairs written to {report} (open a few to check they really match)")
    print(f"[dedupe] {len(records)} unique images "
          f"(exact {stats['exact_duplicates_removed']}, near {stats['near_duplicates_removed']} removed)")

    records = balance(records, args.cap, args.other_cap, rng, stats)
    print(f"[balance] {len(records)} images after caps")

    splits = split_records(records, rng)
    per_class = write_dataset(splits, args.out_dir, stats)

    print(f"[write] {args.out_dir}/  train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])}")

    leak_stats = Counter()
    rows = leakage_check(args.out_dir, args.leak_dist, args.raw_dir / "hash_cache.tsv")
    counts = write_clean_splits(args.out_dir, rows, leak_stats, args.leak_dist)
    update_build_report(args.out_dir, leak_stats)
    for split, (n, fl, kept) in counts.items():
        print(f"[clean] {split}: {n} images; removed {leak_stats[f'clean_removed_leak_{split}']} leaked, "
              f"{leak_stats[f'clean_removed_gt_overlap_{split}']} GT boxes at IoU >= 0.5, "
              f"{leak_stats[f'clean_removed_shared_x_{split}']} GT boxes sharing one x-range -> {kept} in "
              f"{split}_clean.txt")
    print(f"[leakage] wrote {args.out_dir}/val_clean.txt, test_clean.txt, data_clean.yaml "
          f"(python audit_leakage.py for the full report)")
    thin = [c for c in TARGET_CLASSES if c != OTHER and sum(per_class[c].values()) < 300]
    if thin:
        print("[warn] classes under 300 images (collect more before training):")
        for c in thin:
            print(f"        {c}: {sum(per_class[c].values())}")


if __name__ == "__main__":
    main()