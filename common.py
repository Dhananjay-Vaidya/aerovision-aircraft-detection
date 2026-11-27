"""Helpers shared by train.py, corrupt.py, evaluate.py, abstain.py and figures.py."""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent

# ---- Aircraft-C corruption groups ---------------------------------------------------------------------------------
# "seen": the same kinds of degradation that train.py --aug degrade trains on (train.py asserts it uses exactly these).
# "held_out": never used in training.
SEEN_CORRUPTIONS = ("defocus_blur", "motion_blur", "fog", "low_light", "gaussian_noise", "jpeg", "downscale")
HELD_OUT_CORRUPTIONS = ("contrast", "rain")
CORRUPTION_GROUP = {**{c: "seen" for c in SEEN_CORRUPTIONS}, **{c: "held_out" for c in HELD_OUT_CORRUPTIONS}}

# ---- class merging --------------------------------------------------------------------------------------------------
MIN_TRAIN_IMAGES = 50
OTHER_CLASS = "other_aircraft"
MERGED_DIR = ROOT / "data_derived" / "merged"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def load_names(data_yaml: str | Path) -> list[str]:
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = d["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def load_affiliation(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_workers() -> int:
    return 2 if os.name == "nt" else min(8, os.cpu_count() or 2)


def _label_dir(base: Path, split_entry: str) -> Path:
    return base / str(split_entry).replace("images", "labels")


def class_merge_map(data_yaml: str | Path, min_train_images: int = MIN_TRAIN_IMAGES):
    """THE class-merging rule, used by every script: a class with fewer than `min_train_images` training images
    (images containing >= 1 box of it) is relabelled other_aircraft. Class list and ids are unchanged; merged
    classes simply have no boxes. Returns (id_map {old_id: new_id}, merged names, train image counts)."""
    data_yaml = Path(data_yaml).resolve()
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = load_names(data_yaml)
    other = names.index(OTHER_CLASS)
    counts = Counter()
    for f in _label_dir(data_yaml.parent, d["train"]).glob("*.txt"):
        counts.update({int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()})
    merged = [i for i in range(len(names)) if i != other and counts[i] < min_train_images]
    id_map = {i: (other if i in merged else i) for i in range(len(names))}
    return id_map, [names[i] for i in merged], {names[i]: counts[i] for i in range(len(names))}


def remap_label_text(text: str, id_map: dict[int, int]) -> str:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            out.append(" ".join([str(id_map[int(parts[0])]), *parts[1:]]))
    return "\n".join(out) + ("\n" if out else "")


def _link(src: Path, dst: Path):
    """Hardlink (same volume), else symlink (e.g. read-only /kaggle/input), else copy. Re-links if stale."""
    if dst.exists() or dst.is_symlink():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        dst.unlink()
    for fn in (os.link, os.symlink, shutil.copy2):
        try:
            fn(src, dst)
            return
        except OSError:
            continue
    raise OSError(f"could not link or copy {src} -> {dst}")


SPLIT_SETS = ("clean", "full")


def clean_names(dataset_dir: str | Path, split: str) -> set[str]:
    """File names listed in <dataset>/<split>_clean.txt (written by build_dataset.py / audit_leakage.py)."""
    lst = Path(dataset_dir) / f"{split}_clean.txt"
    if not lst.exists():
        raise SystemExit(f"{lst} not found: run `python audit_leakage.py` first (or use --split-set full)")
    return {Path(line.strip()).name for line in lst.read_text(encoding="utf-8").splitlines() if line.strip()}


def merged_dataset(data_yaml: str | Path, out_dir: str | Path = MERGED_DIR, results_dir: str | Path | None = None,
                   min_train_images: int = MIN_TRAIN_IMAGES, split_set: str = "full") -> Path:
    """Build (or refresh) the merged view of the dataset: images linked, labels remapped with class_merge_map.
    Writes results/merged_classes.txt. Returns the merged data.yaml. `cvs_dataset/` is never written to.

    split_set="clean": also writes <out>/val_clean.txt, <out>/test_clean.txt (merged-view paths of the images in
    <dataset>/{val,test}_clean.txt, i.e. without train/val near-copies) and returns <out>/data_clean.yaml, whose
    val/test entries are those lists. Train is the same in both."""
    if split_set not in SPLIT_SETS:
        raise ValueError(f"split_set must be one of {SPLIT_SETS}")
    full_yaml = _merged_full(data_yaml, out_dir, results_dir, min_train_images)
    if split_set == "full":
        return full_yaml
    out = Path(out_dir)
    d = yaml.safe_load(full_yaml.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        keep = clean_names(Path(data_yaml).resolve().parent, split)
        imgs = [p for p in list_images(out / d[split]) if p.name in keep]
        missing = keep - {p.name for p in imgs}
        if missing:
            raise SystemExit(f"{len(missing)} {split}_clean images not in {out / d[split]}, e.g. {sorted(missing)[:3]}")
        lst = out / f"{split}_clean.txt"
        lst.write_text("\n".join(p.resolve().as_posix() for p in imgs) + "\n", encoding="utf-8")
        d[split] = lst.name
    out_yaml = out / "data_clean.yaml"
    out_yaml.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_yaml


def split_images(data_yaml: str | Path, split: str) -> list[Path]:
    """Images of a split from a data yaml whose entry is a folder or a .txt list."""
    data_yaml = Path(data_yaml)
    entry = data_yaml.parent / yaml.safe_load(data_yaml.read_text(encoding="utf-8"))[split]
    if entry.suffix == ".txt":
        return [Path(l.strip()) for l in entry.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list_images(entry)


def merged_labels_dir(data_yaml: str | Path, split: str) -> Path:
    """Label folder of a split in the merged view (same for full and clean)."""
    return Path(data_yaml).parent / "labels" / split


def _merged_full(data_yaml, out_dir, results_dir, min_train_images) -> Path:
    data_yaml = Path(data_yaml).resolve()
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base, out = data_yaml.parent, Path(out_dir)
    id_map, merged, counts = class_merge_map(data_yaml, min_train_images)
    new = {"path": out.resolve().as_posix()}
    for split in ("train", "val", "test"):
        if split not in d:
            continue
        src_img, src_lbl = base / d[split], _label_dir(base, d[split])
        dst_img, dst_lbl = out / "images" / split, out / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        imgs = list_images(src_img)
        keep, keep_stems = {p.name for p in imgs}, {p.stem for p in imgs}
        for p in dst_img.iterdir():                     # drop images no longer in the source split
            if p.name not in keep:
                p.unlink()
        for p in dst_lbl.glob("*.txt"):
            if p.stem not in keep_stems:
                p.unlink()
        for p in imgs:
            _link(p, dst_img / p.name)
            lbl = src_lbl / (p.stem + ".txt")
            text = remap_label_text(lbl.read_text(), id_map) if lbl.exists() else ""
            dst = dst_lbl / (p.stem + ".txt")
            if not dst.exists() or dst.read_text() != text:
                dst.write_text(text)
        for cache in out.glob("labels/*.cache"):        # ultralytics label caches would be stale
            if cache.stem == split:
                cache.unlink()
        new[split] = f"images/{split}"
    new["nc"] = len(load_names(data_yaml))
    new["names"] = load_names(data_yaml)
    aff = base / "class_affiliation.json"
    if aff.exists():
        shutil.copyfile(aff, out / "class_affiliation.json")
    out_yaml = out / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_merged_list(merged, counts, results_dir or ROOT / "results", min_train_images)
    return out_yaml


def write_merged_list(merged, counts, results_dir, min_train_images=MIN_TRAIN_IMAGES):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# classes with < {min_train_images} training images, relabelled {OTHER_CLASS}",
             "# class\ttrain_images", *[f"{n}\t{counts[n]}" for n in merged]]
    (results_dir / "merged_classes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def portable_data_yaml(data_yaml: str | Path, out_dir: str | Path, limit: int | None = None, seed: int = 0) -> Path:
    """Write a copy of data.yaml whose `path` is the yaml's own folder. With `limit`, train/val/test become .txt
    lists of at most `limit` images each (seeded sample), for smoke runs. Returns the path of the new yaml."""
    data_yaml = Path(data_yaml).resolve()
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    d["path"] = base.as_posix()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{base.name}_limit{limit}" if limit else base.name
    if limit:
        rng = random.Random(seed)
        for split in ("train", "val", "test"):
            if split not in d:
                continue
            imgs = list_images(base / d[split])
            if len(imgs) > limit:
                imgs = sorted(rng.sample(imgs, limit))
            lst = out_dir / f"{tag}_{split}.txt"
            lst.write_text("\n".join(p.as_posix() for p in imgs) + "\n", encoding="utf-8")
            d[split] = lst.resolve().as_posix()
    out = out_dir / f"{tag}.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
