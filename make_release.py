"""Build dataset_release/: labels, splits and metadata only (never images).

  python make_release.py

Contents (all generated from cvs_dataset/, data_derived/ and results/ except LICENSE and rebuild.md, which are
written here too so the release is reproducible):
  labels/{train,val,test}/*.txt   YOLO labels after the class merge (common.merged_dataset)
  classes.yaml                    id -> name, nc, merged classes
  class_affiliation.json, merged_classes.txt, build_report.json, leakage_stats.txt
  splits/{train,val_full,test_full,val_clean,test_clean}.txt   relative paths images/<split>/<file>
  sources.csv                     file, split, source, original Roboflow filename, project, version, URL
  aircraft_c_manifest.json        corruption names, groups, parameters, severities, seed
  README.md (dataset card), LICENSE, rebuild.md
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from build_dataset import SOURCES
from common import IMG_EXTS, MIN_TRAIN_IMAGES, OTHER_CLASS, ROOT, class_merge_map, clean_names, list_images, \
    load_affiliation, load_names, merged_dataset

DS = ROOT / "cvs_dataset"
OUT = ROOT / "dataset_release"
URL = {s["tag"]: f"https://universe.roboflow.com/{s['workspace']}/{s['project']}" for s in SOURCES}
REPO = "https://github.com/Dhananjay-Vaidya/aerovision-aircraft-detection"
TITLE = "AeroVision: A Leakage-Audited Dataset and Corruption Benchmark for Robust Military Aircraft Identification (dataset labels, splits, code and results)"
VERSION = "1.0.0"
SPLITS = ("train", "val", "test")


def counts_per_class(label_dir: Path, files, n_cls):
    imgs, boxes = Counter(), Counter()
    for f in files:
        ids = [int(l.split()[0]) for l in (label_dir / (Path(f).stem + ".txt")).read_text().splitlines() if l.strip()]
        boxes.update(ids)
        imgs.update(set(ids))
    return imgs, boxes


def main():
    from label_fixes import FIX_FILE  # dataset_release/suggested_label_fixes.json: kept across regenerations
    keep_fixes = FIX_FILE.read_bytes() if FIX_FILE.exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "splits").mkdir(parents=True)
    if keep_fixes is not None:
        FIX_FILE.write_bytes(keep_fixes)
    names = load_names(DS / "data.yaml")
    aff = load_affiliation(DS / "class_affiliation.json")
    id_map, merged, train_counts = class_merge_map(DS / "data.yaml")
    myaml = merged_dataset(DS / "data.yaml")
    mroot = myaml.parent

    # ---- labels (merged) ----
    files = {}
    for split in SPLITS:
        imgs = list_images(DS / "images" / split)
        files[split] = [p.name for p in imgs]
        dst = OUT / "labels" / split
        dst.mkdir(parents=True)
        for p in imgs:
            shutil.copyfile(mroot / "labels" / split / (p.stem + ".txt"), dst / (p.stem + ".txt"))

    # ---- splits (relative paths only) ----
    clean = {s: clean_names(DS, s) for s in ("val", "test")}
    lists = {"train": ("train", files["train"]),
             "val_full": ("val", files["val"]), "test_full": ("test", files["test"]),
             "val_clean": ("val", [f for f in files["val"] if f in clean["val"]]),
             "test_clean": ("test", [f for f in files["test"] if f in clean["test"]])}
    for key, (split, fl) in lists.items():
        (OUT / "splits" / f"{key}.txt").write_text("".join(f"images/{split}/{f}\n" for f in fl), encoding="utf-8")

    # ---- classes, affiliation, merged list, reports ----
    (OUT / "classes.yaml").write_text(yaml.safe_dump({
        "nc": len(names), "names": dict(enumerate(names)),
        "merged_into_other": {OTHER_CLASS: merged},
        "merge_rule": f"classes with < {MIN_TRAIN_IMAGES} training images are relabelled {OTHER_CLASS}; "
                      "ids are unchanged, so merged ids have no boxes",
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(DS / "class_affiliation.json", OUT / "class_affiliation.json")
    shutil.copyfile(ROOT / "results" / "merged_classes.txt", OUT / "merged_classes.txt")
    shutil.copyfile(DS / "build_report.json", OUT / "build_report.json")
    shutil.copyfile(ROOT / "paper_assets" / "leakage_stats.txt", OUT / "leakage_stats.txt")

    # ---- suggested label fixes (errata, not applied) and the clean-list rule counts ----
    from build_dataset import gt_overlap, shared_x_groups
    from label_fixes import load_fixes, parse
    fixes = load_fixes(FIX_FILE)
    fix_action = {e["file"]: e["action"] for e in fixes["entries"]}
    train_boxes = [parse(p.read_text())[1] for p in (DS / "labels" / "train").glob("*.txt")]
    fx = {"n_fix": len(fixes["entries"]), "localiser": fixes.get("localiser", "yolo26s"),
          "train_shared_x": sum(1 for b in train_boxes if shared_x_groups(b)),
          "train_overlap": sum(1 for b in train_boxes if gt_overlap(b))}
    rep = json.loads((DS / "build_report.json").read_text())
    fx.update({k: rep.get(k, 0) for k in [f"clean_removed_{r}_{s}" for r in ("leak", "gt_overlap", "shared_x")
                                           for s in ("val", "test")]})

    # ---- sources.csv ----
    ver = {s["tag"]: s["version"] for s in SOURCES}
    proj = {s["tag"]: f"{s['workspace']}/{s['project']}" for s in SOURCES}
    with open(DS / "manifest.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    with open(OUT / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "source", "original_filename", "export_path", "source_project",
                    "source_version", "source_url", "in_clean_split", "suggested_label_fix"])
        for r in man:
            parts = Path(r["original_path"].replace("\\", "/")).parts
            export_path = "/".join(parts[parts.index(r["source"]) + 1:])  # path inside the Roboflow export
            in_clean = "" if r["split"] == "train" else int(r["file"] in clean[r["split"]])
            w.writerow([r["file"], r["split"], r["source"], parts[-1], export_path, proj[r["source"]],
                        ver[r["source"]], URL[r["source"]], in_clean, fix_action.get(r["file"], "")])

    # ---- Aircraft-C manifest (no local paths) ----
    m = json.loads((ROOT / "data_derived" / "aircraft_c" / "manifest.json").read_text(encoding="utf-8"))
    (OUT / "aircraft_c_manifest.json").write_text(json.dumps({
        "name": "Aircraft-C", "generator": "corrupt.py", "seed": m["seed"], "severities": m["severities"],
        "source_split": m["source_split"], "n_source_images": m["n_images"],
        "labels": "same merged labels as the source test image (labels/test)",
        "storage_jpeg_quality": m["storage_jpeg_quality"],
        "corruptions": m["corruptions"],
        "per_image_seed": "zlib.crc32(f'{seed}|{corruption}|{severity}|{filename}')",
        "clean_evaluation": "papers report Aircraft-C on the images in splits/test_clean.txt only",
    }, indent=2), encoding="utf-8")

    # ---- statistics for the card ----
    lbl = {s: OUT / "labels" / s for s in SPLITS}
    stats = {}
    for key, (split, fl) in lists.items():
        stats[key] = counts_per_class(lbl[split], fl, len(names))
    n = {k: len(v[1]) for k, v in lists.items()}
    nbox = {k: sum(stats[k][1].values()) for k in lists}

    write_card(names, aff, merged, train_counts, stats, n, nbox, m, fx)
    write_license()
    write_rebuild()

    bad = [p for p in OUT.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if bad:
        raise SystemExit(f"image files in the release: {bad[:3]}")
    print(f"[release] {sum(1 for _ in OUT.rglob('*') if _.is_file())} files in {OUT}; "
          f"labels train {n['train']} val {n['val_full']} test {n['test_full']}; "
          f"clean val {n['val_clean']} test {n['test_clean']}; no image files")


def write_card(names, aff, merged, train_counts, stats, n, nbox, m, fx):
    order = [i for i in range(len(names)) if names[i] not in merged]
    rows = []
    for i in order:
        c = names[i]
        cells = [stats[k][0][i] for k in ("train", "val_full", "val_clean", "test_full", "test_clean")]
        rows.append(f"| {i} | {c} | {aff.get(c, 'unknown')} | " + " | ".join(str(x) for x in cells) + " |")
    merged_rows = ", ".join(f"{c} ({train_counts[c]})" for c in merged)
    corr = ", ".join(f"{c['name']} ({c['group'].replace('_', '-')})" for c in m["corruptions"])
    text = f"""# {TITLE}

Version {VERSION}. Annotations, splits and metadata for a 36-class military-aircraft detection dataset, with
affiliation labels (allied / adversary / ambiguous / unknown) and Aircraft-C, a synthetic corruption benchmark.
**No images are included.** Images stay with their original owners and are rebuilt from the three Roboflow
sources (see [rebuild.md](rebuild.md)).

Code, training and evaluation: {REPO}

## Purpose

Research on aircraft detection and type recognition from ground-level electro-optical photographs, on robustness
to image degradation (Aircraft-C), and on *selective* affiliation decisions, where a model may abstain instead
of guessing. It accompanies the paper by Dhananjay Vaidya and Monika Mangla (see [Citation](#citation)).

## Contents

| Path | What it is |
|---|---|
| `labels/{{train,val,test}}/*.txt` | YOLO labels (`class cx cy w h`, normalised) after the class merge |
| `classes.yaml` | class ids and names (`nc: 36`); the merged classes keep their ids but have no boxes |
| `class_affiliation.json` | class -> allied / adversary / ambiguous / unknown |
| `merged_classes.txt` | classes merged into `{OTHER_CLASS}` and their training-image counts |
| `splits/train.txt`, `val_full.txt`, `test_full.txt` | every image of each split |
| `splits/val_clean.txt`, `test_clean.txt` | val/test without flipped/rotated near-copies of training images and without images whose GT boxes overlap (IoU >= 0.5) or share one x-range (use these) |
| `sources.csv` | per image: our file name, split, source, original Roboflow file name and its path in the export, project, version, URL, clean-list membership, suggested label fix (not applied) |
| `suggested_label_fixes.json` | **unverified errata, not applied**: suspected val/test label errors with a suggested correction (some coordinates are detector output) |
| `aircraft_c_manifest.json` | Aircraft-C corruptions, groups, parameters, severities and seed |
| `build_report.json` | counts from the build (duplicates removed, caps, leakage audit) |
| `leakage_stats.txt` | leakage-audit counts by transform, source and class |
| `LICENSE`, `rebuild.md` | licence; how to recreate the images |

Split lists hold paths relative to the dataset root (`images/<split>/<file>`), e.g. the `cvs_dataset/` folder that
`build_dataset.py` writes.

## Classes

36 class ids. {len(merged)} classes with fewer than {MIN_TRAIN_IMAGES} training images are merged into
`{OTHER_CLASS}`: {merged_rows}. Counts below are images containing at least one box of the class.

| id | class | affiliation | train | val (full) | val (clean) | test (full) | test (clean) |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Splits

| split | images (full) | images (clean) | boxes (full) | boxes (clean) |
|---|---|---|---|---|
| train | {n['train']} | – | {nbox['train']} | – |
| val | {n['val_full']} | {n['val_clean']} | {nbox['val_full']} | {nbox['val_clean']} |
| test | {n['test_full']} | {n['test_clean']} | {nbox['test_full']} | {nbox['test_clean']} |

Report results on the **clean** val/test lists. The full lists are kept for comparison with earlier numbers.

## Construction

1. **Sources.** Three Roboflow exports: two of our own projects of web-collected photographs
   ([aircraft-detection-rtt9x]({URL['own']}) v2, [fyp-uagdf]({URL['fyp']}) v32) and
   [merge_dataset]({URL['merge']}) v1 by SuperRadarOSINT (CC BY 4.0). Roboflow's augmented copies in our own
   export are collapsed back to one image per original.
2. **Class harmonisation.** About 250 source class names are mapped to 35 aircraft types plus `{OTHER_CLASS}`.
3. **Deduplication.** Exact duplicates by MD5; near duplicates when both the whole-image pHash and the pHash of
   the largest box crop are within Hamming distance 4 of an image already kept (priority own > fyp > merge).
   19 exact and 27,563 near duplicates were removed.
4. **Balancing and split.** Per-class caps, then a 70/20/10 train/val/test split stratified by each image's
   rarest class (seed 42).
5. **Flip/rotation-invariant leakage audit.** Each val image (against train) and test image (against train + val)
   is hashed as-is, mirrored horizontally, mirrored vertically and rotated 180 degrees. It is flagged if, for one
   variant, both the whole-image and the crop pHash are within distance 6 of one image.
6. **Clean lists.** The labels are the original exports; no label is corrected. `val_clean` / `test_clean` leave
   out, in this order: images flagged in step 5 ({fx['clean_removed_leak_val']} val, {fx['clean_removed_leak_test']} test);
   images with two GT boxes overlapping at IoU >= 0.5 ({fx['clean_removed_gt_overlap_val']} val,
   {fx['clean_removed_gt_overlap_test']} test); and images with two GT boxes sharing one x-range, i.e. both left
   and right edges within 0.002 of the image width ({fx['clean_removed_shared_x_val']} val,
   {fx['clean_removed_shared_x_test']} test). Only labels and image hashes decide; no model predictions or manual
   choices. The last two rules catch duplicate boxes and an export defect in which several boxes of an image got
   the horizontal extent of one aircraft; they also remove some correctly labelled, tightly packed or aligned
   formations.
7. **Class merge.** Classes with fewer than {MIN_TRAIN_IMAGES} training images are relabelled `{OTHER_CLASS}`
   (same ids, so the class list is stable).
8. **Aircraft-C.** Every test image is corrupted with {len(m['corruptions'])} corruptions x 3 severities,
   deterministic per image (seed {m['seed']}): {corr}. Held-out corruptions are never used for training-time
   augmentation.

## Known limitations

- **Image provenance.** Most images are photographs collected from the web (press, air shows, enthusiasts). Their
  copyright belongs to the original photographers; we do not redistribute them.
- **Domain.** Ground-level, daylight, mostly well-composed photographs. Not representative of overhead, infrared,
  long-range surveillance or operational sensor imagery.
- **Labels.** Boxes and types come from three sources with different annotators. Expect some type confusions
  (e.g. look-alike fighters) and missing boxes on secondary aircraft.
- **Label errors are not corrected.** `suggested_label_fixes.json` lists {fx['n_fix']} val/test images whose exported
  boxes look wrong, with suggested corrections. They are unverified and not applied, and they are not used in our
  evaluation; for 'relocate' suggestions the horizontal box extents come from a {fx['localiser']} detector. The
  clean lists leave the affected val/test images out by the geometry rules above. In train,
  {fx['train_shared_x']} label files have two boxes sharing one x-range and {fx['train_overlap']} have two boxes at
  IoU >= 0.5; train is used as exported.
- **Leakage audit depends on boxes.** The audit's crop hash uses the largest box, so an image whose largest box is
  misplaced can escape it. Most such images are removed by the geometry rules anyway.
- **Imbalance.** Some types have few test images even before cleaning (e.g. Z-10: 14 full, 4 clean). Per-class
  scores for these are noisy.
- **Residual leakage.** The audit catches near-identical copies, mirrors and 180-degree rotations only. Crops,
  colour edits, re-encodes beyond the hash threshold or other frames of the same event can remain.
- **Affiliation.** The allied/adversary mapping is a coarse, single-perspective convention for research. It is not
  an authoritative statement about any country, and some types are flown by both sides (marked ambiguous).
- **Aircraft-C** is synthetic; real degradations differ.

## Intended use

Research, education and **decision support with a human in the loop**: e.g. studying robustness and calibrated
abstention. Do not use this dataset or models trained on it for autonomous targeting or engagement decisions, or
for surveillance of individuals. Model outputs, including affiliation, must be reviewed by a qualified person.

## Licence

Annotations, splits and metadata: CC BY 4.0, (c) 2026 Dhananjay Vaidya and Monika Mangla. Images are not included
and remain the property of their owners. Labels derived from merge_dataset (CC BY 4.0, SuperRadarOSINT) are
redistributed with attribution. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{{vaidya2026aerovision,
  title        = {{{TITLE}}},
  author       = {{Vaidya, Dhananjay and Mangla, Monika}},
  year         = {{2026}},
  version      = {{{VERSION}}},
  howpublished = {{\\url{{{REPO}}}}},
  note         = {{Annotations and splits under CC BY 4.0; images not included}}
}}
```

Please also credit merge_dataset by SuperRadarOSINT ({URL['merge']}, CC BY 4.0).

## Takedown and contact

If you own an image referenced here and want it removed from the source lists, or you find a problem with a label,
email **dhananjayvaidya4154@gmail.com** with the file name from `sources.csv`. We will remove the reference in the
next release.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def write_license():
    (OUT / "LICENSE").write_text(f"""{TITLE}
Copyright (c) 2026 Dhananjay Vaidya and Monika Mangla

ANNOTATIONS, SPLITS AND METADATA
The files in this directory (YOLO label files, class lists, affiliation mapping, split lists, sources.csv, the
Aircraft-C manifest, reports and documentation) are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0).
  Summary:    https://creativecommons.org/licenses/by/4.0/
  Legal code: https://creativecommons.org/licenses/by/4.0/legalcode
You may share and adapt them for any purpose, provided you give appropriate credit, link to the licence and
indicate if changes were made.

IMAGES ARE NOT INCLUDED
No image is distributed with this release. Every image remains the property of its original owner and is subject
to that owner's terms. sources.csv only names the Roboflow project and file each label refers to. Obtaining the
images (see rebuild.md) is your responsibility and is governed by the terms of each source.

THIRD-PARTY CONTENT: merge_dataset
Labels for images whose source is "merge" are derived from merge_dataset by SuperRadarOSINT,
{URL['merge']}, licensed under CC BY 4.0.
Changes made: class names mapped to our class list, boxes of unmapped classes dropped, duplicates removed,
re-split, and rare classes merged into other_aircraft. This content is redistributed only as labels, with this
attribution.

The code in the parent repository is licensed separately (see ../LICENSE).
""", encoding="utf-8")


def write_rebuild():
    (OUT / "rebuild.md").write_text(f"""# Rebuilding the images

This release contains labels only. To recreate the image folders with the same file names:

1. **Get the code and a Roboflow API key.**
   ```bash
   git clone {REPO}
   cd aerovision-aircraft-detection
   python -m venv venv && . venv/bin/activate     # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt roboflow imagehash pillow
   export ROBOFLOW_API_KEY=...                     # from your Roboflow account; keep it in a gitignored .env
   ```
2. **Download the three sources and build the dataset.** `--download` fetches the YOLO exports of
   - {URL['own']} (version 2)
   - {URL['fyp']} (version 32)
   - {URL['merge']} (version 1, CC BY 4.0, SuperRadarOSINT)

   into `raw/`, then dedupes, balances, splits (seed 42) and writes `cvs_dataset/`:
   ```bash
   python build_dataset.py --download
   ```
   The build then runs the flip/rotation-invariant leakage check and the label-geometry rules and writes
   `cvs_dataset/val_clean.txt`, `test_clean.txt` and `data_clean.yaml`. No label is corrected.
3. **Run the leakage audit** (reports, and the same clean lists):
   ```bash
   python audit_leakage.py
   ```
4. **Check the result against this release.** Every row of `sources.csv` must map our file name to the same
   original Roboflow file:
   ```python
   import csv
   rel = {{r["file"]: r["original_filename"] for r in csv.DictReader(open("dataset_release/sources.csv"))}}
   got = {{r["file"]: r["original_path"].replace("\\\\", "/").rsplit("/", 1)[-1]
          for r in csv.DictReader(open("cvs_dataset/manifest.csv"))}}
   print("identical" if rel == got else f"{{sum(rel.get(k) != v for k, v in got.items())}} files differ")
   ```
   If the build differs (for example because a source export changed), use `sources.csv` instead: copy
   `raw/<source>/<export_path>` to `images/<split>/<file>` for every row and use this release's labels.
5. **Optional: the merged view and Aircraft-C.**
   ```bash
   python corrupt.py        # Aircraft-C from the test split (seed {json.loads((ROOT / 'data_derived' / 'aircraft_c' / 'manifest.json').read_text())['seed']}), see aircraft_c_manifest.json
   ```
   Training and evaluation scripts build the merged-label view (`data_derived/merged/`) themselves.

Notes
- Source projects can change or become private; the versions above are the ones used. If a version is no longer
  downloadable, contact us (see README.md).
- `build_dataset.py` caches image hashes in `raw/hash_cache.tsv`; delete it to recompute from scratch.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
