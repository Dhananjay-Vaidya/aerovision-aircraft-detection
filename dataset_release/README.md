# AeroVision aircraft dataset: labels, splits and the Aircraft-C benchmark

Version 1.0.0. Annotations, splits and metadata for a 36-class military-aircraft detection dataset, with
affiliation labels (allied / adversary / ambiguous / unknown) and Aircraft-C, a synthetic corruption benchmark.
**No images are included.** Images stay with their original owners and are rebuilt from the three Roboflow
sources (see [rebuild.md](rebuild.md)).

Code, training and evaluation: https://github.com/Dhananjay-Vaidya/aerovision-aircraft-detection

## Purpose

Research on aircraft detection and type recognition from ground-level electro-optical photographs, on robustness
to image degradation (Aircraft-C), and on *selective* affiliation decisions, where a model may abstain instead
of guessing. It accompanies the paper by Dhananjay Vaidya and Monika Mangla (see [Citation](#citation)).

## Contents

| Path | What it is |
|---|---|
| `labels/{train,val,test}/*.txt` | YOLO labels (`class cx cy w h`, normalised) after the class merge |
| `classes.yaml` | class ids and names (`nc: 36`); the merged classes keep their ids but have no boxes |
| `class_affiliation.json` | class -> allied / adversary / ambiguous / unknown |
| `merged_classes.txt` | classes merged into `other_aircraft` and their training-image counts |
| `splits/train.txt`, `val_full.txt`, `test_full.txt` | every image of each split |
| `splits/val_clean.txt`, `test_clean.txt` | val/test without flipped/rotated near-copies of training images (use these) |
| `sources.csv` | per image: our file name, split, source, original Roboflow file name and its path in the export, project, version, URL |
| `aircraft_c_manifest.json` | Aircraft-C corruptions, groups, parameters, severities and seed |
| `build_report.json` | counts from the build (duplicates removed, caps, leakage audit) |
| `leakage_stats.txt` | leakage-audit counts by transform, source and class |
| `LICENSE`, `rebuild.md` | licence; how to recreate the images |

Split lists hold paths relative to the dataset root (`images/<split>/<file>`), e.g. the `cvs_dataset/` folder that
`build_dataset.py` writes.

## Classes

36 class ids. 7 classes with fewer than 50 training images are merged into
`other_aircraft`: Su-30MKI (15), Jaguar (19), J-16 (10), J-11 (15), KJ-500 (4), Mirage-III-V (13), Saab-2000-Erieye (23). Counts below are images containing at least one box of the class.

| id | class | affiliation | train | val (full) | val (clean) | test (full) | test (clean) |
|---|---|---|---|---|---|---|---|
| 1 | Rafale | allied | 326 | 93 | 92 | 46 | 43 |
| 2 | Tejas | allied | 118 | 34 | 25 | 17 | 15 |
| 3 | Mirage-2000 | allied | 233 | 67 | 58 | 33 | 27 |
| 4 | MiG-29 | allied | 207 | 59 | 52 | 29 | 23 |
| 6 | C-17 | allied | 608 | 176 | 156 | 85 | 72 |
| 7 | C-130J | allied | 145 | 42 | 37 | 21 | 16 |
| 8 | C-295 | allied | 204 | 58 | 55 | 29 | 27 |
| 9 | An-32 | allied | 124 | 35 | 29 | 18 | 15 |
| 10 | AH-64 | allied | 392 | 111 | 107 | 56 | 54 |
| 11 | CH-47 | allied | 485 | 142 | 134 | 74 | 67 |
| 12 | HAL-Prachand | allied | 219 | 62 | 60 | 31 | 30 |
| 13 | HAL-Dhruv | allied | 212 | 60 | 38 | 30 | 15 |
| 14 | HAL-Rudra | allied | 195 | 56 | 45 | 28 | 20 |
| 15 | HAL-LUH | allied | 306 | 88 | 83 | 44 | 41 |
| 16 | HAL-Chetak | allied | 189 | 54 | 41 | 27 | 17 |
| 17 | Mi-26 | allied | 262 | 74 | 43 | 37 | 14 |
| 18 | J-20 | adversary | 352 | 98 | 96 | 50 | 46 |
| 19 | J-10 | adversary | 260 | 69 | 67 | 34 | 30 |
| 22 | H-6 | adversary | 212 | 59 | 52 | 29 | 24 |
| 23 | Y-20 | adversary | 203 | 57 | 47 | 28 | 19 |
| 24 | JH-7 | adversary | 157 | 45 | 40 | 22 | 19 |
| 26 | Z-10 | adversary | 102 | 29 | 15 | 14 | 4 |
| 27 | WZ-7 | adversary | 116 | 33 | 18 | 17 | 13 |
| 28 | JF-17 | adversary | 173 | 49 | 37 | 25 | 19 |
| 29 | F-16 | adversary | 696 | 202 | 192 | 102 | 97 |
| 32 | Mi-8-17 | ambiguous | 529 | 150 | 125 | 74 | 63 |
| 33 | Il-76-78 | ambiguous | 329 | 93 | 75 | 47 | 39 |
| 34 | Mi-24-35 | ambiguous | 320 | 93 | 73 | 45 | 35 |
| 35 | other_aircraft | unknown | 2207 | 613 | 595 | 321 | 309 |

## Splits

| split | images (full) | images (clean) | boxes (full) | boxes (clean) |
|---|---|---|---|---|
| train | 9019 | – | 15503 | – |
| val | 2576 | 2284 | 4256 | 3809 |
| test | 1286 | 1098 | 2199 | 1922 |

Report results on the **clean** val/test lists. The full lists are kept for comparison with earlier numbers.

## Construction

1. **Sources.** Three Roboflow exports: two of our own projects of web-collected photographs
   ([aircraft-detection-rtt9x](https://universe.roboflow.com/dhananjay/aircraft-detection-rtt9x) v2, [fyp-uagdf](https://universe.roboflow.com/dhananjay/fyp-uagdf) v32) and
   [merge_dataset](https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd) v1 by SuperRadarOSINT (CC BY 4.0). Roboflow's augmented copies in our own
   export are collapsed back to one image per original.
2. **Class harmonisation.** About 250 source class names are mapped to 35 aircraft types plus `other_aircraft`.
3. **Deduplication.** Exact duplicates by MD5; near duplicates when both the whole-image pHash and the pHash of
   the largest box crop are within Hamming distance 4 of an image already kept (priority own > fyp > merge).
   19 exact and 27,563 near duplicates were removed.
4. **Balancing and split.** Per-class caps, then a 70/20/10 train/val/test split stratified by each image's
   rarest class (seed 42).
5. **Flip/rotation-invariant leakage audit.** Each val image (against train) and test image (against train + val)
   is hashed as-is, mirrored horizontally, mirrored vertically and rotated 180 degrees. It is flagged if, for one
   variant, both the whole-image and the crop pHash are within distance 6 of one image. 292 val and 188 test images
   were flagged and are left out of the clean lists (see `leakage_stats.txt`).
6. **Class merge.** Classes with fewer than 50 training images are relabelled `other_aircraft`
   (same ids, so the class list is stable).
7. **Aircraft-C.** Every test image is corrupted with 9 corruptions x 3 severities,
   deterministic per image (seed 1234): defocus_blur (seen), motion_blur (seen), fog (seen), low_light (seen), gaussian_noise (seen), jpeg (seen), downscale (seen), contrast (held-out), rain (held-out). Held-out corruptions are never used for training-time
   augmentation.

## Known limitations

- **Image provenance.** Most images are photographs collected from the web (press, air shows, enthusiasts). Their
  copyright belongs to the original photographers; we do not redistribute them.
- **Domain.** Ground-level, daylight, mostly well-composed photographs. Not representative of overhead, infrared,
  long-range surveillance or operational sensor imagery.
- **Labels.** Boxes and types come from three sources with different annotators. Expect some type confusions
  (e.g. look-alike fighters) and missing boxes on secondary aircraft.
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
@misc{vaidya2026aerovision,
  title        = {AeroVision aircraft dataset: labels, splits and the Aircraft-C benchmark},
  author       = {Vaidya, Dhananjay and Mangla, Monika},
  year         = {2026},
  version      = {1.0.0},
  howpublished = {\url{https://github.com/Dhananjay-Vaidya/aerovision-aircraft-detection}},
  note         = {Annotations and splits under CC BY 4.0; images not included}
}
```

Please also credit merge_dataset by SuperRadarOSINT (https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd, CC BY 4.0).

## Takedown and contact

If you own an image referenced here and want it removed from the source lists, or you find a problem with a label,
email **dhananjayvaidya4154@gmail.com** with the file name from `sources.csv`. We will remove the reference in the
next release.
