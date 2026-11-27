"""Leakage audit: flipped / rotated near-copies of training images in val and test.

  python audit_leakage.py

The check itself is build_dataset.leakage_check (the same code runs as the last step of build_dataset.py):
  val  image vs every train image
  test image vs every train AND val image
Each val/test image is hashed as-is, hflip, vflip and rot180 (64-bit pHash of the whole image and of the largest
labelled box +10%, decoded as in build_dataset.py). It is flagged when, for some variant and some reference
image, BOTH distances are <= --dist (6). Train hashes come from raw/hash_cache.tsv.
(An earlier "crop contained in whole image" criterion was dropped: on inspection it was mostly false positives.)

Clean lists = R1 (this leakage check) + R2 (two GT boxes at IoU >= 0.5) + R3 (two GT boxes sharing one x-range),
all label-only and model-independent; see build_dataset.write_clean_splits (shared with build_dataset.py).

Writes
  cvs_dataset/val_clean.txt, test_clean.txt   absolute image paths, R1-R3 images left out (no file is moved)
  data_derived/aircraft_c/clean_test_files.txt  Aircraft-C filename filter = the clean test images (no regeneration)
  paper_assets/numbers.json                   every dataset count the paper cites
  cvs_dataset/data_clean.yaml                 train as data.yaml; val/test -> the lists
  cvs_dataset/build_report.json               + leakage_* counts
  results/leakage_audit.csv                   one row per val/test image
  results/leakage_audit_per_class.csv         per class (after the other_aircraft merge)
  results/leakage_pairs.jpg                   contact sheet of flagged pairs (flagged | matched)
  paper_assets/leakage_stats.txt              counts for the paper
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_dataset import GT_OVERLAP_IOU, LEAK_VARIANTS, SHARED_X_TOL, leakage_check, update_build_report, \
    write_clean_splits
from common import ROOT, class_merge_map, load_names

FOCUS = ["HAL-Rudra", "HAL-Dhruv", "HAL-Chetak", "HAL-Prachand", "HAL-LUH", "C-130J", "An-32", "Tejas"]
CHECK_PAIR = ("fyp_test_000115.jpg", "own_train_001134.jpg")  # C-17: test = train rotated 180 deg + stretched


def contact_sheet(pairs, data: Path, path: Path, th: int = 220, cols: int = 2):
    font = ImageFont.load_default()
    cell_w, cell_h = 2 * th + 10, th + 34
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols - 1) * 16, rows * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(pairs):
        x0, y0 = (k % cols) * (cell_w + 16), (k // cols) * cell_h
        for j, p in enumerate((data / "images" / r["split"] / r["file"],
                               data / "images" / r["match_split"] / r["match_file"])):
            sheet.paste(ImageOps.pad(Image.open(p).convert("RGB"), (th, th), color=(235, 235, 235)),
                        (x0 + j * (th + 10), y0 + 30))
        d.text((x0, y0 + 2), f"{r['split'].upper()} {r['file']}  [{r['classes']}]", fill="black", font=font)
        d.text((x0, y0 + 15), f"{r['match_split'].upper()} {r['match_file']}  {r['variant']}  "
                              f"d_full={r['d_full']} d_crop={r['d_crop']}", fill=(160, 0, 0), font=font)
    sheet.save(path, quality=90)


RULES = {"R1": "flagged", "R2": "gt_overlap", "R3": "shared_x"}
RULE_KEY = {"leak": "R1", "gt_overlap": "R2", "shared_x": "R3"}


def paper_numbers(rows, counts, per, names, id_map, data: Path, dist: int) -> dict:
    """Every dataset count the paper cites (paper_assets/numbers.json)."""
    out = {"source": "audit_leakage.py on the original exported labels; clean-list rules R1-R3 "
                     "(build_dataset.write_clean_splits)",
           "rules": {"R1": f"flip/rotation-invariant near copy: for some variant of none/hflip/vflip/rot180, whole-image "
                           f"AND largest-box-crop pHash within Hamming {dist} of one train image (val) or one "
                           f"train/val image (test)",
                     "R2": f"two GT boxes with IoU >= {GT_OVERLAP_IOU}",
                     "R3": f"two GT boxes sharing one x-range: |dx0| <= {SHARED_X_TOL} and |dx1| <= {SHARED_X_TOL} "
                           f"(normalised width)",
                     "order": "removed counts attribute each image to the first rule it meets (R1, R2, R3)"},
           "clean_lists": {}, "r1_by_transform": {}, "test_images": {}, "per_class_clean": {}}
    for split in ("val", "test"):
        sr = [r for r in rows if r["split"] == split]
        hit = {k: {r["file"] for r in sr if r[f]} for k, f in RULES.items()}
        out["clean_lists"][split] = {
            "images": len(sr), "clean": counts[split][2],
            "removed": {RULE_KEY[k]: sum(1 for r in sr if r["removed_by"] == k) for k in RULE_KEY},
            "meets": {k: len(v) for k, v in hit.items()},
            "overlap": {"R1&R2": len(hit["R1"] & hit["R2"]), "R1&R3": len(hit["R1"] & hit["R3"]),
                        "R2&R3": len(hit["R2"] & hit["R3"]), "R1&R2&R3": len(hit["R1"] & hit["R2"] & hit["R3"]),
                        "any": len(hit["R1"] | hit["R2"] | hit["R3"])},
        }
        out["r1_by_transform"][split] = {v: sum(1 for r in sr if r["flagged"] and r["variant"] == v)
                                         for v in LEAK_VARIANTS}
    out["r1_by_transform"]["total"] = {v: sum(out["r1_by_transform"][s][v] for s in ("val", "test"))
                                       for v in LEAK_VARIANTS}
    out["test_images"] = {"full": out["clean_lists"]["test"]["images"], "clean": out["clean_lists"]["test"]["clean"]}
    out["val_images"] = {"full": out["clean_lists"]["val"]["images"], "clean": out["clean_lists"]["val"]["clean"]}
    out["per_class_clean"] = {e["class"]: {"val_full": e["val_images"], "val_clean": e["val_clean"],
                                           "test_full": e["test_images"], "test_clean": e["test_clean"]}
                              for e in per.values() if e["val_images"] or e["test_images"]}
    out["per_class_note"] = "images containing at least one box of the class, after the class merge"
    return out


def write_aircraft_c_filter(rows, corrupt_root: Path):
    """Aircraft-C is restricted to the clean test images by file name (images are not regenerated). The filter is
    written next to Aircraft-C; evaluate.py / abstain.py check it against test_clean.txt."""
    if not corrupt_root.is_dir():
        print(f"[audit] {corrupt_root} not found; Aircraft-C filter not written")
        return
    keep = sorted(Path(r["file"]).stem for r in rows if r["split"] == "test" and r["in_clean"])
    (corrupt_root / "clean_test_files.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"[audit] wrote {corrupt_root / 'clean_test_files.txt'} ({len(keep)} test images)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset"))
    ap.add_argument("--corrupt-root", default=str(ROOT / "data_derived" / "aircraft_c"))
    ap.add_argument("--cache", default=str(ROOT / "raw" / "hash_cache.tsv"))
    ap.add_argument("--dist", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--paper-assets", default=str(ROOT / "paper_assets"))
    args = ap.parse_args(argv)

    data = Path(args.data)
    res, pa = Path(args.results), Path(args.paper_assets)
    res.mkdir(parents=True, exist_ok=True)
    pa.mkdir(parents=True, exist_ok=True)
    names = load_names(data / "data.yaml")
    id_map, merged, _ = class_merge_map(data / "data.yaml")

    rows = leakage_check(data, args.dist, Path(args.cache))
    stats = Counter()
    counts = write_clean_splits(data, rows, stats, args.dist)
    update_build_report(data, stats)

    for r in rows:  # gt_overlap / shared_x / removed_by / in_clean were set by write_clean_splits
        r["merged_ids"] = sorted({id_map[c] for c in r["class_ids"]})
        r["classes"] = ";".join(names[c] for c in r["merged_ids"])
    flagged = [r for r in rows if r["flagged"]]

    # ---- the C-17 pair named in the review ----
    c17 = next((r for r in rows if r["file"] == CHECK_PAIR[0]), None)
    c17_note = (f"{CHECK_PAIR[0]}: flagged={c17['flagged']} variant={c17['variant']} match={c17['match_file']} "
                f"d_full={c17['d_full']} d_crop={c17['d_crop']}") if c17 else f"{CHECK_PAIR[0]} not found"
    if c17 and not c17["flagged"]:
        raise SystemExit(f"{CHECK_PAIR[0]} must be dropped (same photo as {CHECK_PAIR[1]}) but was not flagged")

    # ---- results/leakage_audit.csv ----
    fields = ["split", "file", "source", "classes", "flagged", "variant", "match_file", "match_split",
              "d_full", "d_crop", "n_matched", "variants_matched", "gt_overlap", "shared_x", "removed_by", "in_clean"]
    with open(res / "leakage_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- per class (merged) ----
    per = {}
    for c in sorted({id_map[i] for i in range(len(names))}, key=lambda c: names[c]):
        e = {"class": names[c]}
        for split in ("val", "test"):
            sr = [r for r in rows if r["split"] == split and c in r["merged_ids"]]
            fl = sum(r["flagged"] for r in sr)
            e.update({f"{split}_images": len(sr), f"{split}_flagged": fl,
                      f"{split}_clean": sum(r["in_clean"] for r in sr),
                      f"{split}_flagged_pct": round(100 * fl / len(sr), 1) if sr else 0.0})
        per[c] = e
    with open(res / "leakage_audit_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(per.values()))))
        w.writeheader()
        w.writerows(per.values())

    # ---- paper_assets/leakage_stats.txt ----
    L = [f"Leakage audit (audit_leakage.py; same check as the last step of build_dataset.py)",
         f"Rule: a val image (vs all train) or test image (vs all train + val) is flagged if, for one of its",
         f"variants none/hflip/vflip/rot180, both the whole-image pHash and the largest-box-crop pHash are within",
         f"Hamming distance {args.dist} of the same reference image. Flagged images are excluded from",
         f"val_clean.txt / test_clean.txt; nothing is moved or deleted.",
         f"Classes are after the merge rule (< 50 training images -> other_aircraft: {', '.join(merged)}).", ""]
    nums = paper_numbers(rows, counts, per, names, id_map, data, args.dist)
    (pa / "numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
    write_aircraft_c_filter(rows, Path(args.corrupt_root))

    L.append("== Clean lists (R1-R3; label-only, model-independent) ==")
    L.append("Labels are the original exported labels (no corrections applied). An image is in val_clean.txt /")
    L.append("test_clean.txt unless")
    L.append(f"  R1 leakage:        the flip/rotation leakage check above flags it;")
    L.append(f"  R2 overlapping GT: two of its GT boxes overlap at IoU >= {GT_OVERLAP_IOU};")
    L.append(f"  R3 stacked GT:     two of its GT boxes share one x-range, |dx0| <= {SHARED_X_TOL} and |dx1| <= "
             f"{SHARED_X_TOL}")
    L.append("                     (normalised image width; build_dataset.shared_x_groups).")
    L.append("No model predictions and no manual decisions are used. 'removed' = by that rule and not an earlier one.")
    L.append(f"{'split':<6}{'images':>8}{'R1 removed':>12}{'R2 removed':>12}{'R3 removed':>12}{'clean':>8}")
    for split in ("val", "test"):
        e = nums["clean_lists"][split]
        L.append(f"{split:<6}{e['images']:>8}{e['removed']['R1']:>12}{e['removed']['R2']:>12}"
                 f"{e['removed']['R3']:>12}{e['clean']:>8}")
    L.append("")
    L.append("Images meeting each rule regardless of the others, and overlaps between rules:")
    L.append(f"{'split':<6}{'R1':>6}{'R2':>6}{'R3':>6}{'R1&R2':>8}{'R1&R3':>8}{'R2&R3':>8}{'all 3':>8}{'any':>6}")
    for split in ("val", "test"):
        m, o = nums["clean_lists"][split]["meets"], nums["clean_lists"][split]["overlap"]
        L.append(f"{split:<6}{m['R1']:>6}{m['R2']:>6}{m['R3']:>6}{o['R1&R2']:>8}{o['R1&R3']:>8}{o['R2&R3']:>8}"
                 f"{o['R1&R2&R3']:>8}{o['any']:>6}")
    L.append("")
    L.append("== Flagged (leakage) ==")
    for split in ("val", "test"):
        n, fl, kept = counts[split]
        L.append(f"{split:<5} flagged {fl:>5} / {n:<5} ({100 * fl / n:5.1f}%)")
    n_all = sum(c[0] for c in counts.values())
    L.append(f"total flagged {len(flagged)} / {n_all} ({100 * len(flagged) / n_all:.1f}%)")
    L.append("")
    L.append("== By transform of the closest match (none = untransformed copy) ==")
    L.append(f"{'transform':<10}{'val':>7}{'test':>7}{'total':>7}")
    for v in LEAK_VARIANTS:
        a = [sum(1 for r in flagged if r["split"] == s and r["variant"] == v) for s in ("val", "test")]
        L.append(f"{v:<10}{a[0]:>7}{a[1]:>7}{sum(a):>7}")
    L.append("")
    L.append("== By source of the flagged image ==")
    srcs = sorted({r["source"] for r in rows})
    L.append(f"{'source':<10}{'val':>12}{'test':>12}")
    for s in srcs:
        cells = []
        for split in ("val", "test"):
            sr = [r for r in rows if r["split"] == split and r["source"] == s]
            fl = sum(r["flagged"] for r in sr)
            cells.append(f"{fl}/{len(sr)}")
        L.append(f"{s:<10}{cells[0]:>12}{cells[1]:>12}")
    L.append("")
    L.append("== Source pair (flagged image source -> matched image source) ==")
    with open(data / "manifest.csv", encoding="utf-8") as f:
        src_of = {r["file"]: r["source"] for r in csv.DictReader(f)}
    pair_counts = Counter((r["split"], r["source"], src_of.get(r["match_file"], "?"), r["match_split"])
                          for r in flagged)
    for (split, s, ms, msplit), k in sorted(pair_counts.items()):
        L.append(f"{split:<5} {s:>6} -> {ms:<6} ({msplit:<5}) {k:>5}")
    L.append("")
    L.append("== Per class (after merge): images containing the class ==")
    L.append(f"{'class':<18}{'val':>6}{'flag':>6}{'clean':>7}{'test':>7}{'flag':>6}{'clean':>7}{'test flag %':>13}")
    for e in per.values():
        L.append(f"{e['class']:<18}{e['val_images']:>6}{e['val_flagged']:>6}{e['val_clean']:>7}"
                 f"{e['test_images']:>7}{e['test_flagged']:>6}{e['test_clean']:>7}{e['test_flagged_pct']:>13.1f}")
    L.append(f"{'all images':<18}{counts['val'][0]:>6}{counts['val'][1]:>6}{counts['val'][2]:>7}"
             f"{counts['test'][0]:>7}{counts['test'][1]:>6}{counts['test'][2]:>7}"
             f"{100 * counts['test'][1] / counts['test'][0]:>13.1f}")
    L.append("")
    L.append("== Checked by eye ==")
    L.append(c17_note)
    L.append("  the train image is the test photo rotated 180 deg and stretched to 640x640 (both have one")
    L.append("  full-frame C-17 box, so crop hash = whole hash): a genuine copy; dropped.")
    (pa / "leakage_stats.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---- contact sheet: half test, half val; in each, closest-to-threshold first, then exact copies ----
    chosen = []
    for split in ("test", "val"):
        fs = sorted((r for r in flagged if r["split"] == split), key=lambda r: r["d_full"] + r["d_crop"])
        k = args.pairs // 2
        border, exact = fs[::-1][: k - k // 3], fs[: k // 3]
        chosen += border + [r for r in exact if r not in border]
    if chosen:
        contact_sheet(chosen[: args.pairs], data, res / "leakage_pairs.jpg")
    print(f"\n[audit] wrote results/leakage_audit.csv, leakage_audit_per_class.csv, leakage_pairs.jpg, "
          f"paper_assets/leakage_stats.txt, {data.name}/val_clean.txt, test_clean.txt, data_clean.yaml")


if __name__ == "__main__":
    main()
