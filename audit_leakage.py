"""Leakage audit: flipped / rotated near-copies of training images in val and test.

  python audit_leakage.py

The check itself is build_dataset.leakage_check (the same code runs as the last step of build_dataset.py):
  val  image vs every train image
  test image vs every train AND val image
Each val/test image is hashed as-is, hflip, vflip and rot180 (64-bit pHash of the whole image and of the largest
labelled box +10%, decoded as in build_dataset.py). It is flagged when, for some variant and some reference
image, BOTH distances are <= --dist (6). Train hashes come from raw/hash_cache.tsv.
(An earlier "crop contained in whole image" criterion was dropped: on inspection it was mostly false positives.)

Writes
  cvs_dataset/val_clean.txt, test_clean.txt   absolute image paths, flagged images left out (no file is moved)
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
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_dataset import LEAK_VARIANTS, leakage_check, update_build_report, write_clean_splits
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset"))
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

    for r in rows:
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
              "d_full", "d_crop", "n_matched", "variants_matched"]
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
            e.update({f"{split}_images": len(sr), f"{split}_flagged": fl, f"{split}_clean": len(sr) - fl,
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
    L.append("== Flagged ==")
    for split in ("val", "test"):
        n, fl, kept = counts[split]
        L.append(f"{split:<5} flagged {fl:>5} / {n:<5} ({100 * fl / n:5.1f}%)   clean {kept}")
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
