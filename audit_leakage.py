"""Leakage audit: is any cvs_dataset test image a (near-)copy of a train or val image?

  python audit_leakage.py                 # writes results/leakage_audit.csv, *_per_class.csv, leakage_pairs.jpg

Hashes are the same as build_dataset.py: 64-bit pHash of the whole image and of the largest labelled box
(+10% margin), on the image decoded the same way (JPEG draft to ~512 px, RGB). Train/val hashes come from
raw/hash_cache.tsv (cvs_dataset files are copy2() copies of the raw files listed in manifest.csv, so the
cache key path/size/mtime still matches); anything missing is hashed fresh.

For every test image, four variants are hashed: identity, horizontal flip, vertical flip, 180 deg rotation.
A test image is flagged when, for some variant and some train/val image:
  near_copy    whole-image distance <= T  AND  crop distance <= T
  containment  same class, and the crop hash of one image is within T of the other's whole-image hash
               (test crop vs train/val whole, or test whole vs train/val crop): one picture is a zoom
               of the other. Skipped when the cropped side's box (+margin) covers > --max-crop-frac of the
               image, because then "crop" is just the whole image and the test reduces to whole-vs-whole,
               which build_dataset.py found unreliable on plain-sky photos (counted separately).
Class of a crop = class of the largest box; "same class" = that class appears in the other image's labels.
Classes are the original 36 (no other_aircraft merge), so rare classes stay visible.

Nothing in cvs_dataset/ is modified.
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_dataset import HashCache, _largest_box_crop
from common import ROOT, load_names

VARIANTS = {
    "identity": lambda im: im,
    "hflip": lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    "rot180": lambda im: im.transpose(Image.Transpose.ROTATE_180),
}
FOCUS = ["HAL-Rudra", "HAL-Dhruv", "HAL-Chetak", "HAL-Prachand", "C-130J", "An-32", "Tejas"]


def read_boxes(label_path: Path):
    """[(class_id, 'cx cy w h'), ...] exactly as written by build_dataset.write_dataset."""
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(p[0]), " ".join(p[1:5])))
    return out


def largest(boxes):
    """(class_id, area fraction of the +10% crop) of the largest box, same choice as _largest_box_crop."""
    best, area = None, -1.0
    for c, coords in boxes:
        cx, cy, w, h = map(float, coords.split()[:4])
        if w * h > area:
            best, area = (c, cx, cy, w, h), w * h
    c, cx, cy, w, h = best
    w, h = w * 1.1, h * 1.1
    fw = min(1, cx + w / 2) - max(0, cx - w / 2)
    fh = min(1, cy + h / 2) - max(0, cy - h / 2)
    return c, max(0.0, fw) * max(0.0, fh)


def load_rgb(path: Path) -> Image.Image:
    """Same decode as build_dataset._image_hashes."""
    with Image.open(io.BytesIO(path.read_bytes())) as im:
        try:
            im.draft("RGB", (512, 512))
        except Exception:
            pass
        if im.width * im.height > 25_000_000:
            im.thumbnail((2048, 2048))
        return im.convert("RGB")


def ph(im) -> int:
    return int(str(imagehash.phash(im)), 16)


def variant_hashes(path: Path, boxes):
    """{variant: (full, crop)}; crop is taken on the original then transformed (= crop of the transformed image)."""
    im = load_rgb(path)
    crop = _largest_box_crop(im, [(c, xy) for c, xy in boxes])
    return {v: (ph(f(im)), ph(f(crop))) for v, f in VARIANTS.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset"))
    ap.add_argument("--cache", default=str(ROOT / "raw" / "hash_cache.tsv"))
    ap.add_argument("--dist", type=int, default=6)
    ap.add_argument("--max-crop-frac", type=float, default=0.8)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    data = Path(args.data)
    names = load_names(data / "data.yaml")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    with open(data / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manifest[r["file"]] = r

    # ---- train/val reference hashes (cache first) ----
    cache = HashCache(Path(args.cache)) if Path(args.cache).exists() else None
    ref = []  # dicts: file, split, full, crop, classes, crop_cls, crop_frac
    hits = fresh = 0
    for split in ("train", "val"):
        for img in sorted((data / "images" / split).iterdir()):
            boxes = read_boxes(data / "labels" / split / (img.stem + ".txt"))
            if not boxes:
                continue
            val = None
            if cache is not None and img.name in manifest:
                rawp = ROOT / manifest[img.name]["original_path"]
                if rawp.exists():
                    k = HashCache.key(Path(manifest[img.name]["original_path"]))
                    if k in cache.data and int(k[1]) == img.stat().st_size:
                        val = cache.data[k]
            if val is not None:
                full, crop = val[1], val[2]
                hits += 1
            else:
                full, crop = variant_hashes(img, boxes)["identity"]
                fresh += 1
            c, frac = largest(boxes)
            ref.append({"file": img.name, "split": split, "full": full, "crop": crop,
                        "classes": {b[0] for b in boxes}, "crop_cls": c, "crop_frac": frac})
    if cache is not None:
        cache.close()  # opened in append mode by HashCache; nothing written
    print(f"[audit] train/val images: {len(ref)}  (hash cache hits {hits}, hashed fresh {fresh})")

    R_full = np.array([r["full"] for r in ref], dtype=np.uint64)
    R_crop = np.array([r["crop"] for r in ref], dtype=np.uint64)
    R_cls = [r["classes"] for r in ref]
    R_ccls = np.array([r["crop_cls"] for r in ref])
    R_cfrac = np.array([r["crop_frac"] for r in ref])

    def dist(h: int, arr):
        return np.bitwise_count(arr ^ np.uint64(h)).astype(np.int16)

    # ---- test images ----
    rows, pairs, per_class_total, per_class_flag = [], [], Counter(), defaultdict(Counter)
    test_imgs = sorted((data / "images" / "test").iterdir())
    id_check = Counter()
    T = args.dist
    for n, img in enumerate(test_imgs, 1):
        boxes = read_boxes(data / "labels" / "test" / (img.stem + ".txt"))
        if not boxes:
            continue
        classes = sorted({b[0] for b in boxes})
        for c in classes:
            per_class_total[c] += 1
        vh = variant_hashes(img, boxes)
        if cache is not None and img.name in manifest:  # sanity: fresh identity hash == cached hash
            k = HashCache.key(Path(manifest[img.name]["original_path"])) if (ROOT / manifest[img.name]["original_path"]).exists() else None
            if k and k in cache.data:
                id_check["match" if cache.data[k][1:] == vh["identity"] else "mismatch"] += 1
        t_ccls, t_cfrac = largest(boxes)
        t_ccls_in = np.array([t_ccls in s for s in R_cls])          # test crop class present in ref image
        r_ccls_in = np.isin(R_ccls, classes)                         # ref crop class present in test image

        matches = []  # (criterion, variant, ref idx, d_a, d_b)
        for v, (full, crop) in vh.items():
            d_full, d_crop = dist(full, R_full), dist(crop, R_crop)
            for i in np.flatnonzero((d_full <= T) & (d_crop <= T)):
                matches.append(("near_copy", v, i, int(d_full[i]), int(d_crop[i])))
            # test crop inside ref whole image
            d_tc_rf = dist(crop, R_full)
            m = (d_tc_rf <= T) & t_ccls_in
            for i in np.flatnonzero(m):
                kind = "containment" if t_cfrac <= args.max_crop_frac else "containment_degenerate"
                matches.append((kind + ":test_crop_in_ref", v, i, int(d_tc_rf[i]), int(d_crop[i])))
            # ref crop inside test whole image
            d_tf_rc = dist(full, R_crop)
            m = (d_tf_rc <= T) & r_ccls_in
            for i in np.flatnonzero(m):
                kind = "containment" if R_cfrac[i] <= args.max_crop_frac else "containment_degenerate"
                matches.append((kind + ":ref_crop_in_test", v, i, int(d_tf_rc[i]), int(d_full[i])))

        real = [m for m in matches if not m[0].startswith("containment_degenerate")]
        degenerate_only = bool(matches) and not real
        best = min(real, key=lambda m: (m[0] != "near_copy", m[3] + m[4])) if real else None
        row = {"test_file": img.name, "source": manifest.get(img.name, {}).get("source", ""),
               "classes": ";".join(names[c] for c in classes), "flagged": int(best is not None),
               "n_matched_ref": len({m[2] for m in real}),
               "n_near_copy": len({m[2] for m in real if m[0] == "near_copy"}),
               "n_containment": len({m[2] for m in real if m[0].startswith("containment")}),
               "degenerate_containment_only": int(degenerate_only)}
        if best is not None:
            r = ref[best[2]]
            row.update({"criterion": best[0], "variant": best[1], "match_file": r["file"],
                        "match_split": r["split"],
                        "match_source": manifest.get(r["file"], {}).get("source", ""),
                        "match_classes": ";".join(names[c] for c in sorted(r["classes"])),
                        "dist_a": best[3], "dist_b": best[4],
                        "variants_matched": ";".join(sorted({m[1] for m in real}))})
            for c in classes:
                per_class_flag[c]["flagged"] += 1
                per_class_flag[c][best[0].split(":")[0]] += 1
            pairs.append((best[3] + best[4], img, data / "images" / r["split"] / r["file"], row))
        if degenerate_only:
            for c in classes:
                per_class_flag[c]["degenerate_only"] += 1
        rows.append(row)
        if n % 200 == 0:
            print(f"[audit] {n}/{len(test_imgs)} test images, flagged so far {sum(r['flagged'] for r in rows)}")

    # ---- outputs ----
    fields = ["test_file", "source", "classes", "flagged", "criterion", "variant", "match_file", "match_split",
              "match_source", "match_classes", "dist_a", "dist_b", "variants_matched", "n_matched_ref",
              "n_near_copy", "n_containment", "degenerate_containment_only"]
    with open(out_dir / "leakage_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    pc_rows = []
    for c in sorted(per_class_total, key=lambda c: names[c]):
        tot, fl = per_class_total[c], per_class_flag[c]
        pc_rows.append({"class": names[c], "test_images": tot, "flagged": fl["flagged"],
                        "flagged_pct": round(100 * fl["flagged"] / tot, 1), "near_copy": fl["near_copy"],
                        "containment": fl["containment"], "degenerate_containment_only": fl["degenerate_only"]})
    with open(out_dir / "leakage_audit_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0]))
        w.writeheader()
        w.writerows(pc_rows)

    flagged = [r for r in rows if r["flagged"]]
    print(f"\n[audit] threshold {T}; test images {len(rows)}; flagged {len(flagged)} "
          f"({100 * len(flagged) / len(rows):.1f}%)")
    print(f"[audit] by best criterion: {dict(Counter(r['criterion'].split(':')[0] for r in flagged))}")
    print(f"[audit] by matched split:  {dict(Counter(r['match_split'] for r in flagged))}")
    print(f"[audit] by variant:        {dict(Counter(r['variant'] for r in flagged))}")
    print(f"[audit] by test source:    {dict(Counter(r['source'] for r in flagged))}")
    print(f"[audit] degenerate-containment-only (not flagged): {sum(r['degenerate_containment_only'] for r in rows)}")
    print(f"[audit] cache identity-hash check on test images: {dict(id_check)}")
    print(f"\n{'class':<18}{'test':>6}{'flagged':>9}{'%':>7}{'near':>6}{'contain':>9}")
    for r in sorted(pc_rows, key=lambda r: (r["class"] not in FOCUS, -r["flagged_pct"])):
        mark = " *" if r["class"] in FOCUS else ""
        print(f"{r['class']:<18}{r['test_images']:>6}{r['flagged']:>9}{r['flagged_pct']:>7.1f}"
              f"{r['near_copy']:>6}{r['containment']:>9}{mark}")

    # ---- contact sheet: up to N flagged pairs, spread across classes, test | match ----
    # Hardest cases first, so the sheet tests the criterion rather than showing only exact mirrors:
    # all containment pairs, then near-copies closest to the threshold, then a few exact (d=0) ones.
    pairs.sort(key=lambda p: p[0])
    contain = [p for p in pairs if p[3]["criterion"].startswith("containment")]
    near = [p for p in pairs if p[3]["criterion"] == "near_copy"]
    n_exact = max(0, min(6, args.pairs - len(contain)))
    n_border = max(0, args.pairs - len(contain) - n_exact)
    chosen = (contain + near[::-1][:n_border] + near[:n_exact])[:args.pairs]
    if chosen:
        th, cols = 220, 2
        font = ImageFont.load_default()
        cell_w, cell_h = 2 * th + 10, th + 34
        sheet = Image.new("RGB", (cols * cell_w + (cols - 1) * 16, ((len(chosen) + cols - 1) // cols) * cell_h), "white")
        d = ImageDraw.Draw(sheet)
        for k, (_, t_img, r_img, row) in enumerate(chosen):
            x0, y0 = (k % cols) * (cell_w + 16), (k // cols) * cell_h
            for j, pth in enumerate((t_img, r_img)):
                im = ImageOps.pad(Image.open(pth).convert("RGB"), (th, th), color=(235, 235, 235))
                sheet.paste(im, (x0 + j * (th + 10), y0 + 30))
            d.text((x0, y0 + 2), f"TEST {row['test_file']}  [{row['classes']}]", fill="black", font=font)
            d.text((x0, y0 + 15), f"{row['match_split'].upper()} {row['match_file']}  {row['criterion'].split(':')[0]}"
                                  f"/{row['variant']}  d={row['dist_a']},{row['dist_b']}", fill=(160, 0, 0), font=font)
        sheet.save(out_dir / "leakage_pairs.jpg", quality=90)
    print(f"\n[audit] wrote {out_dir / 'leakage_audit.csv'}, leakage_audit_per_class.csv, "
          f"leakage_pairs.jpg ({len(chosen)} pairs)")


if __name__ == "__main__":
    main()
