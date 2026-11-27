"""Corrections to val / test labels inherited from the Roboflow exports (mostly merge_dataset).

  python label_fixes.py propose [--weights runs/.../best.pt]   # writes label_fixes/fixes.json (+ local review sheet)
  python label_fixes.py apply                                   # applies it to cvs_dataset/ (idempotent)
  python label_fixes.py document    # provenance + before/after into fixes.json; label_fixes/review_*.jpg (local)

The original labels stay available: every entry of fixes.json keeps the exported label text ("original"), and
eval_test.py --labels original scores test with it (common.original_labels_view).

What is wrong. In a number of multi-aircraft images every box of the image was exported with the SAME x0 / x1 (those
of one aircraft) while each box kept its own y0 / y1 and class: e.g. merge_val_002082 has 12 Y-20 boxes all drawn on
the leftmost of 12 aircraft. Such a "collapsed group" (>= 2 GT boxes with x0 and x1 equal within 0.002) is repaired
by matching every box one-to-one (Hungarian, cost = max(|dy0|, |dy1|) <= --y-tol) to a detection of a localiser
model (any class, conf >= 0.05, class-agnostic NMS at IoU 0.6) and taking the detection's x0 / x1. The class and
y0 / y1 stay the human annotation. If a box finds no match, or two boxes of different classes could swap
detections at almost no cost, the image is EXCLUDED from the clean val / test lists instead (never guessed).
Box x-extents of repaired boxes therefore come from the localiser (YOLO26-s by default): see the README caveat.

Besides the automatic repair, MANUAL decisions from a visual review of every val / test image with stacked GT
boxes (IoU >= 0.5 between two GT boxes):
  * duplicate box on a single aircraft -> dropped (same class)
  * merge_val_001581: Mi-26 also boxed as Il-76-78 -> Il-76-78 box dropped
  * merge_val_001579: map / photo collage, boxes on map markers -> excluded
Stacked boxes on genuinely overlapping aircraft (close formations, lines of parked aircraft) are left alone.

Train labels are not changed (the trained weights were trained on them).

label_fixes/fixes.json holds, per image, split / file / original export file name / action / reason and the
original and fixed label text, so build_dataset.py re-applies it after a rebuild (checked against the original
file name) and write_clean_splits() leaves the excluded images out of {val,test}_clean.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIX_DIR = ROOT / "label_fixes"
FIX_FILE = FIX_DIR / "fixes.json"

# manual decisions (visual review): file -> (action, label line indices to drop, reason)
MANUAL = {
    "fyp_val_001319.jpg": ("drop_duplicate", [0], "two J-20 boxes on one aircraft"),
    "fyp_val_002560.jpg": ("drop_duplicate", [1], "two other_aircraft boxes on one airliner"),
    "merge_val_000660.jpg": ("drop_duplicate", [1], "two F-16 boxes on one aircraft (nested)"),
    "fyp_test_001086.jpg": ("drop_duplicate", [0], "two other_aircraft boxes on one airliner"),
    "merge_test_000860.jpg": ("drop_duplicate", [0], "two Mi-8-17 boxes on one helicopter"),
    "merge_test_001162.jpg": ("drop_duplicate", [0], "two other_aircraft boxes on one aircraft"),
    "merge_val_001581.jpg": ("drop_conflict", [0], "Mi-26 also boxed as Il-76-78; the aircraft is a Mi-26"),
    "merge_val_001579.jpg": ("exclude", [], "map / photo collage; boxes on map markers, not aircraft"),
    # equal x0 / x1 but genuinely aligned aircraft with correct boxes (no change)
    "merge_val_000611.jpg": ("keep", [], "in-trail column of F-16s; boxes correct"),
    "merge_val_001882.jpg": ("keep", [], "two Mirage-2000 stacked vertically; boxes correct"),
    "merge_val_002275.jpg": ("keep", [], "drone in front of a radar aircraft, both full width; boxes correct"),
}


def parse(text: str):
    rows = [l.split() for l in text.splitlines() if l.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, float)
    c, xc, yc, w, h = a.T
    return c.astype(int), np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def fmt(cls, boxes) -> str:
    out = []
    for c, (x0, y0, x1, y1) in zip(cls, boxes):
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in (x0, y0, x1, y1))
        out.append(f"{int(c)} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return "\n".join(out) + "\n"


def collapsed_groups(boxes, tol=0.002):
    groups, seen = [], set()
    for i in range(len(boxes)):
        if i in seen:
            continue
        g = [j for j in range(len(boxes)) if abs(boxes[j, 0] - boxes[i, 0]) < tol and abs(boxes[j, 2] - boxes[i, 2]) < tol]
        if len(g) >= 2:
            groups.append(g)
        seen.update(g)
    return groups


def nms(boxes, conf, thr=0.6):
    from abstain import iou_matrix
    order = np.argsort(-conf)
    keep = []
    for i in order:
        if not keep or iou_matrix(boxes[[i]], boxes[keep]).max() < thr:
            keep.append(i)
    return np.array(keep, int)


def repair(cls, gt, pbox, pconf, y_tol, swap_eps=0.004, rival_conf=0.25):
    """Returns (new boxes, n_moved, problem or None)."""
    from scipy.optimize import linear_sum_assignment

    from abstain import iou_matrix
    keep = nms(pbox, pconf) if len(pbox) else np.zeros(0, int)
    pbox, pconf = pbox[keep], pconf[keep]
    new = gt.copy()
    moved = 0
    used = set()
    groups = collapsed_groups(gt)
    for g in groups:
        if len(pbox) == 0:
            return None, 0, "no detections"
        cost = np.maximum(np.abs(gt[g][:, None, 1] - pbox[None, :, 1]), np.abs(gt[g][:, None, 3] - pbox[None, :, 3]))
        big = np.where(cost <= y_tol, cost, 1e3)
        r, c = linear_sum_assignment(big)
        if len(r) < len(g) or (big[r, c] >= 1e3).any():
            return None, 0, f"{int((big[r, c] >= 1e3).sum()) + len(g) - len(r)} of {len(g)} collapsed boxes unmatched"
        a = dict(zip(r, c))
        for i in range(len(g)):        # boxes of different classes that could swap detections at ~no cost
            for k in range(i + 1, len(g)):
                if cls[g[i]] != cls[g[k]] and big[i, a[k]] < 1e3 and big[k, a[i]] < 1e3 and \
                        abs(big[i, a[k]] + big[k, a[i]] - big[i, a[i]] - big[k, a[k]]) < swap_eps:
                    return None, 0, "ambiguous class assignment"
        for i, j in a.items():
            gi = g[i]
            used.add(int(j))
            if abs(pbox[j, 0] - gt[gi, 0]) > 0.002 or abs(pbox[j, 2] - gt[gi, 2]) > 0.002:
                moved += 1
            new[gi, 0], new[gi, 2] = pbox[j, 0], pbox[j, 2]
    # a confident detection left over at the height of a repaired box is another aircraft the annotation may have
    # meant (rows of parked aircraft): which one was annotated is unknown -> not repairable
    members = [i for g in groups for i in g]
    for j in range(len(pbox)):
        if j in used or pconf[j] < rival_conf:
            continue
        if iou_matrix(pbox[[j]], new).max() >= 0.3:
            continue  # same aircraft as a box already there
        dy = np.maximum(np.abs(gt[members, 1] - pbox[j, 1]), np.abs(gt[members, 3] - pbox[j, 3]))
        if (dy <= y_tol).any():
            return None, 0, "ambiguous: more aircraft at the height of a collapsed box than boxes"
    return new, moved, None


def stacked(boxes, thr=0.5) -> bool:
    from abstain import iou_matrix
    if len(boxes) < 2:
        return False
    m = iou_matrix(boxes, boxes)
    np.fill_diagonal(m, 0)
    return bool((m >= thr).any())


def manifest(data_dir: Path):
    with open(data_dir / "manifest.csv", encoding="utf-8") as f:
        return {r["file"]: Path(r["original_path"]).name for r in csv.DictReader(f)}


def propose(args):
    from ultralytics import YOLO
    data = Path(args.data_dir)
    man = manifest(data)
    model = YOLO(args.weights)
    entries = []
    for split in ("val", "test"):
        for lbl in sorted((data / "labels" / split).glob("*.txt")):
            text = lbl.read_text()
            cls, gt = parse(text)
            img = next((data / "images" / split).glob(lbl.stem + ".*"))
            e = {"split": split, "file": img.name, "original_filename": man.get(img.name, ""), "original": text}
            if img.name in MANUAL:
                action, drop, reason = MANUAL[img.name]
                if action == "keep":
                    continue
                keep = [i for i in range(len(cls)) if i not in drop]
                e.update(action=action, reason=reason, boxes_moved=0, boxes_dropped=len(drop),
                         fixed=None if action == "exclude" else fmt(cls[keep], gt[keep]))
                entries.append(e)
                continue
            if not collapsed_groups(gt):
                continue
            r = model.predict(str(img), conf=args.conf, verbose=False)[0]
            h, w = r.orig_shape
            pb = r.boxes.xyxy.cpu().numpy() / [w, h, w, h]
            pf = r.boxes.conf.cpu().numpy()
            new, moved, problem = repair(cls, gt, pb, pf, args.y_tol)
            if problem:
                e.update(action="exclude", reason=f"collapsed x-coordinates, not repairable ({problem})",
                         boxes_moved=0, boxes_dropped=0, fixed=None)
            elif moved == 0:
                continue  # group of genuinely aligned aircraft (e.g. in-trail formation): nothing to change
            else:
                e.update(action="relocate", reason="collapsed x-coordinates; x0/x1 from the localiser, class and "
                                                   "y0/y1 from the annotation", boxes_moved=moved, boxes_dropped=0,
                         fixed=fmt(cls, new))
            entries.append(e)
            print(f"[label_fixes] {img.name}: {e['action']} ({e['reason']})")
    FIX_DIR.mkdir(exist_ok=True)
    out = {"localiser": Path(args.weights).parent.parent.name if Path(args.weights).parent.name == "weights"
           else Path(args.weights).name, "localiser_conf": args.conf, "y_tol": args.y_tol,
           "n_relocate": sum(e["action"] == "relocate" for e in entries),
           "n_drop": sum(e["action"].startswith("drop") for e in entries),
           "n_exclude": sum(e["action"] == "exclude" for e in entries),
           "boxes_moved": sum(e["boxes_moved"] for e in entries), "entries": entries}
    FIX_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[label_fixes] wrote {FIX_FILE}: {out['n_relocate']} relocated ({out['boxes_moved']} boxes), "
          f"{out['n_drop']} duplicate / conflict drops, {out['n_exclude']} excluded")
    if args.sheet:
        review_sheet(entries, data, Path(args.sheet))


def review_sheet(entries, data: Path, out: Path, tw=480, th=360, cols=5):
    """Local only (dataset photos): original boxes dashed grey, fixed boxes green."""
    from PIL import Image, ImageDraw, ImageFont
    sys_names = load_names_safe(data)
    ents = [e for e in entries if e["action"] != "exclude"] + [e for e in entries if e["action"] == "exclude"]
    rows = (len(ents) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 20)), (20, 20, 20))
    font = ImageFont.truetype("arial.ttf", 13)
    for n, e in enumerate(ents):
        im = Image.open(data / "images" / e["split"] / e["file"]).convert("RGB")
        s = min(tw / im.width, th / im.height)
        im = im.resize((int(im.width * s), int(im.height * s)))
        d = ImageDraw.Draw(im)
        W, H = im.size
        for c, b in zip(*parse(e["original"])):
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(150, 150, 150), width=1)
        if e["fixed"]:
            for c, b in zip(*parse(e["fixed"])):
                d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(0, 230, 0), width=2)
                d.text((b[0] * W + 2, b[1] * H + 1), sys_names[c], fill=(0, 255, 0), font=font)
        x, y = (n % cols) * tw, (n // cols) * (th + 20)
        sheet.paste(im, (x, y))
        ImageDraw.Draw(sheet).text((x + 3, y + th + 3), f"{e['file']} {e['action']}", fill=(255, 255, 255), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    print(f"[label_fixes] review sheet {out}")


def load_names_safe(data: Path):
    import yaml
    return yaml.safe_load((data / "data.yaml").read_text())["names"]


def load_fixes(path: Path = FIX_FILE):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}


def excluded(path: Path = FIX_FILE) -> set[str]:
    """Image file names excluded from the clean val / test lists."""
    return {e["file"] for e in load_fixes(path)["entries"] if e["action"] == "exclude"}


def apply(data_dir: Path, path: Path = FIX_FILE, log=print) -> dict:
    """Write the fixed labels into data_dir (idempotent; checked against the original export file name)."""
    data_dir = Path(data_dir)
    man = manifest(data_dir)
    n = {"applied": 0, "already": 0, "excluded": 0, "mismatch": 0}
    for e in load_fixes(path)["entries"]:
        if e["original_filename"] and man.get(e["file"]) != e["original_filename"]:
            log(f"[label_fixes] {e['file']}: original file name differs (dataset rebuilt differently?), skipped")
            n["mismatch"] += 1
            continue
        if e["action"] == "exclude":
            n["excluded"] += 1
            continue
        lbl = data_dir / "labels" / e["split"] / (Path(e["file"]).stem + ".txt")
        cur = lbl.read_text()
        if cur == e["fixed"]:
            n["already"] += 1
        elif cur == e["original"]:
            lbl.write_text(e["fixed"])
            n["applied"] += 1
        else:
            log(f"[label_fixes] {e['file']}: label is neither the original nor the fix, skipped")
            n["mismatch"] += 1
    return n


DISCOVERY = (
    "1) diagnose_abstain.py (YOLOv8-s and YOLO26-s, clean val/test, ORIGINAL labels, detections with conf >= 0.9 "
    "that abstain.py counts as errors) put merge_val_002082 (12 Y-20 boxes on one aircraft) at the top of "
    "results/abstain_errors.jpg. 2) All val/test images were then scanned for two original GT boxes overlapping at "
    "IoU >= 0.5 and reviewed by eye. 3) The review showed the export defect (boxes sharing one x-range), so every "
    "val/test image with >= 2 original GT boxes sharing x0/x1 within 0.002 was proposed for repair (collapsed-x "
    "rule), and every proposal was reviewed by eye. Manual decisions are listed in label_fixes.MANUAL.")


def _boxes_named(text, names):
    cls, b = parse(text)
    return [{"class": names[int(c)], "xyxy": [round(float(v), 5) for v in bb]} for c, bb in zip(cls, b)]


def describe_changes(e, names):
    """Per original box: unchanged / moved (x0, x1 replaced) / dropped. Boxes are matched by line order for
    relocations (the repair keeps the order) and by identity for drops."""
    before = _boxes_named(e["original"], names)
    if e["action"] == "exclude":
        return {"summary": "labels unchanged; image left out of the clean val/test lists",
                "boxes": [{**b, "change": "none"} for b in before]}
    after = _boxes_named(e["fixed"], names)
    out = []
    if e["action"] == "relocate":
        for b, a in zip(before, after):
            dx = max(abs(b["xyxy"][0] - a["xyxy"][0]), abs(b["xyxy"][2] - a["xyxy"][2]))
            change = ("unchanged" if dx < 1e-4 else
                      "x adjusted < 0.01 (box was on its aircraft; x0/x1 now from the localiser)" if dx < 0.01 else
                      "moved to another aircraft (x0/x1 from localiser; y0/y1 and class kept)")
            out.append({"class": b["class"], "before": b["xyxy"], "after": a["xyxy"], "max_dx": round(dx, 5),
                        "change": change})
    else:  # drop: the kept boxes are an ordered subsequence of the original
        k = 0
        for b in before:
            if k < len(after) and b["class"] == after[k]["class"] and \
                    all(abs(x - y) < 1e-4 for x, y in zip(b["xyxy"], after[k]["xyxy"])):
                out.append({"class": b["class"], "before": b["xyxy"], "after": after[k]["xyxy"], "change": "unchanged"})
                k += 1
            else:
                out.append({"class": b["class"], "before": b["xyxy"], "after": None, "change": "dropped"})
    n = {c: sum(1 for o in out if o["change"].startswith(c)) for c in ("moved", "x adjusted", "dropped", "unchanged")}
    return {"summary": f"{len(before)} boxes before, {len(after)} after: {n['moved']} moved to another aircraft, "
                       f"{n['x adjusted']} x adjusted < 0.01, {n['dropped']} dropped, {n['unchanged']} unchanged",
            "boxes": out}


def found_by(e, err_rows):
    """How the image was found: model errors from the diagnosis on the original labels, and the label scans."""
    ev = []
    for r in err_rows.get(e["file"], []):
        # (a) false alarms count as errors only when risk is over all detections; (b) wrong affiliation also counts
        # when risk is over matched detections
        modes = ["all"] if r["category"] in ("a1", "a2") else ["all", "matched"]
        ev.append({"how": "model error (diagnose_abstain.py, original labels, conf >= 0.9)", "model": r["model"],
                   "split": r["split"], "risk_mode": modes, "category": r["category"],
                   "pred_class": r["pred_class"], "conf": float(r["conf"]), "gt_class": r["gt_class"] or None})
    cls, b = parse(e["original"])
    if stacked(b):
        ev.append({"how": "GT-overlap scan (two original GT boxes at IoU >= 0.5)", "split": e["split"]})
    g = collapsed_groups(b)
    if g:
        ev.append({"how": "collapsed-x rule (>= 2 original GT boxes sharing x0/x1)", "split": e["split"],
                   "group_sizes": [len(x) for x in g]})
    if e["file"] in MANUAL:
        ev.append({"how": "manual decision after visual review", "note": MANUAL[e["file"]][2]})
    return ev


def document(errors_csv: Path, data_dir: Path, sheet_dir: Path, path: Path = FIX_FILE):
    """Add provenance and before/after per entry to fixes.json (label text untouched) and write one local review
    image per entry: original GT red, fixed GT green."""
    names = load_names_safe(data_dir)
    fx = load_fixes(path)
    err_rows = {}
    if errors_csv.exists():
        with open(errors_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                err_rows.setdefault(r["file"], []).append(r)
    for e in fx["entries"]:
        e["found_by"] = found_by(e, err_rows)
        e["changes"] = describe_changes(e, names)
    fx["discovery"] = DISCOVERY
    fx["model_errors_source"] = f"{errors_csv.name} (diagnose_abstain.py run on the original labels)"
    fx["reviewed_no_change"] = [{"file": f, "reason": r} for f, (a, _, r) in MANUAL.items() if a == "keep"]
    path.write_text(json.dumps(fx, indent=1), encoding="utf-8")
    n = review_images(fx["entries"], data_dir, sheet_dir, names)
    print(f"[label_fixes] documented {len(fx['entries'])} entries in {path}; {n} review images in {sheet_dir}")


def review_images(entries, data: Path, out: Path, names, width=1000):
    from PIL import Image, ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("review_*.jpg"):
        old.unlink()
    font = ImageFont.truetype("arial.ttf", 15)
    for e in entries:
        im = Image.open(data / "images" / e["split"] / e["file"]).convert("RGB")
        s = width / im.width
        im = im.resize((width, max(1, int(im.height * s))))
        W, H = im.size
        d = ImageDraw.Draw(im)
        for c, b in zip(*parse(e["original"])):
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(255, 30, 30), width=2)
            d.text((b[0] * W + 3, b[3] * H - 18), names[c], fill=(255, 60, 60), font=font)
        if e["fixed"]:
            for c, b in zip(*parse(e["fixed"])):
                d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=(0, 230, 0), width=2)
                d.text((b[0] * W + 3, b[1] * H + 2), names[c], fill=(0, 255, 0), font=font)
        how = "; ".join(sorted({f["how"].split(" (")[0] + (f" {f['model']}/{f['split']}" if "model" in f else "")
                                for f in e["found_by"]}))
        head = [f"{e['file']}  [{e['split']}]  action: {e['action']}   (red = original GT, green = fixed GT)",
                f"reason: {e['reason']}", f"change: {e['changes']['summary']}", f"found by: {how}"]
        canvas = Image.new("RGB", (W, H + 22 * len(head) + 8), (25, 25, 25))
        canvas.paste(im, (0, 22 * len(head) + 8))
        cd = ImageDraw.Draw(canvas)
        for i, line in enumerate(head):
            cd.text((6, 4 + 22 * i), line, fill=(255, 255, 255), font=font)
        tag = "excluded" if e["action"] == "exclude" else e["action"]
        canvas.save(out / f"review_{tag}_{Path(e['file']).stem}.jpg", quality=88)
    return len(entries)


def sync_aircraft_c(data_dir: Path, corrupt_root: Path, path: Path = FIX_FILE) -> int:
    """Copy the (merged) fixed test labels into every Aircraft-C condition (corruptions do not move boxes)."""
    from common import merged_dataset, merged_labels_dir
    if not corrupt_root.is_dir():
        return 0
    merged = merged_dataset(Path(data_dir) / "data.yaml", split_set="full")
    src = merged_labels_dir(merged, "test")
    n = 0
    for e in load_fixes(path)["entries"]:
        if e["split"] != "test" or e["action"] == "exclude":
            continue
        stem = Path(e["file"]).stem
        text = (src / f"{stem}.txt").read_text()
        for lbl in corrupt_root.glob(f"*/*/labels/{stem}.txt"):
            if lbl.read_text() != text:
                lbl.write_text(text)
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--weights", default=None, help="localiser; default: weights in results/yolo26s_abstain.json")
    p.add_argument("--data-dir", default=str(ROOT / "cvs_dataset"))
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--y-tol", type=float, default=0.015)
    p.add_argument("--sheet", default=str(ROOT / "results" / "label_fixes_review.jpg"))
    p.add_argument("--force", action="store_true", help="overwrite an existing label_fixes/fixes.json")
    a = sub.add_parser("apply")
    a.add_argument("--data-dir", default=str(ROOT / "cvs_dataset"))
    a.add_argument("--corrupt-root", default=str(ROOT / "data_derived" / "aircraft_c"),
                   help="Aircraft-C labels of fixed test images are updated too")
    dc = sub.add_parser("document", help="add provenance + before/after to fixes.json; local review images")
    dc.add_argument("--data-dir", default=str(ROOT / "cvs_dataset"))
    dc.add_argument("--errors-csv", default=str(ROOT / "results" / "abstain_errors.csv"),
                    help="diagnose_abstain.py output on the ORIGINAL labels")
    dc.add_argument("--out", default=str(FIX_DIR), help="review_*.jpg go here (gitignored)")
    args = ap.parse_args(argv)
    if args.cmd == "document":
        document(Path(args.errors_csv), Path(args.data_dir), Path(args.out))
        return
    if args.cmd == "propose":
        if FIX_FILE.exists() and not args.force:
            raise SystemExit(f"{FIX_FILE} exists (propose must run on the ORIGINAL labels); --force to redo")
        if args.weights is None:
            args.weights = json.loads((ROOT / "results" / "yolo26s_abstain.json").read_text())["weights"]
        propose(args)
    else:
        print(f"[label_fixes] {apply(Path(args.data_dir))}")
        n = sync_aircraft_c(Path(args.data_dir), Path(args.corrupt_root))
        print(f"[label_fixes] Aircraft-C: {n} label files updated "
              f"(run python audit_leakage.py next to rewrite the clean lists)")


if __name__ == "__main__":
    main()
