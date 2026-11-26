"""Why does abstention reach (almost) no coverage at r* = 5%? Categorise the confident errors.

  python diagnose_abstain.py                       # models with results/<m>_abstain.json (weights read from it)
  python diagnose_abstain.py --models yolov8s=path/to/best.pt yolo26s=path/to/best.pt

For each model, on clean val and clean test, every detection with conf >= --min-conf (0.9) that abstain.py counts as
an error (predicted class has a definite affiliation, allied / adversary, and it is not correct) is put in one class:
  (a)  false alarm: no GT box at IoU >= 0.5 (same greedy class-agnostic matching as abstain.py)
       (a1) the image has more confident detections (conf >= --min-conf, any class) than GT boxes: likely a
            missing label
       (a2) other
  (b)  matched a GT box whose affiliation differs (incl. GT ambiguous / unknown, which can never be right);
       reported as predicted class -> GT class pairs
Confident detections of ambiguous / unknown classes are always abstained and not errors; they are counted apart.
Extra flag (does not change a1 / a2): n_gt_stacked = GT boxes lying on another GT box (IoU >= 0.5), a sign of
misplaced labels (e.g. merge_val_002082: 12 boxes all drawn on the leftmost of 12 aircraft).

Writes results/abstain_error_diagnosis.json (counts + pairs), results/abstain_errors.csv (one row per error) and a
local contact sheet results/abstain_errors.jpg (24 highest-confidence errors over both models and splits; GT green,
predictions red; gitignored, it contains dataset photos).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from abstain import DEFINITE, collect, iou_matrix
from common import ROOT, load_affiliation, load_names, merged_dataset, merged_labels_dir, split_images

LABEL = {"yolov8s": "YOLOv8-s", "yolo26s": "YOLO26-s", "yolo26s_degrade": "YOLO26-s + degradation FT"}


def categorise(details, names, aff, min_conf):
    """Rows for every confident error, plus the number of confident detections / non-definite ones."""
    rows, n_conf, n_nondef = [], 0, 0
    for d in details:
        conf_mask = d["conf"] >= min_conf
        n_img_conf = int(conf_mask.sum())
        gi = iou_matrix(d["gt_box"], d["gt_box"])
        np.fill_diagonal(gi, 0)
        stacked = int((gi >= 0.5).any(1).sum()) if len(gi) else 0  # GT boxes lying on another GT box
        for j in np.where(conf_mask)[0]:
            n_conf += 1
            pname = names[d["cls"][j]]
            pa = aff.get(pname, "unknown")
            if pa not in DEFINITE:
                n_nondef += 1
                continue
            g = int(d["gt"][j])
            if g < 0:
                cat = "a1" if n_img_conf > len(d["gt_box"]) else "a2"
                gname, ga = "", "none"
            else:
                gname = names[d["gt_cls"][g]]
                ga = aff.get(gname, "unknown")
                if ga == pa:
                    continue  # correct (same affiliation, class may differ)
                cat = "b"
            rows.append({"file": d["path"].name, "category": cat, "pred_class": pname, "pred_affiliation": pa,
                         "conf": float(d["conf"][j]), "gt_class": gname, "gt_affiliation": ga,
                         "n_conf_dets": n_img_conf, "n_gt": len(d["gt_box"]), "n_gt_stacked": stacked,
                         "_det": d, "_j": int(j)})
    return rows, n_conf, n_nondef


def contact_sheet(errors, names, out: Path, n=24, cols=6, tw=420, th=315):
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for f in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(f, sz)
            except OSError:
                pass
        return ImageFont.load_default()

    f_box, f_cap = font(13), font(13)
    top = sorted(errors, key=lambda r: -r["conf"])[:n]
    rows = int(np.ceil(len(top) / cols))
    cap_h = 36
    sheet = Image.new("RGB", (cols * tw, rows * (th + cap_h)), (24, 24, 24))
    for i, r in enumerate(top):
        d, j = r["_det"], r["_j"]
        im = Image.open(d["path"]).convert("RGB")
        s = min(tw / im.width, th / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        dr = ImageDraw.Draw(im)
        for b, c in zip(d["gt_box"], d["gt_cls"]):
            b = [v * s for v in b]
            dr.rectangle(b, outline=(0, 220, 0), width=2)
            dr.text((b[0] + 2, b[3] - 15), names[c], fill=(0, 255, 0), font=f_box)
        for k in np.where(d["conf"] >= 0.25)[0]:  # all predictions thin, the error thick
            b = [v * s for v in d["box"][k]]
            dr.rectangle(b, outline=(255, 40, 40), width=3 if k == j else 1)
            if k == j:
                dr.text((b[0] + 2, b[1] + 1), f"{names[d['cls'][k]]} {d['conf'][k]:.3f}", fill=(255, 60, 60),
                        font=f_box)
        x, y = (i % cols) * tw, (i // cols) * (th + cap_h)
        sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        cap = ImageDraw.Draw(sheet)
        what = {"a1": "(a1) FA, more conf dets than GT", "a2": "(a2) FA, other",
                "b": f"(b) GT {r['gt_class']}"}[r["category"]]
        cap.text((x + 4, y + th + 2), f"{LABEL.get(r['model'], r['model'])} {r['split']} {r['file'][:30]}",
                 fill=(220, 220, 220), font=f_cap)
        cap.text((x + 4, y + th + 18), f"{r['pred_class']} {r['conf']:.3f} | {what}", fill=(255, 200, 120),
                 font=f_cap)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(f"[diagnose] contact sheet {out} ({len(top)} errors)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None, help="name=weights; default: from results/<m>_abstain.json")
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--min-conf", type=float, default=0.9)
    ap.add_argument("--conf", type=float, default=0.25, help="detector operating point (as abstain.py)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--sheet", default=str(ROOT / "results" / "abstain_errors.jpg"))
    args = ap.parse_args(argv)
    res = Path(args.results)

    if args.models:
        models = dict(m.split("=", 1) for m in args.models)
    else:
        models = {}
        for m in ("yolov8s", "yolo26s", "yolo26s_degrade"):
            p = res / f"{m}_abstain.json"
            if p.exists():
                models[m] = json.loads(p.read_text(encoding="utf-8"))["weights"]
    if not models:
        raise SystemExit("no models: run abstain.py first or pass --models name=weights")

    from ultralytics import YOLO

    data_yaml = merged_dataset(args.data, results_dir=args.results, split_set="clean")
    names = load_names(data_yaml)
    aff = load_affiliation(Path(args.data).resolve().parent / "class_affiliation.json")

    summary, all_errors = {"min_conf": args.min_conf, "det_conf": args.conf, "split_set": "clean", "models": {}}, []
    for m, w in models.items():
        if not Path(w).exists():
            raise SystemExit(f"weights not found for {m}: {w}")
        model = YOLO(str(w))
        summary["models"][m] = {}
        for split in ("val", "test"):
            imgs = split_images(data_yaml, split)
            details = []
            collect(model, imgs, merged_labels_dir(data_yaml, split), args.conf, args.imgsz, details=details)
            rows, n_conf, n_nondef = categorise(details, names, aff, args.min_conf)
            for r in rows:
                r["model"], r["split"] = m, split
            all_errors += rows
            cats = Counter(r["category"] for r in rows)
            pairs = Counter(f"{r['pred_class']} -> {r['gt_class']}" for r in rows if r["category"] == "b")
            gt_aff = Counter(r["gt_affiliation"] for r in rows if r["category"] == "b")
            fa_cls = Counter(r["pred_class"] for r in rows if r["category"] in ("a1", "a2"))
            summary["models"][m][split] = {
                "n_images": len(imgs), "n_conf_detections": n_conf, "n_conf_nondefinite": n_nondef,
                "n_conf_definite": n_conf - n_nondef, "n_errors": len(rows),
                "a_false_alarm": cats["a1"] + cats["a2"], "a1_more_dets_than_gt": cats["a1"], "a2_other": cats["a2"],
                "a2_in_images_with_stacked_gt": sum(r["category"] == "a2" and r["n_gt_stacked"] > 0 for r in rows),
                "a2_images": len({r["file"] for r in rows if r["category"] == "a2"}),
                "b_gt_other_aircraft": sum(r["category"] == "b" and r["gt_class"] == "other_aircraft" for r in rows),
                "b_wrong_affiliation": cats["b"], "b_by_gt_affiliation": dict(gt_aff.most_common()),
                "b_pairs": dict(pairs.most_common()), "a_by_pred_class": dict(fa_cls.most_common()),
            }
            s = summary["models"][m][split]
            print(f"[diagnose] {m} {split}: {n_conf} dets >= {args.min_conf} ({s['n_conf_definite']} definite), "
                  f"{len(rows)} errors: a1={cats['a1']} a2={cats['a2']} b={cats['b']}; top pairs "
                  f"{pairs.most_common(5)}")

    (res / "abstain_error_diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = ["model", "split", "file", "category", "pred_class", "pred_affiliation", "conf", "gt_class",
              "gt_affiliation", "n_conf_dets", "n_gt", "n_gt_stacked"]
    with open(res / "abstain_errors.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows({**r, "conf": f"{r['conf']:.4f}"} for r in sorted(all_errors, key=lambda r: -r["conf"]))
    print(f"[diagnose] wrote {res / 'abstain_error_diagnosis.json'}, {res / 'abstain_errors.csv'}")
    if all_errors:
        contact_sheet(all_errors, names, Path(args.sheet))


if __name__ == "__main__":
    main()
