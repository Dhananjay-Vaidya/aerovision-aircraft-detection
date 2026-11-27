"""Evaluate detector weights on the clean test split and every Aircraft-C folder.

  python evaluate.py --weights runs/yolo26s_none/weights/best.pt
  python evaluate.py --weights runs/yolo26s_degrade/weights/best.pt --name yolo26s_degrade

--split-set clean (default) uses only the test images in cvs_dataset/test_clean.txt (no flipped/rotated near-copies
of train/val images; see audit_leakage.py), for the clean score AND for Aircraft-C (restricted by file name, not
regenerated). --split-set full uses every test image and writes results/<name>_full*.csv instead.

Labels (clean and Aircraft-C) are the merged ones (classes with < 50 training images -> other_aircraft;
common.class_merge_map), the same as train.py and abstain.py use.

Writes
  results/<name>.csv            one row per condition (clean, corruption x severity, with its group)
                                + summary rows mPC_seen, mPC_held_out, mPC_all, rPC_seen, rPC_held_out, rPC_all
  results/<name>_per_class.csv  per-class AP50 and AP50:95 on clean test (for figures.py)

mPC = mean metric over the corruptions of a group and all severities; rPC = mPC / clean (Michaelis et al., 2019).
The clean score uses exactly the images that Aircraft-C was built from, so rPC is like-for-like.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from common import CORRUPTION_GROUP, ROOT, SPLIT_SETS, check_aircraft_c_filter, default_workers, list_images, \
    load_affiliation, load_names, merged_dataset, split_images

GROUPS = ("seen", "held_out", "all")
METRICS = ("map50", "map50_95", "precision", "recall")


def _yaml_for(images: list[Path], names: list[str], tmp: Path, tag: str) -> str:
    lst = tmp / f"{tag}.txt"
    lst.write_text("\n".join(p.resolve().as_posix() for p in images) + "\n", encoding="utf-8")
    y = {"path": tmp.resolve().as_posix(), "train": lst.resolve().as_posix(), "val": lst.resolve().as_posix(),
         "names": dict(enumerate(names))}
    p = tmp / f"{tag}.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(p)


def corruption_groups(croot: Path) -> dict[str, str]:
    man = croot / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        if m.get("corruptions") and isinstance(m["corruptions"][0], dict):
            return {c["name"]: c["group"] for c in m["corruptions"]}
    return dict(CORRUPTION_GROUP)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="model name for results/<name>.csv (default: run folder name)")
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--corrupt-root", default=str(ROOT / "data_derived" / "aircraft_c"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--limit", type=int, default=None, help="first N images per condition (smoke test)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--device", default=None)
    ap.add_argument("--split-set", choices=SPLIT_SETS, default="clean",
                    help="clean: test images without train/val near-copies (default); full: all test images")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    name = args.name or (weights.parent.parent.name if weights.parent.name == "weights" else weights.stem)
    if args.split_set == "full":
        name += "_full"
    data_yaml = merged_dataset(args.data, results_dir=args.results, split_set=args.split_set)
    names = load_names(data_yaml)
    aff = load_affiliation(Path(args.data).resolve().parent / "class_affiliation.json")
    croot = Path(args.corrupt_root)
    groups = corruption_groups(croot)
    tmp = ROOT / "runs" / "_eval_lists" / name
    tmp.mkdir(parents=True, exist_ok=True)

    conditions = sorted((c.name, int(s.name), s / "images") for c in croot.iterdir() if c.is_dir()
                        for s in c.iterdir() if (s / "images").is_dir())
    if not conditions:
        raise SystemExit(f"No Aircraft-C folders under {croot}. Run corrupt.py first.")
    keep = {p.stem for p in list_images(conditions[0][2])}  # clean = the same images Aircraft-C contains
    clean = [p for p in split_images(data_yaml, "test") if p.stem in keep]
    check_aircraft_c_filter(croot, {p.stem for p in clean}, args.split_set)
    print(f"[evaluate.py] split set {args.split_set}: {len(clean)} test images (Aircraft-C restricted to these)")
    if args.limit:
        clean = clean[: args.limit]
    keep = {p.stem for p in clean}

    model = YOLO(str(weights))
    if len(model.names) != len(names):
        print(f"[evaluate.py] warning: model has {len(model.names)} classes, dataset {len(names)}")

    def run(images, tag):
        return model.val(data=_yaml_for(images, names, tmp, tag), split="val", imgsz=args.imgsz, batch=args.batch,
                         conf=args.conf, half=True, plots=False, verbose=False, workers=args.workers,
                         device=args.device, project=str(ROOT / "runs" / "eval"), name=name, exist_ok=True)

    def row(m, corruption, severity, group, n):
        return {"corruption": corruption, "severity": severity, "group": group, "map50": m.box.map50,
                "map50_95": m.box.map, "precision": m.box.mp, "recall": m.box.mr, "n_images": n}

    m = run(clean, "clean")
    rows = [row(m, "clean", 0, "clean", len(clean))]
    per_class = [{"class": names[ci], "affiliation": aff.get(names[ci], "unknown"),
                  "ap50": float(m.box.ap50[k]), "ap50_95": float(m.box.ap[k])}
                 for k, ci in enumerate(m.box.ap_class_index)]
    for c, s, img_dir in conditions:
        imgs = [p for p in list_images(img_dir) if p.stem in keep]
        m = run(imgs, f"{c}_{s}")
        rows.append(row(m, c, s, groups.get(c, "unknown"), len(imgs)))
        print(f"[evaluate.py] {c} s{s} ({groups.get(c)}): mAP50={m.box.map50:.4f} mAP50:95={m.box.map:.4f}")

    clean_row = rows[0]
    summary = {}
    for g in GROUPS:
        sel = [r for r in rows[1:] if g == "all" or r["group"] == g]
        if sel:
            summary[g] = {k: sum(r[k] for r in sel) / len(sel) for k in METRICS}
    for g, mpc in summary.items():
        rows.append({"corruption": f"mPC_{g}", "severity": "", "group": g, **mpc, "n_images": ""})
    for g, mpc in summary.items():
        rows.append({"corruption": f"rPC_{g}", "severity": "", "group": g, "n_images": "",
                     **{k: (mpc[k] / clean_row[k] if clean_row[k] > 0 else float("nan")) for k in METRICS}})

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["corruption", "severity", "group", *METRICS, "n_images"]
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(out / f"{name}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "affiliation", "ap50", "ap50_95"])
        w.writeheader()
        w.writerows(per_class)
    print(f"[evaluate.py] clean mAP50={clean_row['map50']:.4f} mAP50:95={clean_row['map50_95']:.4f}")
    for g, mpc in summary.items():
        print(f"[evaluate.py] {g:>8}: mPC(mAP50:95)={mpc['map50_95']:.4f} "
              f"rPC={mpc['map50_95'] / clean_row['map50_95'] if clean_row['map50_95'] else float('nan'):.3f}")
    print(f"[evaluate.py] wrote {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
