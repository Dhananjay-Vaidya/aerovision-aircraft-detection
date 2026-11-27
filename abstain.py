"""Affiliation with abstention for a multiclass detector.

  python abstain.py --weights runs/yolo26s_degrade/weights/best.pt [--target-risk 0.05]

Ground truth uses the merged labels (classes with < 50 training images -> other_aircraft; common.class_merge_map).
For every detection (det conf >= --conf) the predicted class is mapped to an affiliation through
cvs_dataset/class_affiliation.json:
  * predicted class `ambiguous` or `unknown`  -> always abstain ("uncertain")
  * otherwise the detection is accepted (allied / adversary) if its confidence >= tau, else "uncertain".
tau is the lowest threshold whose selective risk on VAL is <= target (i.e. maximum coverage at that risk).

Error of an accepted detection: it matches no ground-truth box (IoU < 0.5, a false alarm), or the matched
box's true affiliation differs (a true `ambiguous`/`unknown` aircraft can never be called correctly).
  coverage = accepted / all detections        risk = errors / accepted
  affiliation accuracy = 1 - risk  (and, separately, accuracy on accepted detections that match a GT box)

--risk-on all (default): the above, over every detection. --risk-on matched: risk, coverage, tau and AURC only over
detections matched to a GT box, i.e. the affiliation risk on detected aircraft; accepted false alarms (unmatched
detections with conf >= tau) are reported separately as false_alarms_per_100_images (reported in both modes).
--risk-on all matched computes both from one prediction pass; matched-mode outputs are named <name>_matched_*.

--split-set clean (default): val and test use cvs_dataset/{val,test}_clean.txt (flipped/rotated near-copies of
train images removed; see audit_leakage.py). --split-set full: every image, outputs named <name>_full_*.

The same tau (from clean val) is then applied to Aircraft-C (data_derived/aircraft_c; only the test file names of
the chosen split set), pooled over all 9 corruptions x 3 severities ("aircraft_c"), per group (seen / held_out) and
per condition. --no-aircraft-c skips it.

Writes results/<name>_abstain.json, results/<name>_riskcov.csv (risk-coverage curves: val, test, aircraft_c) and
results/<name>_abstain_aircraft_c.csv (one row per corruption x severity).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import CORRUPTION_GROUP, ROOT, SPLIT_SETS, check_aircraft_c_filter, list_images, load_affiliation, \
    load_names, merged_dataset, merged_labels_dir, split_images

DEFINITE = ("allied", "adversary")
RISK_MODES = ("all", "matched")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def read_gt(label: Path, w: int, h: int):
    if not label.exists() or label.stat().st_size == 0:
        return np.zeros((0, 4)), np.zeros(0, int)
    a = np.loadtxt(label, ndmin=2)
    cls, xc, yc, bw, bh = a[:, 0].astype(int), a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1), cls


def collect(model, images: list[Path], labels_dir: Path, conf: float, imgsz: int, iou_thr: float = 0.5,
            details: list | None = None):
    """Returns per-detection arrays: pred class, confidence, matched GT class (-1 = false alarm).

    If `details` is a list, one dict per image is appended (path, gt_box, gt_cls, and the detections in the same
    order as the returned arrays: box, cls, conf, gt index or -1); used by diagnose_abstain.py.
    """
    P, C, G = [], [], []
    for i in range(0, len(images), 32):
        for r in model.predict([str(p) for p in images[i:i + 32]], conf=conf, imgsz=imgsz, half=True,
                               verbose=False, stream=True):
            h, w = r.orig_shape
            gt_box, gt_cls = read_gt(labels_dir / (Path(r.path).stem + ".txt"), w, h)
            box = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.cls.cpu().numpy().astype(int)
            pf = r.boxes.conf.cpu().numpy()
            order = np.argsort(-pf)
            ious = iou_matrix(box[order], gt_box)
            used = np.zeros(len(gt_box), bool)
            gidx = []
            for j, k in enumerate(order):  # greedy, highest confidence first, class-agnostic
                g = -1
                if len(gt_box):
                    cand = np.where(~used & (ious[j] >= iou_thr), ious[j], -1)
                    if cand.max() >= iou_thr:
                        g = int(cand.argmax())
                        used[g] = True
                gidx.append(g)
                P.append(pc[k]); C.append(pf[k]); G.append(gt_cls[g] if g >= 0 else -1)
            if details is not None:
                details.append({"path": Path(r.path), "gt_box": gt_box, "gt_cls": gt_cls, "box": box[order],
                                "cls": pc[order], "conf": pf[order], "gt": np.array(gidx, int)})
    return np.array(P, int), np.array(C, float), np.array(G, int)


def score_and_correct(P, C, G, names, aff):
    pa = np.array([aff.get(names[p], "unknown") for p in P])
    ga = np.array([aff.get(names[g], "unknown") if g >= 0 else "none" for g in G])
    definite = np.isin(pa, DEFINITE)
    score = np.where(definite, C, -np.inf)          # non-definite predictions are never accepted
    correct = definite & (pa == ga)
    return score, correct, G >= 0


def risk_coverage(score, correct):
    """Curve over thresholds: accept the top-k definite detections, k = 1..K."""
    n = len(score)
    order = np.argsort(-score, kind="stable")
    s, c = score[order], correct[order]
    k_max = int(np.isfinite(s).sum())
    k = np.arange(1, k_max + 1)
    errors = np.cumsum(~c[:k_max])
    return s[:k_max], k / max(n, 1), errors / k  # thresholds, coverage, risk


def select(sc, risk_on: str):
    """Detections that risk and coverage are computed over: all of them, or only those matched to a GT box."""
    score, correct, matched = sc
    if risk_on == "all":
        return score, correct, matched
    return score[matched], correct[matched], matched[matched]


def evaluate(score, correct, matched, tau, risk_on: str = "all", n_images: int | None = None):
    # accepted false alarms are counted over all detections in both modes (in "matched" mode they are outside
    # the risk / coverage and reported only here)
    fa = int(((score >= tau) & ~matched).sum())
    n_all = len(score)
    score, correct, matched = select((score, correct, matched), risk_on)
    acc = score >= tau
    n, na = len(score), int(acc.sum())
    risk = float((~correct[acc]).sum() / na) if na else 0.0
    m = acc & matched
    th, cov, rk = risk_coverage(score, correct)
    return {
        "risk_on": risk_on,
        "threshold": float(tau) if np.isfinite(tau) else None,  # None = no threshold met the target: abstain all
        "n_detections": n, "n_detections_all": n_all, "n_accepted": na,
        "coverage": na / n if n else 0.0, "selective_risk": risk, "affiliation_accuracy": 1 - risk if na else None,
        "affiliation_accuracy_matched": float(correct[m].mean()) if m.any() else None,
        "false_alarms_accepted": fa,
        "false_alarms_per_100_images": 100 * fa / n_images if n_images else None,
        "aurc": float(rk.mean()) if len(rk) else None,
        "no_abstention_accuracy": float(correct.mean()) if n else None,  # every detection forced to a label
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--affiliation", default=None, help="default: class_affiliation.json next to data.yaml")
    ap.add_argument("--target-risk", type=float, nargs="+", default=[0.05, 0.10],
                    help="target selective risk(s) r*; tau for each is chosen on (clean) val. The first one also "
                         "fills the top-level val/test/aircraft_c fields of the JSON")
    ap.add_argument("--conf", type=float, default=0.25, help="detector operating point")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--split-set", choices=SPLIT_SETS, default="clean",
                    help="clean: val/test without train near-copies (default); full: all images")
    ap.add_argument("--corrupt-root", default=str(ROOT / "data_derived" / "aircraft_c"))
    ap.add_argument("--no-aircraft-c", action="store_true")
    ap.add_argument("--risk-on", nargs="+", choices=RISK_MODES, default=["all"],
                    help="all: risk/coverage over every detection, a false alarm is an error (default); matched: only "
                         "detections matched to a GT box (affiliation risk on detected aircraft), accepted false "
                         "alarms reported separately per 100 images. Both: --risk-on all matched (one pass)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    name = args.name or (weights.parent.parent.name if weights.parent.name == "weights" else weights.stem)
    if args.split_set == "full":
        name += "_full"
    # merged labels: classes with < 50 training images -> other_aircraft (same rule as train.py / evaluate.py)
    data_yaml = merged_dataset(args.data, results_dir=args.results, split_set=args.split_set)
    names = load_names(data_yaml)
    aff = load_affiliation(args.affiliation or Path(args.data).resolve().parent / "class_affiliation.json")
    model = YOLO(str(weights))
    if len(model.names) != len(names):
        raise SystemExit(f"model has {len(model.names)} classes but data.yaml has {len(names)}")

    # ---- predictions (once; every risk mode is computed from them) ----
    data = {}  # split / condition -> (score, correct, matched), n_images
    for split in ("val", "test"):
        imgs = split_images(data_yaml, split)
        imgs = imgs[: args.limit] if args.limit else imgs
        P, C, G = collect(model, imgs, merged_labels_dir(data_yaml, split), args.conf, args.imgsz)
        data[split] = (score_and_correct(P, C, G, names, aff), len(imgs))
        if split == "test":
            test_stems = {p.stem for p in imgs}

    croot = Path(args.corrupt_root)
    conds = []  # (corruption, severity, group, sc, n_images)
    if not args.no_aircraft_c and croot.is_dir():
        if not args.limit:
            check_aircraft_c_filter(croot, test_stems, args.split_set)
        man = croot / "manifest.json"
        groups = dict(CORRUPTION_GROUP)
        if man.exists():
            groups.update({c["name"]: c["group"] for c in json.loads(man.read_text(encoding="utf-8"))["corruptions"]
                           if isinstance(c, dict)})
        for c in sorted(p for p in croot.iterdir() if p.is_dir()):
            for s in sorted((p for p in c.iterdir() if (p / "images").is_dir()), key=lambda p: int(p.name)):
                imgs = [p for p in list_images(s / "images") if p.stem in test_stems]
                P, C, G = collect(model, imgs, s / "labels", args.conf, args.imgsz)
                conds.append((c.name, int(s.name), groups.get(c.name, "unknown"),
                              score_and_correct(P, C, G, names, aff), len(imgs)))
                print(f"[abstain.py] predicted {c.name} s{s.name} ({len(imgs)} images)")

    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    fmt = lambda v, nd=3: "n/a" if v is None else f"{v:.{nd}f}"
    for mode in args.risk_on:
        mname = name if mode == "all" else f"{name}_{mode}"
        out = {"target_risks": args.target_risk, "det_conf": args.conf, "weights": str(weights),
               "split_set": args.split_set, "risk_on": mode, "by_target": {}}
        # risk-coverage curves do not depend on r*
        curves = {split: risk_coverage(*select(data[split][0], mode)[:2]) for split in ("val", "test")}
        pooled_sc = {"aircraft_c": [], "seen": [], "held_out": []}
        for cname, sev, g, sc, n_img in conds:
            for key in ("aircraft_c", g):
                if key in pooled_sc:
                    pooled_sc[key].append((sc, n_img))
        pooled = {k: ([np.concatenate([p[0][i] for p in v]) for i in range(3)], sum(p[1] for p in v), len(v))
                  for k, v in pooled_sc.items() if v}
        if "aircraft_c" in pooled:
            curves["aircraft_c"] = risk_coverage(*select(pooled["aircraft_c"][0], mode)[:2])
        th_val, _, rk_val = curves["val"]
        cond_rows = []
        for r_star in args.target_risk:
            ok = np.where(rk_val <= r_star)[0]
            tau = float(th_val[ok.max()]) if len(ok) else float("inf")   # max coverage meeting the target on val
            block = {"target_risk": r_star}
            for split in ("val", "test"):
                sc, n_img = data[split]
                block[split] = {**evaluate(*sc, tau, mode, n_img), "n_images": n_img}
            for key, (cat, n_img, n_cond) in pooled.items():
                block[key] = {**evaluate(*cat, tau, mode, n_img), "n_conditions": n_cond, "n_images": n_img}
            for cname, sev, g, sc, n_img in conds:
                r = evaluate(*sc, tau, mode, n_img)
                cond_rows.append({"target_risk": r_star, "corruption": cname, "severity": sev, "group": g,
                                  "n_images": n_img,
                                  **{k: r[k] for k in ("n_detections", "n_accepted", "coverage", "selective_risk",
                                                       "affiliation_accuracy", "false_alarms_per_100_images",
                                                       "aurc")}})
            out["by_target"][f"{r_star:.2f}"] = block
            print(f"[abstain.py] risk on {mode}, r*={r_star:.2f}: tau={tau:.4f}")
            for split in ("val", "test", "aircraft_c"):
                if split in block:
                    o = block[split]
                    print(f"[abstain.py]   {split:10s} coverage={fmt(o['coverage'])} risk={fmt(o['selective_risk'])} "
                          f"AURC={fmt(o['aurc'])} accepted FA/100 img={fmt(o['false_alarms_per_100_images'], 2)}")
        first = out["by_target"][f"{args.target_risk[0]:.2f}"]
        out["target_risk"] = args.target_risk[0]
        out.update({k: v for k, v in first.items() if k != "target_risk"})  # backward-compatible top level

        (res / f"{mname}_abstain.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        with open(res / f"{mname}_riskcov.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "threshold", "coverage", "risk"])
            for split, (th, cov, rk) in curves.items():
                w.writerows([split, f"{a:.5f}", f"{b:.5f}", f"{c:.5f}"] for a, b, c in zip(th, cov, rk))
        if cond_rows:
            with open(res / f"{mname}_abstain_aircraft_c.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(cond_rows[0]))
                w.writeheader()
                w.writerows(cond_rows)
        print(f"[abstain.py] wrote {res / (mname + '_abstain.json')}")


if __name__ == "__main__":
    main()
