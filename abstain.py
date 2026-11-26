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

--split-set clean (default): val and test use cvs_dataset/{val,test}_clean.txt (flipped/rotated near-copies of
train images removed; see audit_leakage.py). --split-set full: every image, outputs named <name>_full_*.

Writes results/<name>_abstain.json and results/<name>_riskcov.csv (risk-coverage curves, val + test).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import ROOT, SPLIT_SETS, load_affiliation, load_names, merged_dataset, merged_labels_dir, split_images

DEFINITE = ("allied", "adversary")


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


def collect(model, images: list[Path], labels_dir: Path, conf: float, imgsz: int, iou_thr: float = 0.5):
    """Returns per-detection arrays: pred class, confidence, matched GT class (-1 = false alarm)."""
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
            for j, k in enumerate(order):  # greedy, highest confidence first, class-agnostic
                g = -1
                if len(gt_box):
                    cand = np.where(~used & (ious[j] >= iou_thr), ious[j], -1)
                    if cand.max() >= iou_thr:
                        g = int(cand.argmax())
                        used[g] = True
                P.append(pc[k]); C.append(pf[k]); G.append(gt_cls[g] if g >= 0 else -1)
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


def evaluate(score, correct, matched, tau):
    acc = score >= tau
    n, na = len(score), int(acc.sum())
    risk = float((~correct[acc]).sum() / na) if na else 0.0
    m = acc & matched
    th, cov, rk = risk_coverage(score, correct)
    return {
        "threshold": float(tau) if np.isfinite(tau) else None,  # None = no threshold met the target: abstain all
        "n_detections": n, "n_accepted": na,
        "coverage": na / n if n else 0.0, "selective_risk": risk, "affiliation_accuracy": 1 - risk if na else None,
        "affiliation_accuracy_matched": float(correct[m].mean()) if m.any() else None,
        "false_alarms_accepted": int((acc & ~matched).sum()),
        "aurc": float(rk.mean()) if len(rk) else None,
        "no_abstention_accuracy": float(correct.mean()) if n else None,  # every detection forced to a label
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--affiliation", default=None, help="default: class_affiliation.json next to data.yaml")
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.25, help="detector operating point")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--split-set", choices=SPLIT_SETS, default="clean",
                    help="clean: val/test without train near-copies (default); full: all images")
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

    out, curves = {"target_risk": args.target_risk, "det_conf": args.conf, "weights": str(weights),
                   "split_set": args.split_set}, {}
    for split in ("val", "test"):
        imgs = split_images(data_yaml, split)
        imgs = imgs[: args.limit] if args.limit else imgs
        P, C, G = collect(model, imgs, merged_labels_dir(data_yaml, split), args.conf, args.imgsz)
        score, correct, matched = score_and_correct(P, C, G, names, aff)
        th, cov, rk = risk_coverage(score, correct)
        curves[split] = (th, cov, rk)
        if split == "val":
            ok = np.where(rk <= args.target_risk)[0]
            tau = float(th[ok.max()]) if len(ok) else float("inf")   # max coverage meeting the target
            out["val"] = evaluate(score, correct, matched, tau)
        else:
            out["test"] = evaluate(score, correct, matched, tau)
        out[split]["n_images"] = len(imgs)

    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{name}_abstain.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with open(res / f"{name}_riskcov.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "threshold", "coverage", "risk"])
        for split, (th, cov, rk) in curves.items():
            w.writerows([split, f"{a:.5f}", f"{b:.5f}", f"{c:.5f}"] for a, b, c in zip(th, cov, rk))
    t = out["test"]
    print(f"[abstain.py] tau={tau:.4f} (val risk {out['val']['selective_risk']:.3f}, "
          f"coverage {out['val']['coverage']:.3f})")
    print(f"[abstain.py] TEST coverage={t['coverage']:.3f} selective_risk={t['selective_risk']:.3f} "
          f"affiliation_acc={t['affiliation_accuracy']} AURC={t['aurc']}")
    print(f"[abstain.py] wrote {res / (name + '_abstain.json')}")


if __name__ == "__main__":
    main()
