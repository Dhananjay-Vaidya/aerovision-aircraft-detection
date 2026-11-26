"""Collect every paper number per model into results/summary.csv and results/summary.md.

  python summary_table.py [--models yolov8s yolo26s yolo26s_degrade]

Reads (per model m): results/m_clean_test.json, m_full_test.json (eval_test.py), m.csv (evaluate.py, clean
split), m_abstain.json + m_riskcov.csv (abstain.py, clean split), speed_trained.csv (speed.py).
Also writes results/aircraft_c_all_models.csv: mAP per model x corruption x severity (long format).
Sensitivity column: tau chosen on clean val at r* = 10% (from the saved val curve) and what it gives on clean test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from common import ROOT

LABEL = {"yolov8s": "YOLOv8-s", "yolo26s": "YOLO26-s", "yolo26s_degrade": "YOLO26-s + degradation FT"}


def at_risk(curves: pd.DataFrame, r: float):
    """tau from the val curve at target r (max coverage with risk <= r), applied to the test curve."""
    v = curves[curves.split == "val"].reset_index(drop=True)
    ok = v.index[v.risk <= r]
    if len(ok) == 0:
        return None, None, None
    tau = float(v.threshold[ok.max()])
    t = curves[(curves.split == "test") & (curves.threshold >= tau)]
    return tau, (float(t.coverage.iloc[-1]) if len(t) else 0.0), (float(t.risk.iloc[-1]) if len(t) else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["yolov8s", "yolo26s", "yolo26s_degrade"])
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)
    res = Path(args.results)
    speed = pd.read_csv(res / "speed_trained.csv") if (res / "speed_trained.csv").exists() else None

    rows, long = [], []
    for m in args.models:
        if not (res / f"{m}_clean_test.json").exists():
            print(f"[summary] {m}: no results, skipped")
            continue
        cj = json.loads((res / f"{m}_clean_test.json").read_text())
        fj = json.loads((res / f"{m}_full_test.json").read_text()) if (res / f"{m}_full_test.json").exists() else {}
        r = {"model": LABEL.get(m, m), "params_M": cj.get("params_M"), "gflops": cj.get("gflops"),
             "clean_test_images": cj.get("n_test_images"),
             "full_mAP50": fj.get("mAP50"), "full_mAP50_95": fj.get("mAP50_95"),
             "clean_mAP50": cj["mAP50"], "clean_mAP50_95": cj["mAP50_95"],
             "clean_precision": cj["precision"], "clean_recall": cj["recall"]}
        ev = res / f"{m}.csv"
        if ev.exists():
            df = pd.read_csv(ev)
            s = df.set_index("corruption")
            for g in ("seen", "held_out", "all"):
                for k in ("map50", "map50_95"):
                    r[f"mPC_{g}_{k}"] = float(s.loc[f"mPC_{g}", k])
                    r[f"rPC_{g}_{k}"] = float(s.loc[f"rPC_{g}", k])
            r["aircraft_c_clean_map50"] = float(s.loc["clean", "map50"])
            for _, x in df[~df.corruption.str.match(r"^[mr]PC")].iterrows():
                long.append({"model": m, "corruption": x.corruption, "severity": x.severity, "group": x.group,
                             "map50": x.map50, "map50_95": x.map50_95, "n_images": x.n_images})
        ab = res / f"{m}_abstain.json"
        if ab.exists():
            a = json.loads(ab.read_text())
            r["tau_r5"] = a["val"]["threshold"]
            for split in ("val", "test", "aircraft_c", "seen", "held_out"):
                if split in a:
                    for k in ("coverage", "selective_risk", "affiliation_accuracy", "aurc"):
                        r[f"{split}_{k}"] = a[split][k]
            curves = pd.read_csv(res / f"{m}_riskcov.csv")
            r["tau_r10"], r["test_coverage_r10"], r["test_risk_r10"] = at_risk(curves, 0.10)
        if speed is not None:
            for mode in ("forward", "end_to_end"):
                sel = speed[(speed.model == m) & (speed["mode"] == mode)]
                if len(sel):
                    r[f"ms_{mode}"] = float(sel.ms_mean.iloc[0])
                    r[f"fps_{mode}"] = float(sel.fps.iloc[0])
        rows.append(r)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(res / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    if long:
        pd.DataFrame(long).to_csv(res / "aircraft_c_all_models.csv", index=False, float_format="%.5f")

    def f(v, pct=True, nd=1):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "–"
        return f"{100 * v:.{nd}f}" if pct else f"{v:.{nd}f}"

    specs = [("Params (M)", "params_M", False, 2), ("GFLOPs", "gflops", False, 1),
             ("Full test mAP@50", "full_mAP50", True, 1), ("Clean test mAP@50", "clean_mAP50", True, 1),
             ("Clean test mAP@50:95", "clean_mAP50_95", True, 1),
             ("mPC seen (mAP@50)", "mPC_seen_map50", True, 1), ("mPC held-out (mAP@50)", "mPC_held_out_map50", True, 1),
             ("mPC all (mAP@50)", "mPC_all_map50", True, 1), ("rPC all (mAP@50)", "rPC_all_map50", True, 1),
             ("mPC all (mAP@50:95)", "mPC_all_map50_95", True, 1), ("rPC all (mAP@50:95)", "rPC_all_map50_95", True, 1),
             ("τ (clean val, r*=5%)", "tau_r5", False, 3),
             ("Coverage, clean test", "test_coverage", True, 1), ("Selective risk, clean test", "test_selective_risk", True, 1),
             ("Affiliation acc., clean test", "test_affiliation_accuracy", True, 1), ("AURC, clean test", "test_aurc", False, 3),
             ("Coverage, Aircraft-C", "aircraft_c_coverage", True, 1),
             ("Selective risk, Aircraft-C", "aircraft_c_selective_risk", True, 1),
             ("Affiliation acc., Aircraft-C", "aircraft_c_affiliation_accuracy", True, 1),
             ("AURC, Aircraft-C", "aircraft_c_aurc", False, 3),
             ("FP16 forward (ms)", "ms_forward", False, 2), ("FP16 end-to-end (ms)", "ms_end_to_end", False, 2),
             ("FPS end-to-end", "fps_end_to_end", False, 1)]
    md = ["| Metric | " + " | ".join(r["model"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for label, key, pct, nd in specs:
        md.append(f"| {label} | " + " | ".join(f(r.get(key), pct, nd) for r in rows) + " |")
    (res / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
