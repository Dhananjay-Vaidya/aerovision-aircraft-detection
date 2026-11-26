"""Paper figures from results/ (vector PDF + 300 dpi PNG, 12 cm wide, serif) into results/figures/.

  python figures.py                                   # every evaluated model found in results/
  python figures.py --models yolov8s yolo26s yolo26s_degrade
  python figures.py --split-set full                  # *_full results (all test images) -> results/figures_full/

  robustness_vs_severity_<metric>  mAP vs severity (0 = clean), one panel per Aircraft-C corruption (held-out
                                   panels shaded) + mean over seen / held-out / all; one line per model
  risk_coverage                    selective risk vs coverage of the affiliation decision for the best model:
                                   clean test vs all of Aircraft-C, target risk r* and the operating point (tau)
  per_class_ap                     per-class AP@50 on clean test for the best model, bars coloured by affiliation

"Best model" = highest clean-test mAP@50 among the models plotted (override with --best).
Inputs: evaluate.py -> <m>.csv, <m>_per_class.csv; abstain.py -> <m>_abstain.json, <m>_riskcov.csv.
Smoke-test results (names containing "limit") are skipped unless named with --models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import CORRUPTION_GROUP, HELD_OUT_CORRUPTIONS, ROOT, SEEN_CORRUPTIONS, SPLIT_SETS  # noqa: E402

CM = 1 / 2.54
WIDTH = 12 * CM
AFF_COLOR = {"allied": "#2e7d32", "adversary": "#c62828", "ambiguous": "#e69f00", "unknown": "#757575"}
MODEL_STYLE = {"yolov8s": ("YOLOv8-s", "#1f77b4", "s"), "yolo26s": ("YOLO26-s", "#d62728", "o"),
               "yolo26s_degrade": ("YOLO26-s + degradation FT", "#2ca02c", "^")}
EXTRA_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
CORR_LABEL = {"defocus_blur": "Defocus blur", "motion_blur": "Motion blur", "fog": "Fog", "low_light": "Low light",
              "gaussian_noise": "Gaussian noise", "jpeg": "JPEG", "downscale": "Downscale", "contrast": "Contrast",
              "rain": "Rain"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def style(m: str, models=()):
    if m in MODEL_STYLE:
        return MODEL_STYLE[m]
    extra = [x for x in models if x not in MODEL_STYLE]
    return m, EXTRA_COLORS[(extra.index(m) if m in extra else 0) % len(EXTRA_COLORS)], "D"


def save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[figures.py] {out / stem}.pdf/.png")


def discover(res: Path, suffix: str) -> list[str]:
    """Model names with an evaluate.py result CSV (header starts with 'corruption') for this split set."""
    out = []
    for p in res.glob(f"*{suffix}.csv"):
        stem = p.stem[: len(p.stem) - len(suffix)] if suffix else p.stem
        if stem.endswith(("_per_class", "_riskcov", "_abstain_aircraft_c")) or "limit" in stem \
                or (not suffix and stem.endswith("_full")):
            continue
        with open(p, encoding="utf-8") as fh:
            if fh.readline().startswith("corruption,"):
                out.append(stem)
    order = list(MODEL_STYLE)
    return sorted(out, key=lambda m: (order.index(m) if m in order else len(order), m))


def load_eval(res: Path, m: str, sfx: str):
    df = pd.read_csv(res / f"{m}{sfx}.csv")
    df = df[~df["corruption"].astype(str).str.match(r"^[mr]PC")].copy()
    df["severity"] = df["severity"].astype(int)
    if "group" not in df:
        df["group"] = df["corruption"].map(CORRUPTION_GROUP).fillna("clean")
    return df


def robustness(res: Path, models, out: Path, metric: str, sfx: str):
    data = {m: load_eval(res, m, sfx) for m in models if (res / f"{m}{sfx}.csv").exists()}
    if not data:
        return
    order = list(SEEN_CORRUPTIONS) + list(HELD_OUT_CORRUPTIONS)
    present = {c for df in data.values() for c in df["corruption"] if c != "clean"}
    panels = [c for c in order if c in present] + ["mean seen", "mean held_out", "mean all"]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, 2.9 * CM * nrow + 1.0 * CM), sharex=True, sharey=True,
                             squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        held = CORRUPTION_GROUP.get(panel) == "held_out" or panel == "mean held_out"
        if held:
            ax.set_facecolor("#ececec")
        for m, df in data.items():
            label, color, marker = style(m, models)
            clean = float(df.loc[df["corruption"] == "clean", metric].iloc[0])
            if panel.startswith("mean"):
                g = panel.split()[1]
                sel = df[df["corruption"] != "clean"]
                sel = sel if g == "all" else sel[sel["group"] == g]
                cur = sel.groupby("severity")[metric].mean()
            else:
                cur = df[df["corruption"] == panel].set_index("severity")[metric].sort_index()
            ax.plot([0, *cur.index], [clean, *cur.values], marker=marker, color=color, label=label)
        if panel.startswith("mean"):
            title = {"seen": "Mean (seen)", "held_out": "Mean (held-out)", "all": "Mean (all)"}[panel.split()[1]]
        else:
            title = CORR_LABEL.get(panel, panel) + ("$^{\\dagger}$" if held else "")
        ax.set_title(title, pad=2, fontweight="bold" if panel.startswith("mean") else "normal")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(0, 1)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP@50" if metric == "map50" else "mAP@50:95")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, 0.0, "$^{\\dagger}$held-out corruption (shaded); severity 0 = clean test",
             ha="right", va="bottom", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=0.6, w_pad=0.4)
    save(fig, out, f"robustness_vs_severity_{metric}")


def risk_coverage(res: Path, m: str, out: Path, sfx: str):
    rc, ab = res / f"{m}{sfx}_riskcov.csv", res / f"{m}{sfx}_abstain.json"
    if not rc.exists():
        print(f"[figures.py] no {rc.name}; risk-coverage skipped")
        return
    df = pd.read_csv(rc)
    info = json.loads(ab.read_text()) if ab.exists() else {}
    fig, ax = plt.subplots(figsize=(WIDTH, 6.0 * CM))
    for split, label, ls, color in (("test", "Clean test", "-", "#1f77b4"),
                                    ("aircraft_c", "Aircraft-C (9 corruptions $\\times$ 3 severities)", "--",
                                     "#d62728")):
        d = df[df["split"] == split]
        if d.empty:
            continue
        ax.plot(d["coverage"], d["risk"], ls=ls, color=color, label=label)
        op = info.get(split)
        if op and op.get("n_accepted"):
            ax.plot(op["coverage"], op["selective_risk"], "o", color=color, ms=4, mfc="white", mew=1.0)
    if "target_risk" in info:
        tau = info.get("val", {}).get("threshold")
        ax.axhline(info["target_risk"], ls=":", color="black", lw=0.8,
                   label=f"Target risk $r^*$ = {info['target_risk']:.0%}" +
                         (f" ($\\tau$ = {tau:.3f} from clean val: circles)" if tau is not None else ""))
    ax.set_xlabel("Coverage (fraction of detections given an affiliation)")
    ax.set_ylabel("Selective risk")
    # curves end below 1: detections of ambiguous / unknown classes are never given an affiliation
    ax.set_xlim(0, min(1.0, np.ceil(df["coverage"].max() * 10 + 0.5) / 10))
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(style(m)[0], pad=2)
    fig.tight_layout()
    save(fig, out, "risk_coverage")


def per_class(res: Path, m: str, out: Path, sfx: str):
    p = res / f"{m}{sfx}_per_class.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("ap50", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(WIDTH, (0.32 * n + 1.4) * CM))
    y = np.arange(n)
    ax.barh(y, df["ap50"], color=[AFF_COLOR.get(a, "black") for a in df["affiliation"]], height=0.7)
    for yi, v in zip(y, df["ap50"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6)
    ax.set_yticks(y, df["class"].str.replace("_", " "), fontsize=6.5)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("AP@50 (clean test)")
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    present = [a for a in AFF_COLOR if a in set(df["affiliation"])]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AFF_COLOR[a]) for a in present],
              labels=[a.capitalize() for a in present], loc="lower right", frameon=True, framealpha=0.9)
    ax.set_title(style(m)[0], pad=2)
    fig.tight_layout()
    save(fig, out, "per_class_ap")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--best", default=None, help="model for risk-coverage / per-class (default: best clean mAP@50)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-set", choices=SPLIT_SETS, default="clean")
    args = ap.parse_args(argv)
    res = Path(args.results)
    sfx = "" if args.split_set == "clean" else "_full"
    out = Path(args.out) if args.out else res / ("figures" if args.split_set == "clean" else "figures_full")
    models = args.models or discover(res, sfx)
    if not models:
        raise SystemExit(f"no {args.split_set} results in {res}; run evaluate.py / abstain.py first")
    clean = {m: float(load_eval(res, m, sfx).query("corruption == 'clean'")["map50"].iloc[0])
             for m in models if (res / f"{m}{sfx}.csv").exists()}
    best = args.best or max(clean, key=clean.get)
    print(f"[figures.py] split set {args.split_set}: {models}; best (clean mAP@50) = {best}")
    robustness(res, models, out, "map50", sfx)
    robustness(res, models, out, "map50_95", sfx)
    risk_coverage(res, best, out, sfx)
    per_class(res, best, out, sfx)


if __name__ == "__main__":
    main()
