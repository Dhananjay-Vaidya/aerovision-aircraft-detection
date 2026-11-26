"""Paper figures from results/ (PNG + PDF into results/figures/).

  python figures.py                                   # every model found in results/
  python figures.py --models yolo26s_none yolo26s_degrade

  robustness_vs_severity  mAP vs severity (0 = clean), one panel per corruption + seen / held-out / all means
  risk_coverage           selective risk vs coverage on test, from abstain.py
  per_class_ap            per-class AP@50 on clean test; class labels coloured by affiliation
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

from common import CORRUPTION_GROUP, HELD_OUT_CORRUPTIONS, ROOT, SEEN_CORRUPTIONS  # noqa: E402

AFF_COLOR = {"allied": "#1a7f37", "adversary": "#cf222e", "ambiguous": "#bf8700", "unknown": "#57606a"}


def save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures.py] {out / stem}.png/.pdf")


def discover(res: Path) -> list[str]:
    skip = ("_per_class", "_riskcov")
    return sorted(p.stem for p in res.glob("*.csv") if not p.stem.endswith(skip))


def robustness(res: Path, models, out: Path, metric: str):
    data = {m: pd.read_csv(res / f"{m}.csv") for m in models if (res / f"{m}.csv").exists()}
    if not data:
        return
    for m, df in data.items():
        df = df[~df["corruption"].str.match(r"^[mr]PC")].copy()
        df["severity"] = df["severity"].astype(int)
        df[metric] = df[metric].astype(float)
        if "group" not in df:
            df["group"] = df["corruption"].map(CORRUPTION_GROUP).fillna("clean")
        data[m] = df
    order = list(SEEN_CORRUPTIONS) + list(HELD_OUT_CORRUPTIONS)
    present = {c for df in data.values() for c in df["corruption"] if c != "clean"}
    corrs = [c for c in order if c in present] + sorted(present - set(order))
    panels = corrs + ["mean seen", "mean held_out", "mean all"]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        for m, df in data.items():
            clean = float(df.loc[df["corruption"] == "clean", metric].iloc[0])
            if panel.startswith("mean"):
                g = panel.split()[1]
                sel = df[df["corruption"] != "clean"]
                sel = sel if g == "all" else sel[sel["group"] == g]
                cur = sel.groupby("severity")[metric].mean()
            else:
                cur = df[df["corruption"] == panel].set_index("severity")[metric].sort_index()
            ax.plot([0, *cur.index], [clean, *cur.values], marker="o", ms=3, label=m)
        title = panel.replace("_", " ")
        if CORRUPTION_GROUP.get(panel) == "held_out":
            title += " (held-out)"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel(metric.replace("map", "mAP@").replace("_", ":"))
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, f"robustness_vs_severity_{metric}")


def risk_coverage(res: Path, models, out: Path):
    have = [m for m in models if (res / f"{m}_riskcov.csv").exists()]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    target = None
    for m in have:
        df = pd.read_csv(res / f"{m}_riskcov.csv")
        df = df[df["split"] == "test"]
        line, = ax.plot(df["coverage"], df["risk"], label=m)
        j = res / f"{m}_abstain.json"
        if j.exists():
            info = json.loads(j.read_text())
            target = info["target_risk"]
            t = info["test"]
            ax.plot(t["coverage"], t["selective_risk"], "o", color=line.get_color())
    if target is not None:
        ax.axhline(target, ls="--", color="grey", lw=1, label=f"target risk {target:.0%}")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (affiliation)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out, "risk_coverage")


def per_class(res: Path, models, out: Path):
    data = {m: pd.read_csv(res / f"{m}_per_class.csv") for m in models if (res / f"{m}_per_class.csv").exists()}
    if not data:
        return
    classes = list(dict.fromkeys(c for df in data.values() for c in df["class"]))
    aff = {r["class"]: r["affiliation"] for df in data.values() for _, r in df.iterrows()}
    y = np.arange(len(classes))
    h = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(classes) + 1.2))
    for i, (m, df) in enumerate(data.items()):
        v = df.set_index("class")["ap50"].reindex(classes).fillna(0).values
        ax.barh(y + i * h - 0.4 + h / 2, v, height=h, label=m)
    ax.set_yticks(y, classes, fontsize=7)
    for t in ax.get_yticklabels():
        t.set_color(AFF_COLOR.get(aff.get(t.get_text()), "black"))
    ax.invert_yaxis()
    ax.set_xlabel("AP@50 (clean test)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Line2D([], [], ls="", marker="s", color=c, label=a) for a, c in AFF_COLOR.items()]
    leg1 = ax.legend(fontsize=7, loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=handles, fontsize=6, loc="upper right", title="label colour", title_fontsize=6)
    fig.tight_layout()
    save(fig, out, "per_class_ap")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = Path(args.results)
    out = Path(args.out) if args.out else res / "figures"
    riskcov = sorted(p.stem[: -len("_riskcov")] for p in res.glob("*_riskcov.csv"))
    models = args.models or sorted(set(discover(res)) | set(riskcov))
    if not models:
        raise SystemExit(f"no results in {res}; run evaluate.py / abstain.py first")
    robustness(res, models, out, "map50_95")
    robustness(res, models, out, "map50")
    risk_coverage(res, models, out)
    per_class(res, models, out)


if __name__ == "__main__":
    main()
