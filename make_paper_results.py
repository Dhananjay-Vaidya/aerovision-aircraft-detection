"""All paper outputs from results/: one summary table, LaTeX tables and figures.

  python make_paper_results.py

Models (in this order; a model is included when results/<key>_clean_test.json exists):
  yolov8s, yolo26s, yolo26s_plainft, yolo26s_degrade
Inputs per model (Part B of the final evaluation, clean splits R1-R3):
  results/<key>_clean_test.json, <key>_full_test.json   eval_test.py
  results/<key>.csv, <key>_per_class.csv                evaluate.py (Aircraft-C on the clean test file names)
  results/<key>_abstain.json, <key>_matched_abstain.json, <key>_matched_riskcov.csv   abstain.py (r* 0.05, 0.10)
  results/speed_rounds.csv                              speed.py, 3 runs per model (column "round"); median used
                                                        (falls back to results/speed_final.csv, one run)
Outputs
  results/summary.md                one table, every number, one column per model
  paper_assets/tables/*.tex         booktabs tabulars (no table wrapper): t3_clean, t4_robust, t5_abstain, t6_speed,
                                    t_percorruption. Values in % with one decimal, ms with one decimal; best value
                                    per column in bold (per row in t_percorruption, whose columns are models)
  paper_assets/figures/*.pdf + .png fig_robustness, fig_riskcov, fig_perclass, fig_ft_gain (12 cm, serif, readable
                                    in greyscale). fig_riskcov and fig_ft_gain need yolo26s_degrade and are skipped
                                    with a message until its results exist. PNGs are gitignored.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import HELD_OUT_CORRUPTIONS, ROOT, SEEN_CORRUPTIONS  # noqa: E402

RES = ROOT / "results"
TAB = ROOT / "paper_assets" / "tables"
FIG = ROOT / "paper_assets" / "figures"
MODELS = [("yolov8s", "YOLOv8-s"), ("yolo26s", "YOLO26-s"), ("yolo26s_plainft", "YOLO26-s + plain FT"),
          ("yolo26s_degrade", "YOLO26-s + degradation FT")]
LABEL = dict(MODELS)
CORRUPTIONS = list(SEEN_CORRUPTIONS) + list(HELD_OUT_CORRUPTIONS)
CORR_LABEL = {"defocus_blur": "Defocus blur", "motion_blur": "Motion blur", "fog": "Fog", "low_light": "Low light",
              "gaussian_noise": "Gaussian noise", "jpeg": "JPEG", "downscale": "Downscale", "contrast": "Contrast",
              "rain": "Rain"}
TARGETS = ("0.05", "0.10")
# greyscale-safe: each model has its own marker, line style and grey-distinct colour
STYLE = {"yolov8s": dict(color="#000000", marker="s", ls="-"),
         "yolo26s": dict(color="#0072B2", marker="o", ls="--"),
         "yolo26s_plainft": dict(color="#D55E00", marker="^", ls="-."),
         "yolo26s_degrade": dict(color="#009E73", marker="D", ls=":")}
AFF_STYLE = {"allied": ("#4d4d4d", ""), "adversary": ("#000000", "////"), "ambiguous": ("#a6a6a6", ".."),
             "unknown": ("#ffffff", "xx")}
CM = 1 / 2.54
WIDTH = 12 * CM
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "hatch.linewidth": 0.5,
})


# ----------------------------------------------------------------------------------------------- loading
def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_model(key: str) -> dict | None:
    clean = _json(RES / f"{key}_clean_test.json")
    if clean is None:
        return None
    m = {"key": key, "label": LABEL.get(key, key), "clean": clean, "full": _json(RES / f"{key}_full_test.json"),
         "abstain": {"all": _json(RES / f"{key}_abstain.json"), "matched": _json(RES / f"{key}_matched_abstain.json")}}
    ev = RES / f"{key}.csv"
    m["ev"] = pd.read_csv(ev) if ev.exists() else None
    pc = RES / f"{key}_per_class.csv"
    m["per_class"] = pd.read_csv(pc) if pc.exists() else None
    rc = RES / f"{key}_matched_riskcov.csv"
    m["riskcov_matched"] = pd.read_csv(rc) if rc.exists() else None
    # speed_rounds.csv: several runs of the same protocol (column "round"); the laptop GPU's clocks vary by up to
    # ~50 % between runs, so each statistic is the median over rounds. Falls back to a single speed_final.csv run.
    m["speed"], m["speed_rounds"] = None, 0
    for sp in (RES / "speed_rounds.csv", RES / "speed_final.csv"):
        if not sp.exists():
            continue
        s = pd.read_csv(sp)
        s = s[s["model"] == key]
        if len(s):
            num = ["ms_mean", "ms_p95", "fps", "fps_p95"]
            m["speed"] = {mode: g[num].median() for mode, g in s.groupby("mode")}
            m["speed_rounds"] = s["round"].nunique() if "round" in s else 1
            break
    return m


def ev_row(m, name, col):
    if m["ev"] is None:
        return None
    r = m["ev"][m["ev"]["corruption"] == name]
    return float(r[col].iloc[0]) if len(r) else None


def per_corruption_mpc(m, col="map50"):
    """mPC of one corruption = mean over its 3 severities."""
    if m["ev"] is None:
        return {}
    d = m["ev"][m["ev"]["corruption"].isin(CORRUPTIONS)]
    return d.groupby("corruption")[col].mean().to_dict()


def ab(m, mode, target, split, key):
    a = m["abstain"].get(mode)
    if not a:
        return None
    blk = a.get("by_target", {}).get(target)
    if blk is None or split not in blk:
        return None
    return blk[split].get(key) if key != "threshold" else blk["val"].get("threshold")


# ----------------------------------------------------------------------------------------------- metrics
def metric_rows():
    """(section, label, getter(m), kind) with kind in pct | ms | num1 | num2 | num3 | int | fps."""
    R = []
    sec = "Test (no corruption)"
    for sset in ("clean", "full"):
        for k, lab in (("mAP50", "mAP@50"), ("mAP50_95", "mAP@50:95"), ("precision", "Precision"),
                       ("recall", "Recall")):
            R.append((sec, f"{lab}, {sset} test", lambda m, s=sset, k=k: (m[s] or {}).get(k), "pct"))
        R.append((sec, f"Images, {sset} test", lambda m, s=sset: (m[s] or {}).get("n_images",
                                                                                  (m[s] or {}).get("n_test_images")),
                  "int"))
    R.append((sec, "Params (M, fused)", lambda m: m["clean"].get("params_M"), "num2"))
    R.append((sec, "GFLOPs (fused)", lambda m: m["clean"].get("gflops"), "num1"))
    sec = "Aircraft-C (clean test file names)"
    for col, lab in (("map50", "mAP@50"), ("map50_95", "mAP@50:95")):
        R.append((sec, f"Clean {lab}", lambda m, c=col: ev_row(m, "clean", c), "pct"))
        for g, gl in (("seen", "seen"), ("held_out", "held-out"), ("all", "all")):
            R.append((sec, f"mPC {gl} ({lab})", lambda m, c=col, g=g: ev_row(m, f"mPC_{g}", c), "pct"))
        R.append((sec, f"rPC = mPC all / clean ({lab})", lambda m, c=col: ev_row(m, "rPC_all", c), "pct"))
    for c in CORRUPTIONS:
        R.append((sec, f"mPC {CORR_LABEL[c]}{' †' if c in HELD_OUT_CORRUPTIONS else ''} (mAP@50)",
                  lambda m, c=c: per_corruption_mpc(m).get(c), "pct"))
    for mode, ml in (("all", "risk on all detections"), ("matched", "risk on matched detections")):
        sec = f"Abstention, {ml}"
        for split, sl in (("val", "val"), ("test", "clean test"), ("aircraft_c", "Aircraft-C")):
            R.append((sec, f"AURC, {sl}", lambda m, mo=mode, s=split: ab(m, mo, "0.05", s, "aurc"), "pct"))
        for t in TARGETS:
            tl = f"r*={float(t):.0%}"
            R.append((sec, f"τ (val, {tl})", lambda m, mo=mode, t=t: ab(m, mo, t, "val", "threshold"), "num3"))
            for split, sl in (("val", "val"), ("test", "clean test"), ("aircraft_c", "Aircraft-C")):
                for k, kl, kind in (("coverage", "Coverage", "pct"), ("selective_risk", "Selective risk", "pct"),
                                    ("affiliation_accuracy", "Affiliation acc.", "pct"),
                                    ("false_alarms_per_100_images", "Accepted FA / 100 img", "num2")):
                    R.append((sec, f"{kl}, {sl} ({tl})",
                              lambda m, mo=mode, t=t, s=split, k=k: ab(m, mo, t, s, k), kind))
    sec = ("Speed (RTX 3050 Laptop, FP16, batch 1, 640 px; 50 warm-up + 300 timed; "
           "median over repeated runs, see results/speed_rounds.csv)")
    for mode, ml in (("forward", "Forward"), ("end_to_end", "End-to-end")):
        for k, kl, kind in (("ms_mean", "ms mean", "ms"), ("ms_p95", "ms p95", "ms"), ("fps", "FPS (mean)", "fps"),
                            ("fps_p95", "FPS (at p95 latency)", "fps")):
            R.append((sec, f"{ml} {kl}", lambda m, mo=mode, k=k: _speed(m, mo, k), kind))
    return R


def _speed(m, mode, k):
    s = m["speed"]
    if not s or mode not in s or k not in s[mode] or pd.isna(s[mode][k]):
        return None
    return float(s[mode][k])


def fmt(v, kind):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if kind == "pct":
        return f"{100 * v:.1f}"
    if kind == "ms":
        return f"{v:.1f}"
    if kind == "fps":
        return f"{v:.1f}"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.{int(kind[-1])}f}"


def write_summary(models):
    rows = metric_rows()
    md = ["# Final results (clean splits R1–R3; original labels)", "",
          "Percentages for mAP, precision, recall, mPC, rPC, coverage, risk, accuracy and AURC. τ is a confidence "
          "threshold chosen on clean val (maximum coverage with val risk ≤ r*). Accepted FA / 100 img = accepted "
          "detections that match no GT box, per 100 images. – = not available.", "",
          "Speed caveat: the laptop GPU (RTX 3050 Laptop) varies by up to ~50 % between runs of the same protocol, "
          "and the rounds of different models were not all timed in one session. The three YOLO26-s models have "
          "the same architecture (9.48 M params, 20.8 GFLOPs fused), so speed differences between them are "
          "measurement noise, not model differences.", ""]
    missing = [lab for k, lab in MODELS if k not in {m["key"] for m in models}]
    if missing:
        md += [f"Not yet evaluated: {', '.join(missing)}.", ""]
    md += ["| Metric | " + " | ".join(m["label"] for m in models) + " |", "|---|" + "---:|" * len(models)]
    cur = None
    for sec, lab, get, kind in rows:
        if sec != cur:
            md.append(f"| **{sec}** |" + " |" * len(models))
            cur = sec
        md.append(f"| {lab} | " + " | ".join(fmt(get(m), kind) for m in models) + " |")
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[paper] wrote {RES / 'summary.md'} ({len(rows)} rows x {len(models)} models)")


# ----------------------------------------------------------------------------------------------- LaTeX
def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("†", r"$^{\dagger}$").replace("&", r"\&")


def tex_table(path: Path, header: list[str], rows: list[list], kinds: list[str], better: list[str | None],
              bold_axis: str = "column", first_col_align="l"):
    """rows: [label, v1, v2, ...]; better: per value column 'max' | 'min' | None (no bolding)."""
    vals = [r[1:] for r in rows]
    bold = [[False] * len(kinds) for _ in rows]

    def mark(cells):  # cells: list of (i, j)
        if not cells:
            return
        b = better[cells[0][1]] if bold_axis == "column" else better[0]
        nums = [(vals[i][j], i, j) for i, j in cells if _num(vals[i][j]) is not None]
        if not nums or b is None:
            return
        key = [round(_num(v) * (100 if kinds[j] == "pct" else 1), 1) for v, i, j in nums]
        best = max(key) if b == "max" else min(key)
        for (v, i, j), kv in zip(nums, key):
            if kv == best:
                bold[i][j] = True

    if bold_axis == "column":
        for j in range(len(kinds)):
            mark([(i, j) for i in range(len(rows))])
    else:
        for i in range(len(rows)):
            mark([(i, j) for j in range(len(kinds))])
    spec = first_col_align + "r" * len(kinds)
    out = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\",
           r"\midrule"]
    for r, (row, b) in enumerate(zip(rows, bold)):
        cells = [tex_escape(str(row[0]))]
        for j, v in enumerate(row[1:]):
            s = fmt(v, kinds[j]).replace("–", "--")
            cells.append(r"\textbf{" + s + "}" if b[j] and s != "--" else s)
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[paper] wrote {path}")


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def write_tables(models):
    c = lambda m, k: m["clean"].get(k)
    tex_table(TAB / "t3_clean.tex",
              ["Model", "mAP@50", "mAP@50:95", "Precision", "Recall", "Params (M, fused)", "GFLOPs"],
              [[m["label"], c(m, "mAP50"), c(m, "mAP50_95"), c(m, "precision"), c(m, "recall"), c(m, "params_M"),
                c(m, "gflops")] for m in models],
              ["pct", "pct", "pct", "pct", "num1", "num1"], ["max", "max", "max", "max", "min", "min"])
    tex_table(TAB / "t4_robust.tex",
              ["Model", "Clean", "mPC seen", "mPC held-out", "mPC all", "rPC"],
              [[m["label"], ev_row(m, "clean", "map50"), ev_row(m, "mPC_seen", "map50"),
                ev_row(m, "mPC_held_out", "map50"), ev_row(m, "mPC_all", "map50"), ev_row(m, "rPC_all", "map50")]
               for m in models],
              ["pct"] * 5, ["max"] * 5)
    tex_table(TAB / "t5_abstain.tex",
              ["Model", "AURC test", "AURC Aircraft-C", "Cov@5% test", "Risk@5% test", "Cov@10% test",
               "Risk@10% test", "Cov@10% Aircraft-C", "FA/100 img (test, at 10%)"],
              [[m["label"], ab(m, "matched", "0.05", "test", "aurc"), ab(m, "matched", "0.05", "aircraft_c", "aurc"),
                ab(m, "matched", "0.05", "test", "coverage"), ab(m, "matched", "0.05", "test", "selective_risk"),
                ab(m, "matched", "0.10", "test", "coverage"), ab(m, "matched", "0.10", "test", "selective_risk"),
                ab(m, "matched", "0.10", "aircraft_c", "coverage"),
                ab(m, "matched", "0.10", "test", "false_alarms_per_100_images")] for m in models],
              ["pct", "pct", "pct", "pct", "pct", "pct", "pct", "num1"],
              ["min", "min", "max", "min", "max", "min", "max", "min"])
    tex_table(TAB / "t6_speed.tex", ["Model", "Forward ms", "End-to-end ms", "FPS"],
              [[m["label"], _speed(m, "forward", "ms_mean"), _speed(m, "end_to_end", "ms_mean"),
                _speed(m, "end_to_end", "fps")] for m in models],
              ["ms", "ms", "fps"], ["min", "min", "max"])
    mpcs = {m["key"]: per_corruption_mpc(m) for m in models}
    tex_table(TAB / "t_percorruption.tex",
              ["Corruption"] + [m["label"] for m in models],
              [[CORR_LABEL[c] + ("†" if c in HELD_OUT_CORRUPTIONS else "")] + [mpcs[m["key"]].get(c) for m in models]
               for c in CORRUPTIONS],
              ["pct"] * len(models), ["max"], bold_axis="row")


# ----------------------------------------------------------------------------------------------- figures
def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[paper] wrote {FIG / stem}.pdf/.png")


def fig_robustness(models):
    models = [m for m in models if m["ev"] is not None]
    if not models:
        return print("[paper] fig_robustness skipped: no evaluate.py results")
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 10.5 * CM), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, CORRUPTIONS):
        held = c in HELD_OUT_CORRUPTIONS
        if held:
            ax.set_facecolor("#ededed")
        for m in models:
            d = m["ev"]
            cl = float(d.loc[d.corruption == "clean", "map50"].iloc[0])
            cur = d[d.corruption == c].set_index("severity")["map50"].sort_index()
            ax.plot([0, *cur.index], [cl, *cur.values], label=m["label"], **STYLE.get(m["key"], {}))
        ax.set_title(CORR_LABEL[c] + (r"$^{\dagger}$" if held else ""), pad=2)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP@50")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 4), frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handlelength=2.6)
    fig.text(0.99, 0.0, r"$^{\dagger}$held-out corruption (shaded); severity 0 = clean test", ha="right",
             va="bottom", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=0.5, w_pad=0.4)
    save(fig, "fig_robustness")


def fig_riskcov(models):
    need = ["yolo26s", "yolo26s_plainft", "yolo26s_degrade"]
    have = {m["key"]: m for m in models}
    miss = [k for k in need if k not in have or have[k]["riskcov_matched"] is None]
    if miss:
        return print(f"[paper] fig_riskcov skipped: no matched-mode abstention results for {', '.join(miss)}")
    fig, ax = plt.subplots(figsize=(WIDTH, 6.5 * CM))
    for k in need:
        d = have[k]["riskcov_matched"]
        st = STYLE[k]
        for split, ls in (("test", "-"), ("aircraft_c", "--")):
            s = d[d.split == split]
            ax.plot(s.coverage, s.risk, color=st["color"], ls=ls, marker=st["marker"],
                    markevery=max(1, len(s) // 8), lw=1.0)
    for r in (0.05, 0.10):
        ax.axhline(r, color="black", lw=0.6, ls=":")
        ax.text(0.995, r, f" {r:.0%}", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=6.5)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=STYLE[k]["color"], marker=STYLE[k]["marker"], ls="-", label=LABEL[k]) for k in need]
    hs += [Line2D([], [], color="0.3", ls="-", label="Clean test"),
           Line2D([], [], color="0.3", ls="--", label="Aircraft-C")]
    ax.legend(handles=hs, loc="upper left", frameon=False, ncol=1)
    ax.set_xlabel("Coverage (fraction of matched detections given an affiliation)")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    save(fig, "fig_riskcov")


def fig_perclass(models):
    ok = [m for m in models if m["per_class"] is not None]
    if not ok:
        return print("[paper] fig_perclass skipped: no per-class results")
    best = max(ok, key=lambda m: m["clean"]["mAP50"])
    d = best["per_class"].sort_values("ap50")
    n = len(d)
    fig, ax = plt.subplots(figsize=(WIDTH, (0.32 * n + 1.5) * CM))
    y = np.arange(n)
    for yi, (_, r) in zip(y, d.iterrows()):
        col, hatch = AFF_STYLE.get(r["affiliation"], ("#808080", ""))
        ax.barh(yi, r["ap50"], color=col, hatch=hatch, edgecolor="black", linewidth=0.4, height=0.7)
        ax.text(r["ap50"] + 0.01, yi, f"{r['ap50']:.2f}", va="center", fontsize=6)
    ax.set_yticks(y, d["class"].str.replace("_", " "), fontsize=6.5)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("AP@50 (clean test)")
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    present = [a for a in AFF_STYLE if a in set(d["affiliation"])]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=AFF_STYLE[a][0], hatch=AFF_STYLE[a][1],
                                     edgecolor="black", linewidth=0.4) for a in present],
              labels=[a.capitalize() for a in present], loc="lower right", frameon=True, framealpha=0.9)
    ax.set_title(best["label"], pad=2)
    fig.tight_layout()
    save(fig, "fig_perclass")


def fig_ft_gain(models):
    have = {m["key"]: m for m in models}
    if "yolo26s_degrade" not in have or "yolo26s_plainft" not in have or have["yolo26s_degrade"]["ev"] is None \
            or have["yolo26s_plainft"]["ev"] is None:
        return print("[paper] fig_ft_gain skipped: needs evaluate.py results for yolo26s_plainft and yolo26s_degrade")
    a, b = per_corruption_mpc(have["yolo26s_degrade"]), per_corruption_mpc(have["yolo26s_plainft"])
    gain = [100 * (a[c] - b[c]) for c in CORRUPTIONS]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.5 * CM))
    x = np.arange(len(CORRUPTIONS))
    for xi, c, g in zip(x, CORRUPTIONS, gain):
        held = c in HELD_OUT_CORRUPTIONS
        ax.bar(xi, g, color="#bfbfbf" if held else "#404040", hatch="////" if held else "", edgecolor="black",
               linewidth=0.4, width=0.7)
        ax.text(xi, g + (0.2 if g >= 0 else -0.2), f"{g:+.1f}", ha="center", va="bottom" if g >= 0 else "top",
                fontsize=6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [CORR_LABEL[c] + (r"$^{\dagger}$" if c in HELD_OUT_CORRUPTIONS else "") for c in CORRUPTIONS],
                  rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta$ mPC, mAP@50 (points)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#404040", edgecolor="black", linewidth=0.4),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#bfbfbf", hatch="////", edgecolor="black",
                                     linewidth=0.4)],
              labels=["Seen in fine-tuning", r"Held-out$^{\dagger}$"], frameon=False, loc="best")
    ax.set_title("Degradation fine-tune minus plain fine-tune", pad=2)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    save(fig, "fig_ft_gain")


def main():
    models = [m for m in (load_model(k) for k, _ in MODELS) if m is not None]
    if not models:
        raise SystemExit("no results/<model>_clean_test.json found; run the final evaluation first")
    print("[paper] models:", ", ".join(m["key"] for m in models))
    for m in models:
        miss = [n for n, v in (("full test", m["full"]), ("Aircraft-C", m["ev"]), ("per-class", m["per_class"]),
                               ("abstain all", m["abstain"]["all"]), ("abstain matched", m["abstain"]["matched"]),
                               ("speed", m["speed"])) if v is None]
        for n in ("all", "matched"):
            a = m["abstain"][n]
            if a is not None and not all(t in a.get("by_target", {}) for t in TARGETS):
                miss.append(f"abstain {n} at r*={'/'.join(TARGETS)}")
        if miss:
            print(f"[paper] {m['key']}: missing {', '.join(miss)}")
    write_summary(models)
    write_tables(models)
    fig_robustness(models)
    fig_riskcov(models)
    fig_perclass(models)
    fig_ft_gain(models)


if __name__ == "__main__":
    main()
