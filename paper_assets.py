"""Paper-ready tables, statistics and figures -> paper_assets/.

  python paper_assets.py                     # all assets
  python paper_assets.py --image <test.jpg>  # choose the Aircraft-C example image yourself

  table_sources.tex       images per source: seen -> removed (unlabelled, augmented copies, exact, near dup.)
                          -> after dedupe -> dropped by class caps -> final
  table_classes.tex       per-class train/val/test image counts + affiliation; merged classes marked with a dagger
  dedupe_stats.txt        duplicate counts, per source, cross-source, and merge_dataset valid/test -> train leakage
  fig_aircraft_c.*        one test image: clean + 9 corruptions x 3 severities
  fig_pipeline.*          image -> detector -> type + confidence -> affiliation lookup + tau -> decision
  fig_class_counts.*      train images per class, coloured by affiliation

Inputs: cvs_dataset/{stats_per_source,stats_per_class}.csv, cvs_dataset/build_report.json,
        cvs_dataset_dedupe_report.csv, raw/<source>/<split>/images (only to count merge_dataset's valid/test).
Figures: serif font, 12 cm wide (single-column Springer), vector PDF + 300 dpi PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

import corrupt  # noqa: E402
from common import HELD_OUT_CORRUPTIONS, MIN_TRAIN_IMAGES, OTHER_CLASS, ROOT, SEEN_CORRUPTIONS, \
    class_merge_map, list_images, load_names  # noqa: E402

CM = 1 / 2.54
WIDTH = 12 * CM
AFF_COLOR = {"allied": "#2e7d32", "adversary": "#c62828", "ambiguous": "#e69f00", "unknown": "#757575"}
SOURCE_LABEL = {"own": "Own", "fyp": "FYP", "merge": r"merge\_dataset"}
SHORT = {"defocus_blur": "defocus", "motion_blur": "motion", "fog": "fog", "low_light": "low light",
         "gaussian_noise": "noise", "jpeg": "JPEG", "downscale": "downscale", "contrast": "contrast",
         "rain": "rain"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def tex(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[paper_assets] {stem}.pdf/.png")


def src_split(path: str) -> tuple[str, str]:
    parts = path.replace("\\", "/").split("/")
    i = parts.index("raw") if "raw" in parts else 0
    return parts[i + 1], parts[i + 2]


# ---------------------------------------------------------------------------------------------------------------------
def load_inputs(ds: Path, dedupe_csv: Path):
    report = json.loads((ds / "build_report.json").read_text())
    per_source = {r["source"]: r for r in csv.DictReader(open(ds / "stats_per_source.csv", encoding="utf-8"))}
    per_class = list(csv.DictReader(open(ds / "stats_per_class.csv", encoding="utf-8")))
    dedupe = list(csv.DictReader(open(dedupe_csv, encoding="utf-8")))
    return report, per_source, per_class, dedupe


def source_rows(report, per_source, dedupe):
    removed = Counter((r["type"], src_split(r["removed_image"])[0]) for r in dedupe)
    rows = []
    for s in ("own", "fyp", "merge"):
        seen = report.get(f"{s}_files_seen", 0)
        unl = report.get(f"{s}_unlabelled_skipped", 0)
        aug = report.get(f"{s}_augmented_copies_removed", 0)
        ex, nr = removed[("exact", s)], removed[("near", s)]
        after = seen - unl - aug - ex - nr
        final = int(per_source[s]["total"])
        rows.append({"source": s, "seen": seen, "unlabelled": unl, "augmented": aug, "exact": ex, "near": nr,
                     "after_dedupe": after, "capped": after - final, "final": final,
                     "train": int(per_source[s]["train"]), "val": int(per_source[s]["val"]),
                     "test": int(per_source[s]["test"])})
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k != "source"}
    tot["source"] = "total"
    # consistency with build_report.json
    assert tot["exact"] == report["exact_duplicates_removed"], "exact duplicates do not match build_report.json"
    assert tot["near"] == report["near_duplicates_removed"], "near duplicates do not match build_report.json"
    caps = report.get("dropped_by_class_cap", 0) + report.get("dropped_by_other_cap", 0)
    unreadable = report.get("unreadable_images_skipped", 0)
    assert tot["capped"] - unreadable == caps, f"cap drops {tot['capped']} != build_report {caps}"
    return rows, tot


def table_sources(rows, tot, out: Path):
    cols = ["seen", "unlabelled", "augmented", "exact", "near", "after_dedupe", "capped", "final"]
    lines = [
        "% Generated by paper_assets.py -- images per source through the build pipeline.",
        "% 'Capped' per source = after dedupe - final (caps are logged only in total; the column sums to it).",
        "% Requires \\usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Images per source before and after de-duplication. \emph{Unlab.}: no box of a target class; "
        r"\emph{Aug.}: Roboflow augmented copies collapsed to one per original; \emph{Exact}/\emph{Near}: MD5 and "
        r"perceptual-hash duplicates removed; \emph{Capped}: removed by the per-class caps that limit dominant "
        r"classes; \emph{Final}: images in the released dataset.}",
        r"\label{tab:sources}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{Removed} & After & & \\",
        r"\cmidrule(lr){3-6}",
        r"Source & Seen & Unlab. & Aug. & Exact & Near & dedupe & Capped & Final \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(SOURCE_LABEL[r["source"]] + " & " + " & ".join(fmt(r[c]) for c in cols) + r" \\")
    lines += [r"\midrule", "Total & " + " & ".join(fmt(tot[c]) for c in cols) + r" \\", r"\bottomrule",
              r"\end{tabular}", r"\end{table}", ""]
    (out / "table_sources.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[paper_assets] table_sources.tex")


def table_classes(per_class, merged: list[str], out: Path):
    lines = [
        "% Generated by paper_assets.py -- per-class image counts (images containing >= 1 box of the class).",
        "% Requires \\usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Images per class and split, with the affiliation used by the abstention layer. "
        rf"$^\dagger$Fewer than {MIN_TRAIN_IMAGES} training images: relabelled \texttt{{other\_aircraft}} "
        rf"for training and evaluation.}}",
        r"\label{tab:classes}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Class & Affiliation & Train & Val & Test & Total \\",
        r"\midrule",
    ]
    prev = None
    tot = Counter()
    for r in per_class:
        if prev is not None and r["affiliation"] != prev:
            lines.append(r"\addlinespace")
        prev = r["affiliation"]
        name = tex(r["class"]) + (r"$^\dagger$" if r["class"] in merged else "")
        nums = [int(r[k]) for k in ("train_images", "val_images", "test_images", "total_images")]
        tot.update(dict(zip("tvsa", nums)))
        lines.append(f"{name} & {r['affiliation'].capitalize()} & " + " & ".join(fmt(n) for n in nums) + r" \\")
    lines += [r"\midrule", "Total (images) & & " + " & ".join(fmt(tot[k]) for k in "tvsa") + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}",
              "% Note: an image with several classes is counted once per class, so column totals exceed the",
              "% number of images (see table_sources.tex for image totals).", ""]
    (out / "table_classes.tex").write_text("\n".join(lines), encoding="utf-8")
    print("[paper_assets] table_classes.tex")


def dedupe_stats(report, rows, tot, dedupe, raw: Path, out: Path):
    by_type = Counter(r["type"] for r in dedupe)
    pair = Counter((src_split(r["removed_image"])[0], src_split(r["kept_duplicate_of"])[0]) for r in dedupe)
    sp = Counter((r["type"],) + src_split(r["removed_image"]) + src_split(r["kept_duplicate_of"]) for r in dedupe)
    merge_split_seen = {s: len(list_images(raw / "merge" / s / "images")) if (raw / "merge" / s / "images").is_dir()
                        else None for s in ("train", "valid", "test")}
    leak = {s: sum(v for (t, rs, rsp, ks, ksp), v in sp.items() if rs == "merge" and rsp == s
                   and ks == "merge" and ksp == "train") for s in ("valid", "test")}
    rev = sum(v for (t, rs, rsp, ks, ksp), v in sp.items() if rs == "merge" and rsp == "train"
              and ks == "merge" and ksp in ("valid", "test"))
    within_vt = sum(v for (t, rs, rsp, ks, ksp), v in sp.items() if rs == "merge" and rsp in ("valid", "test")
                    and ks == "merge" and ksp in ("valid", "test"))
    L = ["CVS dataset - duplicate statistics (generated by paper_assets.py)",
         "Source of truth: cvs_dataset_dedupe_report.csv (one row per removed image) and build_report.json.",
         "Rule: exact = identical MD5; near = whole-image pHash AND largest-aircraft-crop pHash both within "
         "Hamming distance 4 of an already kept image. Sources are processed in priority order own > fyp > merge, "
         "so the earlier copy is kept.",
         "",
         f"Images removed as duplicates: {len(dedupe)}  (exact {by_type['exact']}, near {by_type['near']})",
         f"  build_report.json agrees: exact {report['exact_duplicates_removed']}, "
         f"near {report['near_duplicates_removed']}",
         "",
         "Per source (removed image's source):"]
    L.append(f"  {'source':<8}{'seen':>8}{'unlab.':>8}{'aug.':>8}{'exact':>8}{'near':>8}{'dup. total':>12}"
             f"{'after dedupe':>14}{'capped':>8}{'final':>8}")
    for r in rows + [tot]:
        L.append(f"  {r['source']:<8}{r['seen']:>8}{r['unlabelled']:>8}{r['augmented']:>8}{r['exact']:>8}"
                 f"{r['near']:>8}{r['exact'] + r['near']:>12}{r['after_dedupe']:>14}{r['capped']:>8}{r['final']:>8}")
    L += ["", "Cross-source (removed source -> kept source, images):"]
    for (a, b), v in sorted(pair.items(), key=lambda x: -x[1]):
        L.append(f"  {a:>6} -> {b:<6} {v}")
    n_vt = sum(merge_split_seen[s] or 0 for s in ("valid", "test"))
    n_leak = leak["valid"] + leak["test"]
    L += ["", "merge_dataset (superradarosint/merge_dataset-9nrmd v1) - its own split leakage:",
          f"  images in its splits: train {merge_split_seen['train']}, valid {merge_split_seen['valid']}, "
          f"test {merge_split_seen['test']}",
          f"  valid images that duplicate one of its train images: {leak['valid']} / {merge_split_seen['valid']}"
          f" ({100 * leak['valid'] / max(merge_split_seen['valid'] or 1, 1):.1f}%)",
          f"  test  images that duplicate one of its train images: {leak['test']} / {merge_split_seen['test']}"
          f" ({100 * leak['test'] / max(merge_split_seen['test'] or 1, 1):.1f}%)",
          f"  valid+test total: {n_leak} / {n_vt} ({100 * n_leak / max(n_vt, 1):.1f}%)",
          f"  (reverse direction, train removed as a copy of valid/test: {rev}; "
          f"valid/test duplicates of other valid/test images: {within_vt})",
          "  -> evaluating on merge_dataset's original split would test mostly on near-copies of training images. "
          "CVS removes these duplicates (within the pHash threshold) before re-drawing its own 70/20/10 split.",
          "  Note: 'capped' per source = after dedupe - final (the build logs class-cap drops only in total); "
          "the per-source values sum to build_report.json's dropped_by_class_cap + dropped_by_other_cap.",
          "", "By type, removed split and kept split (type, removed source/split -> kept source/split: n):"]
    for (t, rs, rsp, ks, ksp), v in sorted(sp.items(), key=lambda x: -x[1]):
        L.append(f"  {t:<5} {rs}/{rsp} -> {ks}/{ksp}: {v}")
    (out / "dedupe_stats.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[paper_assets] dedupe_stats.txt")
    return {"exact": by_type["exact"], "near": by_type["near"], "leak": n_leak, "leak_of": n_vt, **leak,
            "merge_split_seen": merge_split_seen}


# ---------------------------------------------------------------------------------------------------------------------
# Aircraft-C example image: only merge_dataset (CC BY 4.0, SuperRadarOSINT) images may appear in the paper, never the
# web-collected own/fyp images. Chosen by eye from the clean test split: one F-16, fully visible against the sky.
EXAMPLE_IMAGE = "merge_test_000310.jpg"
MERGE_URL = "https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd"


def _manifest(ds: Path) -> dict:
    with open(ds / "manifest.csv", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def pick_example(ds: Path, names: list[str]) -> Path:
    """EXAMPLE_IMAGE if present, else the first clean-test merge_dataset image (sorted) with one non-'other'
    aircraft covering 8-35% of the frame, away from the border."""
    man = _manifest(ds)
    if (ds / "images" / "test" / EXAMPLE_IMAGE).exists():
        return ds / "images" / "test" / EXAMPLE_IMAGE
    clean = ds / "test_clean.txt"
    keep = {Path(l.strip()).name for l in clean.read_text().splitlines() if l.strip()} if clean.exists() else None
    for img in list_images(ds / "images" / "test"):
        if man.get(img.name, {}).get("source") != "merge" or (keep is not None and img.name not in keep):
            continue
        lbl = ds / "labels" / "test" / (img.stem + ".txt")
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        if len(rows) != 1:
            continue
        c, x, y, w, h = int(rows[0][0]), *map(float, rows[0][1:5])
        if names[c] != OTHER_CLASS and 0.08 <= w * h <= 0.35 and 0.3 < x < 0.7 and 0.3 < y < 0.7:
            return img
    raise SystemExit("no suitable merge_dataset test image for fig_aircraft_c; pass --image")


def write_credit(ds: Path, img_path: Path, out: Path):
    row = _manifest(ds).get(img_path.name)
    if row is None or row["source"] != "merge":
        raise SystemExit(f"{img_path.name} is not a merge_dataset image: it must not appear in the paper")
    lines = ["Example image in fig_aircraft_c (pdf/png):",
             f"  dataset file:      cvs_dataset/images/test/{img_path.name}",
             f"  source filename:   {Path(row['original_path']).name}",
             f"  source dataset:    merge_dataset (SuperRadarOSINT), {MERGE_URL}",
             "  licence:           CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
             "  modifications:     corrupted with Aircraft-C (corrupt.py, seed 1234) at 9 corruptions x 3 severities",
             "Suggested caption credit: \"Example image from merge_dataset by SuperRadarOSINT (Roboflow Universe), "
             "CC BY 4.0; corruptions ours.\""]
    (out / "fig_aircraft_c_credit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_aircraft_c(ds: Path, names, out: Path, image: Path | None, seed: int = 1234):
    img_path = image or pick_example(ds, names)
    write_credit(ds, img_path, out)
    img = cv2.imread(str(img_path))
    panels = [(c, s) for c in SEEN_CORRUPTIONS + HELD_OUT_CORRUPTIONS for s in (1, 2, 3)] + [("clean", 0)]
    ncol, nrow = 6, int(np.ceil(len(panels) / 6))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, WIDTH / ncol * nrow * 1.13))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (c, s) in zip(axes.flat, panels):
        if c == "clean":
            x, title = img, r"$\bf{clean}$"
        else:
            if c == "jpeg":
                x = cv2.imdecode(corrupt.jpeg_bytes(img, s), cv2.IMREAD_COLOR)
            else:
                x = corrupt.apply(img, c, s, np.random.default_rng(corrupt.seed_for(seed, c, s, img_path.name)))
            title = f"{SHORT[c]}{'$^\\dagger$' if c in HELD_OUT_CORRUPTIONS else ''} {s}"
        ax.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), interpolation="lanczos")
        ax.set_title(title, fontsize=6.5, pad=1.5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.04, hspace=0.18)
    save(fig, out, "fig_aircraft_c")
    return img_path


def fig_pipeline(out: Path):
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0 * CM))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f2f2", ec="#333333", lw=0.8, fs=7.5, bold=False, color="black"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(p, q, style="arc3", text=None, tpos=None, fs=6.5):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333",
                                     connectionstyle=style, shrinkA=0, shrinkB=0))
        if text:
            ax.text(*tpos, text, fontsize=fs, ha="center", va="center", color="#333333")

    # row 1: image -> detector -> type + confidence -> affiliation lookup   (box edges in cm)
    y1, h, fs = 3.95, 1.05, 7
    row1 = [(0.1, 1.75, "Input image\n(640 px)"), (2.3, 5.3, "Detector\nYOLO26s / YOLOv8s"),
            (5.85, 8.2, "Aircraft type $\\hat{y}$\n+ confidence $c$"), (8.75, 11.9, "Affiliation\nlookup $a(\\hat{y})$")]
    for (x0, x1, t) in row1:
        box((x0 + x1) / 2, y1, x1 - x0, h, t, fs=fs)
    for (_, a, _), (b, _, _) in zip(row1, row1[1:]):
        arrow((a, y1), (b, y1))

    # decision
    xd, yd, wd = 9.95, 2.15, 3.8
    box(xd, yd, wd, 1.0, "$a(\\hat{y}) \\in$ {allied, adversary}\nand $c \\geq \\tau$ ?", fc="#ffffff", fs=fs)
    arrow((10.3, y1 - h / 2), (10.3, yd + 0.5))
    ax.text(6.25, yd, "$\\tau$ chosen on val for\nselective risk $\\leq$ 5%", fontsize=6.5, ha="center",
            va="center", style="italic", color="#444444")
    ax.add_patch(FancyArrowPatch((7.55, yd), (xd - wd / 2, yd), arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 ls="--", color="#666666", shrinkA=0, shrinkB=0))

    # outputs
    yo, ho = 0.5, 0.75
    box(3.0, yo, 2.0, ho, "Allied", fc=AFF_COLOR["allied"], ec=AFF_COLOR["allied"], color="white", bold=True,
        fs=7.5)
    box(5.9, yo, 2.0, ho, "Adversary", fc=AFF_COLOR["adversary"], ec=AFF_COLOR["adversary"], color="white",
        bold=True, fs=7.5)
    box(10.8, yo, 2.0, ho, "Uncertain\n(human review)", fc=AFF_COLOR["ambiguous"], ec=AFF_COLOR["ambiguous"],
        fs=6.8, bold=True)
    ybus, xy = 1.2, 8.8
    ax.plot([xy, xy], [yd - 0.5, ybus], color="#333333", lw=0.8)       # yes: down from the decision box
    ax.plot([3.0, xy], [ybus, ybus], color="#333333", lw=0.8)          # then left to allied / adversary
    arrow((3.0, ybus), (3.0, yo + ho / 2))
    arrow((5.9, ybus), (5.9, yo + ho / 2))
    ax.text(xy - 0.12, 1.42, "yes", fontsize=6.5, ha="right", va="center")
    ax.text(4.45, ybus + 0.17, "by $a(\\hat{y})$", fontsize=6.3, ha="center", va="center", color="#444444")
    arrow((10.8, yd - 0.5), (10.8, yo + ho / 2))                        # no: straight down to uncertain
    ax.text(10.95, 1.3, "no", fontsize=6.5, ha="left", va="center")
    save(fig, out, "fig_pipeline")


def fig_class_counts(per_class, merged, out: Path):
    order = ["allied", "adversary", "ambiguous", "unknown"]
    rows = sorted(per_class, key=lambda r: (order.index(r["affiliation"]), -int(r["train_images"])))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(WIDTH, (0.3 * n + 1.3) * CM))
    y = np.arange(n)
    vals = [int(r["train_images"]) for r in rows]
    for i, r in enumerate(rows):
        m = r["class"] in merged
        ax.barh(i, vals[i], height=0.72, color="white" if m else AFF_COLOR[r["affiliation"]],
                edgecolor=AFF_COLOR[r["affiliation"]], hatch="//////" if m else None, lw=0.6)
        ax.text(max(vals[i], MIN_TRAIN_IMAGES) + 18, i, f"{vals[i]:,}", va="center", fontsize=5.5)
    ax.set_yticks(y, [r["class"] + ("$^\\dagger$" if r["class"] in merged else "") for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.4, -0.6)
    ax.axvline(MIN_TRAIN_IMAGES, ls="--", lw=0.7, color="black")
    ax.set_xlabel("training images")
    ax.set_xlim(0, max(vals) * 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(fc=AFF_COLOR[a], ec=AFF_COLOR[a], label=a.capitalize()) for a in order]
    handles.append(Patch(fc="white", ec="#555555", hatch="//////",
                         label=f"merged into other_aircraft$^\\dagger$"))
    handles.append(plt.Line2D([], [], ls="--", lw=0.7, color="black",
                              label=f"{MIN_TRAIN_IMAGES}-image merge threshold"))
    ax.legend(handles=handles, loc="center right", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.2)
    save(fig, out, "fig_class_counts")


# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "cvs_dataset"))
    ap.add_argument("--dedupe-report", default=str(ROOT / "cvs_dataset_dedupe_report.csv"))
    ap.add_argument("--raw", default=str(ROOT / "raw"))
    ap.add_argument("--image", default=None, help="test image for fig_aircraft_c (default: auto-picked)")
    ap.add_argument("--out", default=str(ROOT / "paper_assets"))
    args = ap.parse_args(argv)

    ds, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(ds / "data.yaml")
    _, merged, _ = class_merge_map(ds / "data.yaml")
    report, per_source, per_class, dedupe = load_inputs(ds, Path(args.dedupe_report))

    rows, tot = source_rows(report, per_source, dedupe)
    table_sources(rows, tot, out)
    table_classes(per_class, merged, out)
    d = dedupe_stats(report, rows, tot, dedupe, Path(args.raw), out)
    ex = fig_aircraft_c(ds, names, out, Path(args.image) if args.image else None)
    fig_pipeline(out)
    fig_class_counts(per_class, merged, out)
    print(f"[paper_assets] duplicates removed: exact {d['exact']}, near {d['near']}; merge_dataset valid+test "
          f"duplicating its train: {d['leak']}/{d['leak_of']}; Aircraft-C example: {ex.name}")
    print(f"[paper_assets] wrote {out}")


if __name__ == "__main__":
    main()
