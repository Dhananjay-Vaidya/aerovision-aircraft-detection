"""Three more paper figures (called by make_paper_results.py, or run alone):

  fig_dataset_flow      source -> dedupe -> class balance -> 70/20/10 split -> clean val/test (R1-R3); no photos.
                        Counts from dataset_release/build_report.json, sources.csv, labels and paper_assets/numbers.json.
  fig_leakage_examples  one row per transform (none, hflip, vflip, rot180): a train image and the val/test image R1
                        flagged as its copy (results/leakage_audit.csv). Only pairs with BOTH images from merge_dataset;
                        a transform without such a pair gets no row (stated in the figure and the credits file).
  fig_qualitative       4 clean-test merge_dataset images x (clean | seen corruption s2 | held-out corruption s2),
                        Aircraft-C images (corrupt.py), yolo26s_degrade predictions with the abstention tau for
                        r* = 10 % (matched mode) from numbers.json. Rows picked by a fixed seed (see pick_rows).

Photo rule: every photo is a merge_dataset image (source = merge in dataset_release/sources.csv, CC BY 4.0).
paper_assets/figure_credits.txt lists each photo's source filename and its figure.
Outputs: paper_assets/figures/<name>.pdf and .png (300 dpi), 12 cm wide, serif, greyscale-readable.
"""
from __future__ import annotations

import collections
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from common import HELD_OUT_CORRUPTIONS, ROOT, SEEN_CORRUPTIONS
from make_paper_results import CM, CORR_LABEL, RES, WIDTH, save

REL = ROOT / "dataset_release"
NUMBERS = ROOT / "paper_assets" / "numbers.json"
CREDITS = ROOT / "paper_assets" / "figure_credits.txt"
IMAGES = ROOT / "cvs_dataset" / "images"
MERGED = ROOT / "data_derived" / "merged"
ACORR = ROOT / "data_derived" / "aircraft_c"
SEED = 2026
TRANSFORMS = ("none", "hflip", "vflip", "rot180")
TF_LABEL = {"none": "no transform", "hflip": "horizontal flip", "vflip": "vertical flip", "rot180": "rotation 180°"}
# decision colours + line styles (greyscale: solid / dashed / dotted)
DEC_STYLE = {"allied": ("#1a9641", "-"), "adversary": ("#d7191c", "--"), "uncertain": ("#f0a202", ":")}


def _sources() -> pd.DataFrame:
    return pd.read_csv(REL / "sources.csv")


def _source_of(sources: pd.DataFrame) -> dict:
    return dict(zip(sources["file"], sources["source"]))


def _orig(sources: pd.DataFrame) -> dict:
    return dict(zip(sources["file"], sources["original_filename"]))


# ----------------------------------------------------------------------------------------------- 1. dataset flow
def flow_numbers() -> dict:
    rep = json.loads((REL / "build_report.json").read_text(encoding="utf-8"))
    num = json.loads(NUMBERS.read_text(encoding="utf-8"))
    src = _sources()
    n = {}
    for tag in ("own", "fyp", "merge"):
        seen = rep[f"{tag}_files_seen"]
        unlab = rep.get(f"{tag}_unlabelled_skipped", 0)
        aug = rep.get(f"{tag}_augmented_copies_removed", 0)
        n[tag] = {"seen": seen, "unlabelled": unlab, "augmented": aug, "into_dedupe": seen - unlab - aug,
                  "final": int((src["source"] == tag).sum())}
    n["dup_removed"] = rep["exact_duplicates_removed"] + rep["near_duplicates_removed"]
    n["into_dedupe"] = sum(n[t]["into_dedupe"] for t in ("own", "fyp", "merge"))
    n["after_dedupe"] = n["into_dedupe"] - n["dup_removed"]
    n["capped"] = rep["dropped_by_class_cap"] + rep["dropped_by_other_cap"]
    n["final"] = n["after_dedupe"] - n["capped"]
    split = src.groupby("split").size().to_dict()
    n.update({f"{s}_full": int(split[s]) for s in ("train", "val", "test")})
    for s in ("val", "test"):
        cl = num["clean_lists"][s]
        n[f"{s}_clean"], n[f"{s}_removed"] = cl["clean"], cl["images"] - cl["clean"]
        assert cl["images"] == n[f"{s}_full"], (s, cl["images"], n[f"{s}_full"])
    assert n["final"] == len(src) == n["train_full"] + n["val_full"] + n["test_full"], "build report != sources.csv"
    assert (n["val_clean"], n["test_clean"]) == (2221, 1071), "clean counts differ from the paper"
    # images and classes contributed by the authors' collections (own + fyp)
    n["authors_final"] = n["own"]["final"] + n["fyp"]["final"]
    n["authors_share"] = n["authors_final"] / n["final"]
    names = yaml.safe_load((REL / "classes.yaml").read_text(encoding="utf-8"))["names"]
    by_class = collections.defaultdict(set)
    for split_dir in (REL / "labels").iterdir():
        for f in split_dir.glob("*.txt"):
            srcs = f.name.split("_")[0]
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    by_class[names[int(line.split()[0])]].add(srcs)
    n["authors_only_classes"] = sorted(c for c, s in by_class.items() if "merge" not in s)
    return n


def fig_dataset_flow():
    n = flow_numbers()
    # axis units: 14.2 x 7.2 on a 12 x 6.08 cm canvas (1 unit = 0.845 cm in both directions)
    fig = plt.figure(figsize=(WIDTH, WIDTH * 7.2 / 14.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 14.2), ax.set_ylim(0, 7.2), ax.axis("off")
    FT, FB, LH = 5.8, 5.3, 0.27  # title pt, body pt, body line height (axis units)

    def box(x, y, w, h, title, body="", authors=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#d0d0d0" if authors else "white", ec="black",
                                    lw=2.0 if authors else 0.7, zorder=2))
        n_lines = 1 + (body.count("\n") + 1 if body else 0)
        ty = y + h / 2 + (n_lines - 1) * LH / 2
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=FT, weight="bold", zorder=3)
        if body:
            ax.text(x + w / 2, ty - LH / 2, body, ha="center", va="top", fontsize=FB, linespacing=1.2, zorder=3)
        return (x, y, w, h)

    def arrow(p, q, label=None, dx=0.0, dy=0.06, ha="center", va="bottom", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=1))
        if label:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, ha=ha, va=va, fontsize=5.8,
                    weight="bold", zorder=4, linespacing=1.1, bbox=dict(fc="white", ec="none", pad=0.3))

    f = lambda v: f"{v:,}"  # noqa: E731
    own, fyp, mrg = n["own"], n["fyp"], n["merge"]
    A = box(0.0, 4.1, 3.0, 1.35, "Authors' collection A",
            f"{f(own['seen'])} files\n−{f(own['augmented'])} augmented copies\n−{own['unlabelled']} unlabelled",
            authors=True)
    B = box(0.0, 2.65, 3.0, 1.05, "Authors' collection B",
            f"{f(fyp['seen'])} files\n−{fyp['unlabelled']} unlabelled", authors=True)
    M = box(0.0, 0.85, 3.0, 1.35, "merge_dataset", f"SuperRadarOSINT\nCC BY 4.0\n{f(mrg['seen'])} files")
    D = box(4.35, 2.25, 1.95, 1.95, "Dedupe", f"MD5 + pHash\n(image and\nlargest box)\n−{f(n['dup_removed'])}")
    C = box(7.45, 2.55, 2.1, 1.35, "Class balance", f"per-class caps\n−{f(n['capped'])}")
    S = box(10.65, 2.65, 1.5, 1.15, "Split", "70/20/10")
    T = box(12.6, 4.5, 1.6, 0.85, "Train", f(n["train_full"]))
    R = box(7.2, 0.1, 2.6, 1.45, "Clean lists", "R1–R3 label-only rules\n"
                               f"val −{n['val_removed']}, test −{n['test_removed']}")
    V = box(12.6, 1.6, 1.6, 0.85, "Clean val", f(n["val_clean"]))
    E = box(12.6, 0.1, 1.6, 0.85, "Clean test", f(n["test_clean"]))

    def right(b, fy=0.5):
        return (b[0] + b[2], b[1] + b[3] * fy)

    def left(b, fy=0.5):
        return (b[0], b[1] + b[3] * fy)

    arrow(right(A), left(D, 0.85), f(own["into_dedupe"]), dx=-0.1, dy=0.1)
    arrow(right(B), left(D, 0.5), f(fyp["into_dedupe"]), dy=0.03)
    arrow(right(M), left(D, 0.15), f(mrg["into_dedupe"]), dx=-0.1, dy=0.02)
    arrow(right(D), left(C), f(n["after_dedupe"]))
    arrow(right(C), left(S), f(n["final"]))
    arrow((S[0] + S[2] * 0.5, S[1] + S[3]), left(T), f(n["train_full"]), dx=-0.5, dy=0.05, rad=-0.3)
    arrow((S[0] + S[2] * 0.3, S[1]), (R[0] + R[2], R[1] + R[3] * 0.85),
          f"val {f(n['val_full'])}\ntest {f(n['test_full'])}", dx=-0.2, dy=0.1, ha="right", va="center")
    arrow(right(R, 0.62), left(V), f(n["val_clean"]), dx=0.25, dy=0.08)
    arrow(right(R, 0.3), left(E), f(n["test_clean"]), dx=0.25, dy=0.05)
    k = len(n["authors_only_classes"])
    ax.text(0.02, 7.12, f"Authors' collection: {f(n['authors_final'])} images in the final set "
                        f"({100 * n['authors_share']:.1f}%); {k} classes found only here",
            ha="left", va="top", fontsize=6.2, style="italic", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="#d0d0d0", ec="black", lw=1.4))
    ax.add_patch(FancyArrowPatch((1.5, 6.6), (1.5, A[1] + A[3]), arrowstyle="-", lw=1.4, color="black",
                                 shrinkA=0, shrinkB=0))
    ax.text(14.15, 6.2, f"Final set: {f(n['final'])} images\ntrain {f(n['train_full'])}, val {f(n['val_full'])}, "
                        f"test {f(n['test_full'])}", ha="right", va="center", fontsize=FB, linespacing=1.2)
    save(fig, "fig_dataset_flow")
    return n


# ----------------------------------------------------------------------------------------------- 2. leakage examples
def leakage_pairs():
    a = pd.read_csv(RES / "leakage_audit.csv")
    src = _source_of(_sources())
    a = a[a["flagged"] == 1].copy()
    a["match_source"] = a["match_file"].map(src)
    ok = a[(a["source"] == "merge") & (a["match_source"] == "merge") & (a["match_split"] == "train")]
    rng = np.random.default_rng(SEED)
    rows, missing = [], []
    for t in TRANSFORMS:
        cand = ok[ok["variant"] == t].sort_values("file")
        if cand.empty:
            missing.append((t, int((a["variant"] == t).sum())))
            continue
        rows.append(cand.iloc[int(rng.integers(len(cand)))])
    return rows, missing


def fig_leakage_examples():
    rows, missing = leakage_pairs()
    h_row = 5.4 * CM
    fig, axes = plt.subplots(len(rows), 2, figsize=(WIDTH, h_row * len(rows) + 1.2 * CM), squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.32))
    for i, r in enumerate(rows):
        for j, (split, file) in enumerate((("train", r["match_file"]), (r["split"], r["file"]))):
            ax = axes[i, j]
            ax.imshow(Image.open(IMAGES / split / file).convert("RGB"))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{'train' if j == 0 else split}: {file}", fontsize=6.2, pad=2)
        axes[i, 0].set_ylabel(f"{TF_LABEL[r['variant']]}\nd_full = {int(r['d_full'])}, d_crop = {int(r['d_crop'])}",
                              fontsize=7)
    axes[0, 0].text(0.5, 1.32, "Train image", transform=axes[0, 0].transAxes, ha="center", fontsize=8,
                    weight="bold")
    axes[0, 1].text(0.5, 1.32, "Val/test image flagged by R1", transform=axes[0, 1].transAxes, ha="center",
                    fontsize=8, weight="bold")
    note = ("Hamming distances of the pHash of the whole image (d_full) and of the largest-box crop (d_crop) after "
            "applying the transform; R1 flags both ≤ 6.")
    if missing:
        note += " " + "; ".join(f"{TF_LABEL[t]}: none of its {k} flags has both images from merge_dataset"
                                for t, k in missing) + ", so no row is shown."
    fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=6.2, wrap=True)
    save(fig, "fig_leakage_examples")
    return rows, missing


# ----------------------------------------------------------------------------------------------- 3. qualitative
def _tau():
    num = json.loads(NUMBERS.read_text(encoding="utf-8"))
    t = num["abstention_tau"]["yolo26s_degrade"]
    weights = json.loads((RES / "yolo26s_degrade_matched_abstain.json").read_text(encoding="utf-8"))["weights"]
    return float(t["matched"]["0.10"]), float(t["det_conf"]), weights


def _gt(stem, w, h):
    p = MERGED / "labels" / "test" / f"{stem}.txt"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c, x, y, bw, bh = map(float, line.split()[:5])
            out.append((int(c), np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])))
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Predictor:
    def __init__(self):
        from ultralytics import YOLO
        self.tau, self.conf, weights = _tau()
        self.model = YOLO(weights)
        self.names = self.model.names
        self.aff = json.loads((MERGED / "class_affiliation.json").read_text(encoding="utf-8"))

    def __call__(self, path):
        r = self.model.predict(str(path), conf=self.conf, imgsz=640, half=True, verbose=False)[0]
        dets = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            name = self.names[int(c)]
            a = self.aff.get(name, "unknown")
            dec = a if (a in ("allied", "adversary") and s >= self.tau) else "uncertain"
            dets.append({"box": b, "name": name, "conf": float(s), "decision": dec})
        return dets


def _accepted_hits(gt, dets):
    """GT boxes covered (IoU >= 0.5) by an accepted (allied/adversary) detection."""
    acc = [d for d in dets if d["decision"] != "uncertain"]
    return sum(any(_iou(g, d["box"]) >= 0.5 for d in acc) for _, g in gt)


def _corr_path(corruption, stem):
    hits = list((ACORR / corruption / "2" / "images").glob(f"{stem}.*"))
    return hits[0] if hits else None


def pick_rows(pred: Predictor, n_rows: int = 4):
    """Fixed-seed selection, not hand-picked. Candidates: clean-test merge_dataset images whose GT boxes all have
    one class with a definite affiliation (allied/adversary), 1-3 boxes. Shuffle with SEED; each candidate gets a
    seen corruption and a held-out corruption drawn with the same generator. Take candidates in order, one per class.
    The first n_rows-1 rows are taken unconditionally; if none of them shows a degradation (a GT box accepted on the
    clean image but uncertain or missed under a corruption), the last row is the next new-class candidate that does;
    otherwise the last row is simply the next new-class candidate."""
    src = _sources()
    aff = pred.aff
    names = yaml.safe_load((MERGED / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    clean = {p.rsplit("/", 1)[-1] for p in (MERGED / "test_clean.txt").read_text(encoding="utf-8").split()}
    cand = []
    for file in sorted(src[(src["split"] == "test") & (src["source"] == "merge")]["file"]):
        if file not in clean:
            continue
        lines = [l for l in (MERGED / "labels" / "test" / f"{file.rsplit('.', 1)[0]}.txt").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        cls = {names[int(l.split()[0])] for l in lines}
        if 1 <= len(lines) <= 3 and len(cls) == 1 and aff.get(next(iter(cls))) in ("allied", "adversary"):
            cand.append((file, next(iter(cls))))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cand))
    seen_pick = rng.integers(len(SEEN_CORRUPTIONS), size=len(cand))
    held_pick = rng.integers(len(HELD_OUT_CORRUPTIONS), size=len(cand))
    rows, used = [], set()
    for k in order:
        file, cls = cand[k]
        if cls in used:
            continue
        stem = file.rsplit(".", 1)[0]
        cs, ch = SEEN_CORRUPTIONS[seen_pick[k]], HELD_OUT_CORRUPTIONS[held_pick[k]]
        paths = [IMAGES / "test" / file, _corr_path(cs, stem), _corr_path(ch, stem)]
        if any(p is None for p in paths):
            continue
        w, h = Image.open(paths[0]).size
        gt = _gt(stem, w, h)
        dets = [pred(p) for p in paths]
        hits = [_accepted_hits(gt, d) for d in dets]
        row = {"file": file, "cls": cls, "corruptions": ("clean", cs, ch), "paths": paths, "gt": gt, "dets": dets,
               "hits": hits, "degrades": hits[1] < hits[0] or hits[2] < hits[0]}
        last = len(rows) == n_rows - 1
        if last and not any(r["degrades"] for r in rows) and not row["degrades"]:
            continue
        rows.append(row)
        used.add(cls)
        if len(rows) == n_rows:
            break
    return rows


def fig_qualitative():
    pred = Predictor()
    rows = pick_rows(pred)
    fig, axes = plt.subplots(len(rows), 3, figsize=(WIDTH, 4.85 * CM * len(rows) + 1.6 * CM),
                             gridspec_kw=dict(wspace=0.03, hspace=0.2, left=0.045, right=0.998, top=0.94,
                                              bottom=0.08))
    for i, r in enumerate(rows):
        for j, (path, dets, corr) in enumerate(zip(r["paths"], r["dets"], r["corruptions"])):
            ax = axes[i, j]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlim(0, img.size[0]), ax.set_ylim(img.size[1], 0)
            for d in sorted(dets, key=lambda d: d["conf"]):
                col, ls = DEC_STYLE[d["decision"]]
                x0, y0, x1, y1 = d["box"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.1, ls=ls))
                tag = f"{d['name']} {d['conf']:.2f}" if d["decision"] != "uncertain" else \
                    f"uncertain: {d['name']} {d['conf']:.2f}"
                W, H = img.size  # keep the label inside the image: anchor on the side with room
                right_side = x0 > 0.55 * W
                ax.text(x1 if right_side else x0, y0 if y0 > 0.08 * H else y1, tag, fontsize=4.8, color="black",
                        va="bottom" if y0 > 0.08 * H else "top", ha="right" if right_side else "left",
                        clip_on=True, bbox=dict(fc="white", ec=col, lw=0.6, ls=ls, pad=0.6, alpha=0.9))
            if not dets:
                ax.text(0.5, 0.06, "no detection", transform=ax.transAxes, ha="center", fontsize=5.5,
                        bbox=dict(fc="white", ec="black", lw=0.5, pad=0.8))
            if j > 0:
                ax.set_title(f"{CORR_LABEL[corr]}" + ("$^{\\dagger}$" if corr in HELD_OUT_CORRUPTIONS else ""),
                             fontsize=6.3, pad=1.5)
        axes[i, 0].set_ylabel(f"GT: {r['cls']}", fontsize=7)
    for j, t in enumerate(("Clean", "Seen corruption\n(severity 2)", "Held-out corruption$^{\\dagger}$\n(severity 2)")):
        axes[0, j].text(0.5, 1.13, t, transform=axes[0, j].transAxes, ha="center", va="bottom", fontsize=7.2,
                        weight="bold", linespacing=1.1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DEC_STYLE[k][0], ls=DEC_STYLE[k][1], lw=1.3, label=lab) for k, lab in
               (("allied", "allied (accepted)"), ("adversary", "adversary (accepted)"),
                ("uncertain", "uncertain (abstained)"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.075),
               handlelength=2.6)
    fig.text(0.5, 0.03, f"YOLO26-s + degradation FT; detections with confidence ≥ {pred.conf:g}; accepted if the class "
                        f"has a definite affiliation\nand confidence ≥ τ = {pred.tau:.3f} (r* = 10 %, matched mode, "
                        f"chosen on clean val); otherwise uncertain.", ha="center", va="top", fontsize=5.8,
             linespacing=1.2)
    save(fig, "fig_qualitative")
    return rows, pred


# ----------------------------------------------------------------------------------------------- credits
def write_credits(leak_rows, leak_missing, qual_rows):
    orig = _orig(_sources())
    src = _source_of(_sources())
    out = ["Photo credits for the paper figures. Every photo is from merge_dataset by SuperRadarOSINT, CC BY 4.0,",
           "https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd (source = merge in",
           "dataset_release/sources.csv). Columns: figure, panel, our filename, original Roboflow filename.",
           "Generated by make_extra_figures.py.", "",
           "fig_dataset_flow: no photos.", ""]
    for r in leak_rows:
        for panel, f in (("train", r["match_file"]), (r["split"], r["file"])):
            assert src[f] == "merge", f
            out.append(f"fig_leakage_examples\t{r['variant']} / {panel}\t{f}\t{orig[f]}")
    for t, k in leak_missing:
        out.append(f"fig_leakage_examples\t{t}: no row (none of its {k} R1 flags has both images from merge_dataset)")
    out.append("")
    for r in qual_rows:
        assert src[r["file"]] == "merge", r["file"]
        out.append(f"fig_qualitative\tGT {r['cls']} (clean, {r['corruptions'][1]} s2, {r['corruptions'][2]} s2)\t"
                   f"{r['file']}\t{orig[r['file']]}")
    out += ["", "Corrupted panels are Aircraft-C images (corrupt.py, seed in data_derived/aircraft_c/manifest.json) "
                "derived from the listed photos."]
    CREDITS.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[extra] wrote {CREDITS}")


def main():
    n = fig_dataset_flow()
    print(f"[extra] flow: into dedupe {n['into_dedupe']}, after {n['after_dedupe']}, final {n['final']} "
          f"(train {n['train_full']}, val {n['val_full']}->{n['val_clean']}, test {n['test_full']}->{n['test_clean']});"
          f" authors {n['authors_final']} ({100 * n['authors_share']:.1f}%), authors-only classes "
          f"{len(n['authors_only_classes'])}: {', '.join(n['authors_only_classes'])}")
    leak_rows, leak_missing = fig_leakage_examples()
    for r in leak_rows:
        print(f"[extra] leakage {r['variant']}: train {r['match_file']} -> {r['split']} {r['file']} "
              f"(d_full {int(r['d_full'])}, d_crop {int(r['d_crop'])})")
    for t, k in leak_missing:
        print(f"[extra] leakage {t}: no all-merge pair ({k} flags) -> no row")
    qual_rows, pred = fig_qualitative()
    for r in qual_rows:
        summ = [f"{c}: {sum(d['decision'] != 'uncertain' for d in ds)} acc / "
                f"{sum(d['decision'] == 'uncertain' for d in ds)} unc, GT hit {h}/{len(r['gt'])}"
                for c, ds, h in zip(r["corruptions"], r["dets"], r["hits"])]
        print(f"[extra] qualitative {r['file']} GT {r['cls']}: " + " | ".join(summ))
    write_credits(leak_rows, leak_missing, qual_rows)


if __name__ == "__main__":
    main()
