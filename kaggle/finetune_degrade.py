# =====================================================================================================================
# CVS - fine-tune YOLO26s with the Aircraft-C "seen" degradations. ONE self-contained Kaggle cell (paste and run).
#
# Attach to the notebook (GPU T4/P100):
#   * a dataset containing cvs_dataset/ (folder or Kaggle-extracted zip)  -> found by locating images/train
#   * a dataset containing the clean-trained weights best.pt              -> found by searching for best.pt
# Output (in /kaggle/working):
#   runs/yolo26s_degrade/weights/best.pt, yolo26s_degrade_test_results.json, yolo26s_degrade_runs.zip
#
# What it does
#   1. copies cvs_dataset fresh to /kaggle/working/cvs_dataset
#   2. class merge: classes with < 50 train images -> other_aircraft (same ids; merged classes have no boxes)
#   3. fine-tunes for 30 epochs; each TRAIN image gets, with p=0.5, one of the 7 seen corruptions
#      (defocus_blur, motion_blur, fog, low_light, gaussian_noise, jpeg, downscale) at a random severity 1-3,
#      using the exact functions/parameters of corrupt.py. Val/test images are never corrupted (asserted).
#   4. evaluates on the clean test split and zips the run
#
# Local smoke test (Windows/Linux): set CVS_INPUT, CVS_WORK, CVS_EPOCHS=1, CVS_LIMIT=64 and run
#   python kaggle/finetune_degrade.py
# =====================================================================================================================
import json
import os
import random
import shutil
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("CVS_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("CVS_WORK", "/kaggle/working"))
EPOCHS = int(os.environ.get("CVS_EPOCHS", 30))
LIMIT = int(os.environ.get("CVS_LIMIT", 0)) or None        # images per split, smoke tests only
BATCH = int(os.environ.get("CVS_BATCH", 32))
WORKERS = int(os.environ.get("CVS_WORKERS", 2 if os.name == "nt" else 4))
WEIGHTS = os.environ.get("CVS_WEIGHTS")                     # optional explicit path to best.pt
RUN_NAME = "yolo26s_degrade"
P_DEGRADE = 0.5
MIN_TRAIN_IMAGES = 50
OTHER_CLASS = "other_aircraft"
ULTRALYTICS_VERSION = "8.4.219"

try:
    import ultralytics
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise ImportError
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS_VERSION}"], check=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------
# Corruptions: copied VERBATIM from corrupt.py (seen group only). Do not edit here without editing corrupt.py.
# ---------------------------------------------------------------------------------------------------------------------
SEEN_CORRUPTIONS = ("defocus_blur", "motion_blur", "fog", "low_light", "gaussian_noise", "jpeg", "downscale")
SEVERITIES = (1, 2, 3)
PARAMS = {
    "defocus_blur": (3, 6, 9),                      # disk radius, px at 640
    "motion_blur": (9, 17, 27),                     # kernel length, px at 640
    "fog": (0.35, 0.55, 0.75),                      # haze strength
    "low_light": ((0.55, 1.4, 0.01), (0.35, 1.8, 0.02), (0.18, 2.2, 0.03)),  # (gain, gamma, noise std)
    "gaussian_noise": (0.06, 0.12, 0.20),          # noise std (fraction of 255)
    "jpeg": (30, 15, 7),                            # JPEG quality
    "downscale": (0.5, 0.3, 0.18),                  # downscale factor (area down, nearest up)
}


def _scale(img):
    return max(img.shape[:2]) / 640.0


def _to_u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gaussian_noise(img, s, rng):
    x = img.astype(np.float32) / 255.0
    return _to_u8(x + rng.normal(0, PARAMS["gaussian_noise"][s - 1], x.shape).astype(np.float32))


def defocus_blur(img, s, rng):
    r = max(1, int(round(PARAMS["defocus_blur"][s - 1] * _scale(img))))
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1, lineType=cv2.LINE_AA)
    return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)


def _line_kernel(L, angle):
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    return k / max(k.sum(), 1e-6)


def motion_blur(img, s, rng):
    L = max(3, int(round(PARAMS["motion_blur"][s - 1] * _scale(img))))
    return cv2.filter2D(img, -1, _line_kernel(L, float(rng.uniform(0, 180))), borderType=cv2.BORDER_REFLECT)


def _smooth_field(h, w, rng):
    """Multi-octave value noise in [0, 1] (cheap stand-in for a plasma fractal)."""
    field = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for cells in (2, 4, 8, 16):
        g = rng.random((cells + 1, cells + 1)).astype(np.float32)
        field += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    field /= total
    return (field - field.min()) / max(field.max() - field.min(), 1e-6)


def fog(img, s, rng):
    a = PARAMS["fog"][s - 1]
    x = img.astype(np.float32) / 255.0
    m = (a * (0.6 + 0.4 * _smooth_field(*img.shape[:2], rng)))[..., None]
    return _to_u8(x * (1 - m) + 0.82 * m)


def low_light(img, s, rng):
    gain, gamma, sd = PARAMS["low_light"][s - 1]
    x = (img.astype(np.float32) / 255.0) ** gamma * gain
    return _to_u8(x + rng.normal(0, sd, x.shape).astype(np.float32))


def jpeg_bytes(img, s):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, PARAMS["jpeg"][s - 1]])
    assert ok
    return buf


def jpeg(img, s, rng):
    return cv2.imdecode(jpeg_bytes(img, s), cv2.IMREAD_COLOR)


def downscale(img, s, rng):
    h, w = img.shape[:2]
    f = PARAMS["downscale"][s - 1]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


FUNCS = {n: globals()[n] for n in SEEN_CORRUPTIONS}


def seed_for(seed: int, corruption: str, severity: int, name: str) -> int:
    return zlib.crc32(f"{seed}|{corruption}|{severity}|{name}".encode())


# ---------------------------------------------------------------------------------------------------------------------
# Train-only degradation hook for ultralytics 8.4.219
#   DetectionTrainer.build_dataset -> build_yolo_dataset -> YOLODataset(augment=mode == "train").
#   YOLODataset.build_transforms() is also re-run by close_mosaic(), so the hook lives there and survives it.
# ---------------------------------------------------------------------------------------------------------------------
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402


class SeenDegradation:
    """With probability p, apply one random seen corruption at a random severity to labels['img'] (uint8 BGR)."""

    def __init__(self, p: float = P_DEGRADE):
        self.p = p

    def __call__(self, labels):
        if random.random() < self.p:  # python/numpy RNGs are re-seeded per DataLoader worker by ultralytics
            name, sev = random.choice(SEEN_CORRUPTIONS), random.choice(SEVERITIES)
            rng = np.random.default_rng(random.getrandbits(64))
            labels["img"] = np.ascontiguousarray(FUNCS[name](labels["img"], sev, rng))
        return labels


class DegradeYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:  # train split only; val/test datasets have augment=False
            transforms.insert(-1, SeenDegradation(P_DEGRADE))  # photometric only, before Format
        return transforms


class DegradeTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            assert type(ds) is YOLODataset, f"unexpected dataset class {type(ds)}"
            ds.__class__ = DegradeYOLODataset
            ds.transforms = ds.build_transforms(hyp=self.args)
        return ds


def _has_degrade(dataset) -> bool:
    return any(isinstance(t, SeenDegradation) for t in dataset.transforms.transforms)


def _check_train_start(trainer):
    assert _has_degrade(trainer.train_loader.dataset), "train set is missing the degradation"
    assert not _has_degrade(trainer.test_loader.dataset), "val set must never be degraded"
    print("[finetune] check ok: degradation on train only")


def _check_epoch_end(trainer):
    assert _has_degrade(trainer.train_loader.dataset), "degradation lost after close_mosaic"


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset(root: Path) -> Path:
    hits = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        p = Path(dirpath)
        if (p / "images" / "train").is_dir():
            hits.append(p)
            dirnames[:] = []
    if not hits:
        raise SystemExit(f"no folder with images/train under {root}")
    hits.sort(key=lambda p: (not (p / "data.yaml").exists(), len(p.parts)))
    print("[finetune] dataset candidates:", [str(h) for h in hits], "-> using", hits[0])
    return hits[0]


def find_weights(root: Path) -> Path:
    if WEIGHTS:
        return Path(WEIGHTS)
    hits = sorted(Path(d) / "best.pt" for d, _, files in os.walk(root, followlinks=True) if "best.pt" in files)
    if not hits:
        raise SystemExit(f"no best.pt under {root}; attach the weights dataset")
    hits.sort(key=lambda p: "yolo26s" not in str(p).lower())
    print("[finetune] best.pt candidates:", [str(h) for h in hits], "-> using", hits[0])
    return hits[0]


def load_names(data_yaml: Path):
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def prepare_dataset(src: Path, dst: Path):
    """Fresh copy + class merge (same rule as common.class_merge_map). Returns (data.yaml, merged class names)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    names = load_names(dst / "data.yaml")
    other = names.index(OTHER_CLASS)
    counts = Counter()
    for f in (dst / "labels" / "train").glob("*.txt"):
        counts.update({int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()})
    merged = [i for i in range(len(names)) if i != other and counts[i] < MIN_TRAIN_IMAGES]
    for f in (dst / "labels").rglob("*.txt"):
        lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
        f.write_text("".join(" ".join([str(other if int(p[0]) in merged else p[0]), *p[1:]]) + "\n" for p in lines))
    for c in (dst / "labels").glob("*.cache"):
        c.unlink()
    d = {"path": dst.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
         "nc": len(names), "names": names}
    if LIMIT:  # smoke test: seeded subsets as image lists
        rng = random.Random(0)
        for split in ("train", "val", "test"):
            imgs = sorted(p for p in (dst / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
            imgs = sorted(rng.sample(imgs, min(LIMIT, len(imgs))))
            lst = dst / f"{split}_limit{LIMIT}.txt"
            lst.write_text("\n".join(p.resolve().as_posix() for p in imgs) + "\n")
            d[split] = lst.resolve().as_posix()
    out = dst / "data_merged.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    merged_names = [names[i] for i in merged]
    print(f"[finetune] merged into {OTHER_CLASS}:", ", ".join(f"{n} ({counts[names.index(n)]})" for n in merged_names))
    return out, names, merged_names


# ---------------------------------------------------------------------------------------------------------------------
def main():
    WORK.mkdir(parents=True, exist_ok=True)
    data_yaml, names, merged = prepare_dataset(find_dataset(INPUT_ROOT), WORK / "cvs_dataset")
    weights = find_weights(INPUT_ROOT)

    model = YOLO(str(weights))
    model.add_callback("on_train_start", _check_train_start)
    model.add_callback("on_train_epoch_end", _check_epoch_end)
    batch = BATCH
    while True:
        try:
            model.train(trainer=DegradeTrainer, data=str(data_yaml), epochs=EPOCHS, imgsz=640, batch=batch,
                        workers=WORKERS, project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True, seed=0,
                        amp=True, plots=True, patience=EPOCHS)
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch <= 2:
                raise
            import torch
            torch.cuda.empty_cache()
            batch //= 2
            print(f"[finetune] CUDA OOM -> retrying with batch={batch}")
            model = YOLO(str(weights))
            model.add_callback("on_train_start", _check_train_start)
            model.add_callback("on_train_epoch_end", _check_epoch_end)

    run_dir = WORK / "runs" / RUN_NAME
    best = run_dir / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=640, batch=16, conf=0.001, half=True,
                            plots=False, project=str(WORK / "runs"), name=f"{RUN_NAME}_test", exist_ok=True,
                            workers=WORKERS)
    test_entry = Path(yaml.safe_load(data_yaml.read_text())["test"])
    test_src = test_entry if test_entry.is_absolute() else data_yaml.parent / test_entry
    n_test = (sum(1 for l in test_src.read_text().splitlines() if l.strip()) if test_src.suffix == ".txt"
              else sum(p.suffix.lower() in IMG_EXTS for p in test_src.iterdir()))
    results = {
        "model": RUN_NAME, "init_weights": str(weights), "epochs": EPOCHS, "p_degrade": P_DEGRADE,
        "degradations": list(SEEN_CORRUPTIONS), "severities": list(SEVERITIES), "limit": LIMIT,
        "split": "test (clean)", "n_test_images": n_test, "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class_AP50": {names[c]: float(m.box.ap50[k]) for k, c in enumerate(m.box.ap_class_index)},
        "merged_classes": merged, "merged_into": OTHER_CLASS, "min_train_images": MIN_TRAIN_IMAGES,
    }
    out_json = WORK / f"{RUN_NAME}_test_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    shutil.copy(out_json, run_dir / out_json.name)
    archive = shutil.make_archive(str(WORK / f"{RUN_NAME}_runs"), "zip", WORK / "runs")
    print(f"[finetune] test mAP50={results['mAP50']:.4f} mAP50-95={results['mAP50_95']:.4f} "
          f"P={results['precision']:.4f} R={results['recall']:.4f}")
    print(f"[finetune] wrote {out_json} and {archive}")


if __name__ == "__main__":  # also true inside a Jupyter/Kaggle cell
    main()
