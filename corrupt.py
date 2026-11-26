"""Build Aircraft-C: deterministic photometric corruptions of the test split.

9 corruptions x 3 severities.
  seen      (same kinds as train.py --aug degrade): defocus_blur, motion_blur, fog, low_light, gaussian_noise,
                                                    jpeg, downscale
  held_out  (never used in training):               contrast, rain
Boxes are unchanged (no geometric change). Labels come from the merged dataset view (classes with < 50 training
images -> other_aircraft, see common.class_merge_map).
Layout:  <out>/<corruption>/<severity>/images/*.jpg  and  .../labels/*.txt   (+ <out>/manifest.json)

  python corrupt.py                                          # full test split -> data_derived/aircraft_c
  python corrupt.py --limit 64 --out data_derived/aircraft_c_smoke

The output folder is deleted and rebuilt on every run. Every image's randomness is seeded from
(seed, corruption, severity, file name), so reruns are byte-identical. Images are stored as JPEG q=95,
except `jpeg`, whose own encoding is stored directly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import CORRUPTION_GROUP, HELD_OUT_CORRUPTIONS, ROOT, SEEN_CORRUPTIONS, default_workers, list_images, \
    merged_dataset

SEVERITIES = (1, 2, 3)
# Parameters are tuned for 640 px images (the dataset); spatial ones scale with image size.
PARAMS = {
    # seen
    "defocus_blur": (3, 6, 9),                      # disk radius, px at 640
    "motion_blur": (9, 17, 27),                     # kernel length, px at 640
    "fog": (0.35, 0.55, 0.75),                      # haze strength
    "low_light": ((0.55, 1.4, 0.01), (0.35, 1.8, 0.02), (0.18, 2.2, 0.03)),  # (gain, gamma, noise std)
    "gaussian_noise": (0.06, 0.12, 0.20),          # noise std (fraction of 255)
    "jpeg": (30, 15, 7),                            # JPEG quality
    "downscale": (0.5, 0.3, 0.18),                  # downscale factor (area down, nearest up)
    # held out
    "contrast": (0.5, 0.3, 0.15),                   # contrast factor
    "rain": ((0.0015, 12, 0.55), (0.003, 18, 0.65), (0.006, 26, 0.75)),  # (drops per px, streak length px, opacity)
}
CORRUPTIONS = SEEN_CORRUPTIONS + HELD_OUT_CORRUPTIONS
assert set(PARAMS) == set(CORRUPTIONS)


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


def contrast(img, s, rng):
    c = PARAMS["contrast"][s - 1]
    x = img.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1), keepdims=True)
    return _to_u8((x - mean) * c + mean)


def rain(img, s, rng):
    """Bright, slanted, motion-blurred streaks over a slightly darkened, hazier scene."""
    density, length, opacity = PARAMS["rain"][s - 1]
    h, w = img.shape[:2]
    sc = _scale(img)
    L = max(3, int(round(length * sc)))
    angle = float(rng.uniform(-20, 20))                 # degrees from vertical
    n = int(density * h * w)
    layer = np.zeros((h, w), np.float32)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    layer[ys, xs] = rng.uniform(0.6, 1.0, n).astype(np.float32)
    layer = cv2.filter2D(layer, -1, _line_kernel(L, 90 + angle)) * L     # streaks keep peak brightness
    layer = cv2.GaussianBlur(np.clip(layer, 0, 1), (0, 0), 0.6 * sc + 0.1)
    x = img.astype(np.float32) / 255.0
    x = x * (1 - 0.12 * s) + 0.08 * s * 0.75           # darker, hazier sky
    a = (opacity * layer)[..., None]
    return _to_u8(x * (1 - a) + 0.9 * a)


FUNCS = {n: globals()[n] for n in CORRUPTIONS}


def seed_for(seed: int, corruption: str, severity: int, name: str) -> int:
    return zlib.crc32(f"{seed}|{corruption}|{severity}|{name}".encode())


def apply(img: np.ndarray, corruption: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """img: HxWx3 uint8 (BGR or RGB - all corruptions are channel-symmetric)."""
    return FUNCS[corruption](img, severity, rng)


def _process(img_path: Path, out: Path, labels_dir: Path, seed: int, quality: int):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    lbl = labels_dir / (img_path.stem + ".txt")
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            d = out / c / str(s)
            dst = d / "images" / (img_path.stem + ".jpg")
            if c == "jpeg":
                jpeg_bytes(img, s).tofile(str(dst))
            else:
                rng = np.random.default_rng(seed_for(seed, c, s, img_path.name))
                cv2.imwrite(str(dst), apply(img, c, s, rng), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if lbl.exists():
                shutil.copyfile(lbl, d / "labels" / lbl.name)
            else:
                (d / "labels" / lbl.name).write_text("")
    return img_path.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "data_derived" / "aircraft_c"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, default_workers() * 2))
    args = ap.parse_args(argv)

    merged_yaml = merged_dataset(args.data)          # merged labels (shared rule)
    d = yaml.safe_load(merged_yaml.read_text(encoding="utf-8"))
    base = merged_yaml.parent
    img_dir = base / d[args.split]
    labels_dir = base / str(d[args.split]).replace("images", "labels")
    imgs = list_images(img_dir)[: args.limit] if args.limit else list_images(img_dir)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            (out / c / str(s) / "images").mkdir(parents=True, exist_ok=True)
            (out / c / str(s) / "labels").mkdir(parents=True, exist_ok=True)

    fn = partial(_process, out=out, labels_dir=labels_dir, seed=args.seed, quality=args.quality)
    with Pool(args.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(fn, imgs, chunksize=8), 1):
            if n % 200 == 0 or n == len(imgs):
                print(f"[corrupt.py] {n}/{len(imgs)} images x {len(CORRUPTIONS) * len(SEVERITIES)} conditions")
    (out / "manifest.json").write_text(json.dumps({
        "source_split": args.split, "source_data": str(Path(args.data).resolve()), "labels": "merged (common.py)",
        "n_images": len(imgs), "seed": args.seed, "severities": list(SEVERITIES),
        "corruptions": [{"name": c, "group": CORRUPTION_GROUP[c], "params": PARAMS[c]} for c in CORRUPTIONS],
        "storage_jpeg_quality": args.quality, "limit": args.limit,
    }, indent=2), encoding="utf-8")
    print(f"[corrupt.py] wrote {out}")


if __name__ == "__main__":
    main()
