"""Inference speed on the local GPU: ms/image and FPS (FP16, batch 1, 640 px).

  python speed.py --weights yolo26s.pt yolov8s.pt --out results/speed_pretrained.csv
  python speed.py --weights runs/yolo26s_degrade/weights/best.pt --out results/speed_trained.csv

Two modes per model, each with --warmup untimed and --runs timed iterations, synchronised with
torch.cuda.synchronize() around every timed call:
  forward     the fused FP16 network on a fixed 1x3x640x640 tensor (pure GPU compute)
  end_to_end  model.predict() on a real 640x640 test image: letterbox + upload + forward + post-processing
              (NMS for YOLOv8; YOLO26 is NMS-free) + Results objects
Also reported: ultralytics' own per-stage means (preprocess / inference / postprocess) from the end_to_end runs.
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, list_images


def timed(fn, warmup: int, runs: int, sync) -> list[float]:
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def summarise(ms: list[float]) -> dict:
    a = np.array(ms)
    mean = float(a.mean())
    return {"ms_mean": mean, "ms_std": float(a.std(ddof=1)), "ms_p50": float(np.percentile(a, 50)),
            "ms_p95": float(np.percentile(a, 95)), "fps": 1000.0 / mean}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--image", default=None, help="image for end_to_end (default: first test image)")
    ap.add_argument("--out", default=str(ROOT / "results" / "speed.csv"))
    args = ap.parse_args(argv)

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    dev = torch.device("cuda:0")
    sync = torch.cuda.synchronize
    torch.backends.cudnn.benchmark = True
    img_path = Path(args.image) if args.image else list_images(ROOT / "cvs_dataset" / "images" / "test")[0]
    img = cv2.imread(str(img_path))
    gpu = torch.cuda.get_device_name(0)

    rows = []
    for w in args.weights:
        name = Path(w).stem if Path(w).parent.name != "weights" else Path(w).parent.parent.name
        # forward: fused FP16 network on a fixed tensor
        net = YOLO(w).model.fuse(verbose=False).to(dev).half().eval()
        params, gflops = get_num_params(net), get_flops(net, args.imgsz)
        x = torch.rand(1, 3, args.imgsz, args.imgsz, device=dev, dtype=torch.half)
        with torch.inference_mode():
            ms = timed(lambda: net(x), args.warmup, args.runs, sync)
        rows.append({"model": name, "mode": "forward", **summarise(ms)})
        del net
        torch.cuda.empty_cache()

        # end to end: predict() on a real image
        model = YOLO(w)
        stages = []

        def predict():
            r = model.predict(img, imgsz=args.imgsz, half=True, device=0, batch=1, verbose=False)
            stages.append(r[0].speed)

        ms = timed(predict, args.warmup, args.runs, sync)
        stages = stages[args.warmup:]
        st = {f"ul_{k}_ms": statistics.fmean(s[k] for s in stages) for k in ("preprocess", "inference", "postprocess")}
        rows.append({"model": name, "mode": "end_to_end", **summarise(ms), **st})
        for r in rows[-2:]:
            r.update({"weights": w, "params_M": params / 1e6, "gflops": gflops, "imgsz": args.imgsz, "batch": 1,
                      "precision": "fp16", "warmup": args.warmup, "runs": args.runs, "gpu": gpu,
                      "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                      "python": platform.python_version()})
            print(f"[speed.py] {name:<20} {r['mode']:<10} {r['ms_mean']:6.2f} ms/img (p95 {r['ms_p95']:.2f})"
                  f"  {r['fps']:7.1f} FPS")
        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "mode", "ms_mean", "ms_std", "ms_p50", "ms_p95", "fps", "ul_preprocess_ms", "ul_inference_ms",
              "ul_postprocess_ms", "params_M", "gflops", "imgsz", "batch", "precision", "warmup", "runs", "gpu",
              "torch", "ultralytics", "python", "weights"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[speed.py] wrote {out}  (image: {img_path.name}, GPU: {gpu})")


if __name__ == "__main__":
    main()
