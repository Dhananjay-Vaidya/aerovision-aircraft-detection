"""Train an ultralytics detector on cvs_dataset.

Examples (PowerShell, Kaggle and Colab alike):
  python train.py --model yolo26s.pt
  python train.py --model yolov8s.pt --aug degrade
  python train.py --model yolo26s.pt --aug degrade --from runs/yolo26s_none/weights/best.pt --epochs 30
  python train.py --model yolo26n.pt --epochs 1 --limit 64          # smoke test

Weights end up in <project>/<name>/weights/best.pt (default project: runs/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ROOT, SEEN_CORRUPTIONS, default_workers, merged_dataset, portable_data_yaml


def degrade_augmentations(p: float = 0.5):
    """Photometric degradations (boxes stay valid), exactly the Aircraft-C "seen" corruptions.
    One of them is applied with probability p. The held-out ones (contrast, rain) are never used."""
    import albumentations as A

    transforms = {
        "defocus_blur": A.Defocus(radius=(2, 9), alias_blur=(0.1, 0.5), p=1.0),
        "motion_blur": A.MotionBlur(blur_limit=(5, 21), p=1.0),
        "fog": A.RandomFog(fog_coef_range=(0.2, 0.6), alpha_coef=0.1, p=1.0),
        # low light = darker + gamma; brightness only (contrast_limit=0) so it never doubles as `contrast`
        "low_light": A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.2), contrast_limit=(0, 0), p=1.0),
                                A.RandomGamma(gamma_limit=(120, 250), p=1.0)]),
        "gaussian_noise": A.GaussNoise(std_range=(0.03, 0.15), p=1.0),
        "jpeg": A.ImageCompression(quality_range=(8, 40), p=1.0),
        "downscale": A.Downscale(scale_range=(0.2, 0.6), p=1.0),
    }
    assert tuple(transforms) == SEEN_CORRUPTIONS, "train-time degradations must equal the Aircraft-C seen set"
    return [A.OneOf(list(transforms.values()), p=p)]


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="ultralytics weights, e.g. yolov8s.pt or yolo26s.pt")
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--aug", choices=["none", "degrade"], default="none")
    ap.add_argument("--aug-p", type=float, default=0.5, help="probability of a degradation per image")
    ap.add_argument("--from", dest="from_weights", default=None, help="initialise from these weights (fine-tuning)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU for s models; use 32 on Kaggle/Colab")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--device", default=None, help="0, cpu, ... (default: auto)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use at most N images per split (smoke test)")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    name = args.name or f"{Path(args.model).stem}_{args.aug}" + ("_ft" if args.from_weights else "") \
        + (f"_limit{args.limit}" if args.limit else "")
    merged = merged_dataset(args.data)  # classes with < 50 train images -> other_aircraft (results/merged_classes.txt)
    data = portable_data_yaml(merged, Path(args.project) / "_data", args.limit, args.seed)
    extra = {"augmentations": degrade_augmentations(args.aug_p)} if args.aug == "degrade" else {}

    batch, workers = args.batch, args.workers
    while True:
        model = YOLO(args.from_weights or args.model)
        try:
            model.train(
                data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=batch, workers=workers,
                device=args.device, patience=args.patience, seed=args.seed, amp=True, project=args.project,
                name=name, exist_ok=True, plots=True, **extra,
            )
            break
        except Exception as e:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
            import torch

            if is_oom(e) and batch > 1:
                torch.cuda.empty_cache()
                batch //= 2
                print(f"[train.py] CUDA out of memory - retrying with batch={batch}", file=sys.stderr)
            elif "DataLoader worker" in str(e) and workers > 0:
                # Windows: each worker loads torch's CUDA DLLs; with a small page file they can die (WinError 1455)
                workers = 0
                print("[train.py] DataLoader worker died - retrying with workers=0", file=sys.stderr)
            else:
                raise

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"[train.py] done. best weights: {best}")
    return best


if __name__ == "__main__":
    main()
