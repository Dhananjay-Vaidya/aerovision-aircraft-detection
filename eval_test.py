"""Clean-image test-split metrics for one checkpoint, as JSON in the same format as the Kaggle results file.

  python eval_test.py --weights path/to/best.pt                          # -> results/<model>_clean_test.json
  python eval_test.py --weights path/to/best.pt --split-set full         # -> results/<model>_full_test.json

Labels are the merged ones (common.merged_dataset); --split-set clean uses cvs_dataset/test_clean.txt (no
flipped/rotated near-copies of train/val images), full uses every test image. Ultralytics val() on split "test",
imgsz 640, batch 16, conf 0.001, FP16, as in kaggle/finetune_degrade.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common import MIN_TRAIN_IMAGES, OTHER_CLASS, ROOT, SPLIT_SETS, class_merge_map, default_workers, load_names, \
    merged_dataset, split_images


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="output name (default: run folder, e.g. yolo26s)")
    ap.add_argument("--data", default=str(ROOT / "cvs_dataset" / "data.yaml"))
    ap.add_argument("--split-set", choices=SPLIT_SETS, default="clean")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    args.name = args.name or (weights.parent.parent.name if weights.parent.name == "weights" else weights.stem)
    data_yaml = merged_dataset(args.data, split_set=args.split_set)
    names = load_names(data_yaml)
    _, merged, _ = class_merge_map(args.data)
    n_test = len(split_images(data_yaml, "test"))
    model = YOLO(str(weights))
    n_params, gflops = get_num_params(model.model), get_flops(model.model, args.imgsz)
    m = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, conf=0.001, half=True,
                  plots=False, verbose=False, workers=args.workers, project=str(ROOT / "runs" / "eval"),
                  name=f"{args.name}_{args.split_set}_test", exist_ok=True)
    ckpt_args = {}
    run_args = weights.parent.parent / "args.yaml"  # training args saved next to weights/
    if run_args.exists():
        ckpt_args = yaml.safe_load(run_args.read_text(encoding="utf-8"))
    res = {
        "model": ckpt_args.get("model", weights.name),
        "epochs": ckpt_args.get("epochs"),
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class_AP50": {names[c]: float(m.box.ap50[k]) for k, c in enumerate(m.box.ap_class_index)},
        "merged_classes": merged,
        # extra fields (not in the Kaggle file)
        "params": int(n_params), "params_M": round(n_params / 1e6, 3), "gflops": round(float(gflops), 2),
        "split_set": args.split_set, "n_test_images": n_test, "weights": str(weights),
        "merged_into": OTHER_CLASS, "min_train_images": MIN_TRAIN_IMAGES,
    }
    out = Path(args.out) if args.out else ROOT / "results" / f"{args.name}_{args.split_set}_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[eval_test.py] {args.split_set}: {n_test} test images  mAP50={res['mAP50']:.4f} "
          f"mAP50-95={res['mAP50_95']:.4f}  P={res['precision']:.4f} R={res['recall']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
