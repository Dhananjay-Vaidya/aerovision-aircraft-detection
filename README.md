# aerovision-aircraft-detection
YOLO-based aircraft detection and affiliation classification with degradation-aware training, robustness evaluation, and confidence-based abstention for uncertain predictions.

## Dataset & licence

The dataset release is in [`dataset_release/`](dataset_release/): YOLO labels for 36 classes (after the
`other_aircraft` merge), train / val / test split lists (full and leakage-free *clean* lists), class affiliations,
per-image source records and the Aircraft-C manifest. See its [dataset card](dataset_release/README.md).

- **No images are included.** They remain the property of their original owners. Rebuild them from the three
  Roboflow sources with [`dataset_release/rebuild.md`](dataset_release/rebuild.md)
  (`build_dataset.py --download`, then `audit_leakage.py`).
- **Annotations, splits and metadata:** CC BY 4.0, © 2026 Dhananjay Vaidya and Monika Mangla
  ([`dataset_release/LICENSE`](dataset_release/LICENSE)).
- **merge_dataset** by SuperRadarOSINT
  ([Roboflow Universe](https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd)) is CC BY 4.0. Its
  content is redistributed only as labels, with attribution.
- **Code:** MIT ([`LICENSE`](LICENSE)).
- **Citation:** [`CITATION.cff`](CITATION.cff).
- **Takedown requests / contact:** dhananjayvaidya4154@gmail.com

Intended for research and decision support with a human in the loop, not for autonomous targeting.
