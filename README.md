# AeroVision: A Leakage-Audited Dataset and Corruption Benchmark for Robust Military Aircraft Identification (dataset labels, splits, code and results)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24608077.svg)](https://doi.org/10.5281/zenodo.24608077)

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
- **Citation:** see [Cite this](#cite-this) and [`CITATION.cff`](CITATION.cff).
- **Takedown requests / contact:** dhananjayvaidya4154@gmail.com

Intended for research and decision support with a human in the loop, not for autonomous targeting.

## Cite this

Version 1.0.0 is archived on Zenodo: [doi:10.5281/zenodo.24608077](https://doi.org/10.5281/zenodo.24608077).

```bibtex
@dataset{vaidya2026aerovision,
  author    = {Vaidya, Dhananjay and Mangla, Monika},
  title     = {AeroVision: A Leakage-Audited Dataset and Corruption Benchmark for Robust Military Aircraft Identification (dataset labels, splits, code and results)},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.24608077},
  url       = {https://doi.org/10.5281/zenodo.24608077}
}
```

Please also credit merge_dataset by SuperRadarOSINT (https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd, CC BY 4.0).
