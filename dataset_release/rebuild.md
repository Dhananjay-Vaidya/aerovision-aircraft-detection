# Rebuilding the images

This release contains labels only. To recreate the image folders with the same file names:

1. **Get the code and a Roboflow API key.**
   ```bash
   git clone https://github.com/Dhananjay-Vaidya/aerovision-aircraft-detection
   cd aerovision-aircraft-detection
   python -m venv venv && . venv/bin/activate     # Windows: venv\Scripts\activate
   pip install -r requirements.txt roboflow imagehash pillow
   export ROBOFLOW_API_KEY=...                     # from your Roboflow account; keep it in a gitignored .env
   ```
2. **Download the three sources and build the dataset.** `--download` fetches the YOLO exports of
   - https://universe.roboflow.com/dhananjay/aircraft-detection-rtt9x (version 2)
   - https://universe.roboflow.com/dhananjay/fyp-uagdf (version 32)
   - https://universe.roboflow.com/superradarosint/merge_dataset-9nrmd (version 1, CC BY 4.0, SuperRadarOSINT)

   into `raw/`, then dedupes, balances, splits (seed 42) and writes `cvs_dataset/`:
   ```bash
   python build_dataset.py --download
   ```
   The build then runs the flip/rotation-invariant leakage check and the label-geometry rules and writes
   `cvs_dataset/val_clean.txt`, `test_clean.txt` and `data_clean.yaml`. No label is corrected.
3. **Run the leakage audit** (reports, and the same clean lists):
   ```bash
   python audit_leakage.py
   ```
4. **Check the result against this release.** Every row of `sources.csv` must map our file name to the same
   original Roboflow file:
   ```python
   import csv
   rel = {r["file"]: r["original_filename"] for r in csv.DictReader(open("dataset_release/sources.csv"))}
   got = {r["file"]: r["original_path"].replace("\\", "/").rsplit("/", 1)[-1]
          for r in csv.DictReader(open("cvs_dataset/manifest.csv"))}
   print("identical" if rel == got else f"{sum(rel.get(k) != v for k, v in got.items())} files differ")
   ```
   If the build differs (for example because a source export changed), use `sources.csv` instead: copy
   `raw/<source>/<export_path>` to `images/<split>/<file>` for every row and use this release's labels.
5. **Optional: the merged view and Aircraft-C.**
   ```bash
   python corrupt.py        # Aircraft-C from the test split (seed 1234), see aircraft_c_manifest.json
   ```
   Training and evaluation scripts build the merged-label view (`data_derived/merged/`) themselves.

Notes
- Source projects can change or become private; the versions above are the ones used. If a version is no longer
  downloadable, contact us (see README.md).
- `build_dataset.py` caches image hashes in `raw/hash_cache.tsv`; delete it to recompute from scratch.
