# AeroVision Live

A local FastAPI demo for the paper's detector, Aircraft-C corruptions and matched-risk abstention thresholds. No frontend build step or React.

## Setup (Windows 11 / Python 3.13)

From the repository root, activate your Python environment and install the project's dependencies, including its CUDA-enabled PyTorch build as described in the root requirements file:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

The requirements use `uvicorn[standard]` to include WebSocket support. If installing only the three web packages above into an existing environment, also run `pip install "uvicorn[standard]" imageio-ffmpeg`. `imageio-ffmpeg` supplies a browser-compatible H.264 encoder when OpenCV cannot encode H.264 directly. A system FFmpeg on PATH also works. Tests additionally need `pip install pytest httpx`.

Place **trained project checkpoints**, with the matching aircraft class names, in the gitignored `models/` folder:

```text
models/yolov8s.pt
models/yolo26s.pt
models/yolo26s_plainft.pt
models/yolo26s_degrade.pt
```

Only existing files appear in the selector. The default is `yolo26s_degrade`, falling back to the first available checkpoint. Generic COCO checkpoints with the same names will not recognize the paper's aircraft types and are not substitutes for trained weights.

Keep `dataset_release/class_affiliation.json` and `paper_assets/numbers.json` in place. The app reads the affiliation mapping with `common.load_affiliation`, uses `abstention_tau[model]["matched"][risk]`, and directly calls `corrupt.apply` for the nine degradations with seed 42. The displayed image is returned as a lossless PNG.

## Run

```powershell
python -m app.server
```

Open **http://localhost:8000**. One server process owns the model cache and the single background camera. Models load lazily once per name; inference is serialized across inputs and uses CUDA FP16 when available, otherwise CPU. Inference runs at 640px to suit a 4 GB GPU. Inference timing includes preprocessing, prediction and transfer back to CPU, but excludes first model loading and lock wait.

- **Image:** upload or drop a file; select a model, target risk and optional degradation, then detect. The canvas shows exactly the inference pixels with boxes.
- **Video file:** select a video and frame interval (default 2). Wait for processing, then play or download the annotated H.264 MP4. Only sampled frames are encoded, at source FPS / interval to preserve timing; audio is omitted. The side panel follows playback with per-frame counts. Video FPS in the panel is not measured live.
- **Webcam (browser):** allow browser camera permission. One frame is in flight at a time, capped at 10 FPS; the server also retains only the newest pending frame. Boxes are drawn over the corresponding captured frame. Changing model/risk stops the stream; restart to apply settings.
- **IP / RTSP camera:** enter a reachable `rtsp://`, `http://` or `https://` URL, or an integer local webcam index such as `0`. Start connects in a background thread; connection failures appear above the workspace. Only one source runs at a time. An MJPEG feed and polled inference summary are shown. Network open/read timeouts are five seconds where supported by OpenCV FFmpeg. Stop releases the source; switching tabs stops live capture.

Green solid boxes indicate allied decisions, red dashed boxes adversary decisions, and amber dotted boxes uncertainty. Ambiguous affiliations yield “shared airframe”; unknown/unmapped classes yield “other aircraft”; definite affiliations below τ yield “low confidence”. Detections below 0.25 are omitted. The chosen risk is a validation calibration target, not a guarantee for a new camera or scene.

The server binds to localhost. Browser webcam access works on localhost or HTTPS. This is a single-user local demo; keep it local. Temporary uploaded videos are deleted after processing. Generated MP4s remain in `app/outputs/` until you remove them; that folder and all video/weight files are gitignored. No uploads or camera credentials are committed.

## API

- `GET /api/models`: available names, matched τ at both risks, default name, and corruption list. Works without weights.
- `POST /api/detect`: multipart `file`, optional `model`, `risk` (0.05/0.10), `corruption` (`none` or a listed key), `severity` (1–3). Returns `image_size`, `inference_ms`, `detections`, counts, τ and `image_base64` / `image_mime`.
- `POST /api/video`: multipart `file`, `model`, `risk`, `every_n`. Returns a playable `/outputs/*.mp4` URL and summaries in `frames` (source frame index, timestamp, counts, inference ms).
- `WS /ws/stream?model=...&risk=0.10`: binary JPEG frames in, detection JSON out; pending stale frames are replaced. Errors use `{"error": "message"}`.
- `POST /api/camera/start`: JSON `{"source": "rtsp://...", "model": "yolo26s_degrade", "risk": 0.10}`; returns immediately while connecting.
- `POST /api/camera/stop`: stops the active source (an optional `{source}` body is accepted and ignored).
- `GET /api/camera/feed`: annotated MJPEG; `GET /api/camera/status`: running state, error and latest result.

## Tests

```powershell
python -m pytest app/test_app.py -q
```

Uses synthetic detections/images/video and mocked model inference. No GPU, real weights, or real camera is needed. The video test uses the bundled FFmpeg fallback if OpenCV lacks H.264 encoding.

## Screenshot placeholders

- Image tab: add a screenshot of your own image with annotated detections.
- Video tab: add a screenshot of the annotated player and frame counts.
- Live tabs: add browser-webcam and IP-camera screenshots using authorized sources.

No screenshot binaries are included.

Research prototype for decision support. Affiliation is inferred from aircraft type via a configurable table; outputs must be verified by a trained operator.
