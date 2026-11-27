"""Run with python -m app.server. Inference is serialized for small GPUs."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, suppress
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common import ROOT, load_affiliation
from corrupt import CORRUPTIONS, apply

APP_DIR = Path(__file__).resolve().parent
OUTPUTS = APP_DIR / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
MODELS_DIR = ROOT / 'models'
MODEL_NAMES = ('yolov8s', 'yolo26s', 'yolo26s_plainft', 'yolo26s_degrade')
DEFAULT_MODEL = 'yolo26s_degrade'
INFERENCE_LOCK = threading.Lock()
MODEL_CACHE = {}


def thresholds():
    try:
        return json.loads((ROOT / 'paper_assets/numbers.json').read_text(encoding='utf-8'))['abstention_tau']
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(503, 'Calibration metadata is missing or invalid: paper_assets/numbers.json') from exc


def available_models():
    return [name for name in MODEL_NAMES if (MODELS_DIR / f'{name}.pt').is_file()]


def settings(model=None, risk=0.10):
    available = available_models()
    model = model or (DEFAULT_MODEL if DEFAULT_MODEL in available else next(iter(available), DEFAULT_MODEL))
    if model not in available:
        raise HTTPException(404, f'Model {model!r} is unavailable. Put trained weights in models/{model}.pt.')
    if risk not in (0.05, 0.10):
        raise HTTPException(422, 'Risk must be 0.05 or 0.10.')
    try:
        tau = float(thresholds()[model]['matched'][f'{risk:.2f}'])
        if not math.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(503, f'No valid matched calibration threshold for {model}.') from exc
    return model, tau


def decide(class_name, conf, affiliation_table, tau):
    """Apply affiliation abstention after the detector's 0.25 confidence filter."""
    affiliation = affiliation_table.get(class_name, 'unknown')
    if affiliation == 'ambiguous':
        return affiliation, 'uncertain', 'shared airframe'
    if affiliation not in ('allied', 'adversary'):
        return affiliation, 'uncertain', 'other aircraft'
    if conf < tau:
        return affiliation, 'uncertain', 'low confidence'
    return affiliation, affiliation, None


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(422, 'Unreadable image. Upload a JPEG, PNG, WebP or BMP image.')
    return img


def encode(img, extension='.jpg'):
    ok, data = cv2.imencode(extension, img)
    if not ok:
        raise HTTPException(500, 'Could not encode the image.')
    return data.tobytes()


def detect(img, model=None, risk=0.10):
    model, tau = settings(model, risk)
    with INFERENCE_LOCK:
        try:
            import torch
            from ultralytics import YOLO
            if model not in MODEL_CACHE:
                MODEL_CACHE[model] = YOLO(str(MODELS_DIR / f'{model}.pt'))
            gpu = torch.cuda.is_available()
            started = time.perf_counter()
            result = MODEL_CACHE[model].predict(img, conf=0.25, imgsz=640, device=0 if gpu else 'cpu',
                                               half=gpu, verbose=False)[0]
            boxes = result.boxes.data.cpu().tolist()
            elapsed = (time.perf_counter() - started) * 1000
            table = load_affiliation(ROOT / 'dataset_release/class_affiliation.json')
            detections = []
            for x1, y1, x2, y2, conf, cls in boxes:
                if conf < 0.25:
                    continue
                name = result.names[int(cls)]
                affiliation, decision, reason = decide(name, conf, table, tau)
                detections.append(dict(box=[x1, y1, x2, y2], **{'class': name}, conf=conf,
                                       affiliation=affiliation, decision=decision, reason=reason))
        except Exception as exc:
            raise HTTPException(503, 'Inference failed. Check the trained weights, class metadata and available GPU memory.') from exc
    h, w = img.shape[:2]
    return dict(image_size={'width': w, 'height': h}, inference_ms=round(elapsed, 2),
                detections=detections, model=model, risk=risk, tau=tau,
                counts={key: sum(d['decision'] == key for d in detections)
                        for key in ('allied', 'adversary', 'uncertain')})


def annotate(img, result):
    img = img.copy()
    for d in result['detections']:
        x1, y1, x2, y2 = map(int, d['box'])
        color = {'allied': (120, 220, 65), 'adversary': (95, 85, 245), 'uncertain': (60, 190, 245)}[d['decision']]
        step, length = {'allied': (1, 1), 'adversary': (16, 10), 'uncertain': (7, 2)}[d['decision']]
        for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            distance = max(1, int(math.dist(a, b)))
            if d['decision'] == 'allied':
                cv2.line(img, a, b, color, 2)
            else:
                for start in range(0, distance, step):
                    p = tuple(int(a[i] + (b[i] - a[i]) * start / distance) for i in (0, 1))
                    q = tuple(int(a[i] + (b[i] - a[i]) * min(start + length, distance) / distance) for i in (0, 1))
                    cv2.line(img, p, q, color, 2)
        # Draw separators explicitly: OpenCV's built-in font has no middle-dot glyph.
        cursor, baseline = max(0, x1), max(18, y1 - 8)
        for index, part in enumerate((d['class'], f"{d['conf']:.0%}", d['decision'])):
            if index:
                cv2.circle(img, (cursor + 7, baseline - 5), 1, color, -1)
                cursor += 16
            cv2.putText(img, part, (cursor, baseline), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            cursor += cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)[0][0]
    return img


class CameraRequest(BaseModel):
    source: str | int
    model: str | None = None
    risk: float = 0.10


class Camera:
    def __init__(self):
        self.control = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.jpeg = None
        self.result = None
        self.error = None
        self.sequence = 0

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise HTTPException(503, 'Camera is still closing. Please retry shortly.')
        self.thread = None
        self.jpeg = None

    def run(self, source, model, risk):
        cap = None
        try:
            if isinstance(source, int):
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if not cap.isOpened():
                raise ValueError('Cannot open camera. Check its address, credentials and network connection.')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Camera stopped delivering frames. Check the source and restart.')
                self.result = detect(frame, model, risk)
                self.jpeg = encode(annotate(frame, self.result))
                self.sequence += 1
        except Exception as exc:
            self.error = exc.detail if isinstance(exc, HTTPException) else str(exc)
        finally:
            if cap is not None:
                cap.release()


camera = Camera()


@asynccontextmanager
async def lifespan(app):
    yield
    await asyncio.to_thread(camera.stop)


app = FastAPI(title='AeroVision Live', lifespan=lifespan)


@app.get('/api/models')
def models():
    calibration = thresholds()
    names = available_models()
    return {'models': [{'name': n, 'tau': calibration.get(n, {}).get('matched', {})} for n in names],
            'default': DEFAULT_MODEL if DEFAULT_MODEL in names else next(iter(names), None),
            'corruptions': list(CORRUPTIONS)}


@app.post('/api/detect')
def image(file: UploadFile = File(...), model: str | None = Form(None), risk: float = Form(.10),
          corruption: str = Form('none'), severity: int = Form(1)):
    if severity not in (1, 2, 3) or corruption not in ('none', *CORRUPTIONS):
        raise HTTPException(422, 'Choose a listed corruption and severity 1–3.')
    img = decode(file.file.read())
    if corruption != 'none':
        img = apply(img, corruption, severity, np.random.default_rng(42))
    result = detect(img, model, risk)
    # PNG is lossless so the canvas shows the exact pixels used for inference.
    result['image_base64'] = base64.b64encode(encode(img, '.png')).decode('ascii')
    result['image_mime'] = 'image/png'
    return result


def ffmpeg_path():
    binary = shutil.which('ffmpeg')
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def video_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, size)
    if writer.isOpened():
        return writer, False
    writer.release()
    if not ffmpeg_path():
        raise HTTPException(503, 'H.264 encoding is unavailable. Run pip install imageio-ffmpeg, then restart the server.')
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        writer.release()
        raise HTTPException(503, 'Could not initialize the MP4 encoder.')
    return writer, True


@app.post('/api/video')
def video(file: UploadFile = File(...), model: str | None = Form(None), risk: float = Form(.10),
          every_n: int = Form(2)):
    settings(model, risk)
    if not 1 <= every_n <= 120:
        raise HTTPException(422, 'Frame interval must be between 1 and 120.')
    token = uuid.uuid4().hex
    source = OUTPUTS / f'{token}.upload'
    raw = OUTPUTS / f'{token}.working.mp4'
    target = OUTPUTS / f'{token}.mp4'
    cap = writer = None
    completed = False
    try:
        with source.open('wb') as destination:
            shutil.copyfileobj(file.file, destination)
        cap = cv2.VideoCapture(str(source))
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(422, 'Unreadable video. Try a valid MP4 file.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        h, w = frame.shape[:2]
        size = (w + w % 2, h + h % 2)
        writer, transcode = video_writer(raw, fps / every_n, size)
        summaries, index = [], 0
        while ok:
            if index % every_n == 0:
                result = detect(frame, model, risk)
                writer.write(cv2.resize(annotate(frame, result), size))
                summaries.append(dict(frame=index, time_s=round(index / fps, 3),
                                      counts=result['counts'], inference_ms=result['inference_ms']))
            index += 1
            ok, frame = cap.read()
        writer.release()
        writer = None
        if transcode:
            process = subprocess.run([ffmpeg_path(), '-y', '-i', str(raw), '-an',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)],
                capture_output=True, timeout=600)
            if process.returncode:
                raise HTTPException(503, 'Video conversion failed. Check that FFmpeg includes libx264.')
        else:
            raw.replace(target)
        completed = True
        return {'url': f'/outputs/{target.name}', 'frames': summaries, 'every_n': every_n,
                'source_fps': fps, 'processed_frames': len(summaries)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, 'Video processing failed. Check the file and available disk space.') from exc
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        source.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        if not completed:
            target.unlink(missing_ok=True)


@app.websocket('/ws/stream')
async def stream(ws: WebSocket):
    await ws.accept()
    queue = asyncio.Queue(maxsize=1)

    async def receive():
        while True:
            data = await ws.receive_bytes()
            if queue.full():
                queue.get_nowait()  # Replace stale pending frame with the latest frame.
            queue.put_nowait(data)

    receiver = asyncio.create_task(receive())
    pending = None
    try:
        model = ws.query_params.get('model') or None
        risk = float(ws.query_params.get('risk', '.10'))
        settings(model, risk)
        while True:
            pending = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait([receiver, pending], return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                receiver.result()
                break
            try:
                result = await asyncio.to_thread(lambda: detect(decode(pending.result()), model, risk))
                await ws.send_json(result)
            except HTTPException as exc:
                await ws.send_json({'error': exc.detail})
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except (HTTPException, ValueError) as exc:
        await ws.send_json({'error': getattr(exc, 'detail', 'Invalid stream settings.')})
        await ws.close(code=1008)
    finally:
        receiver.cancel()
        tasks = [receiver]
        if pending is not None:
            pending.cancel()
            tasks.append(pending)
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)


@app.post('/api/camera/start')
def camera_start(request: CameraRequest):
    model, _ = settings(request.model, request.risk)
    source = request.source
    if isinstance(source, str):
        source = source.strip()
        if source.isdecimal():
            source = int(source)
        elif urlparse(source).scheme not in ('rtsp', 'http', 'https') or not urlparse(source).hostname:
            raise HTTPException(422, 'Enter a webcam index or a valid RTSP/HTTP camera URL.')
    if isinstance(source, int) and source < 0:
        raise HTTPException(422, 'Webcam index must be zero or greater.')
    with camera.control:
        camera.stop()
        camera.error = camera.result = None
        camera.stop_event = threading.Event()
        camera.thread = threading.Thread(target=camera.run, args=(source, model, request.risk), daemon=True)
        camera.thread.start()
    return {'status': 'starting'}


@app.post('/api/camera/stop')
def camera_stop():
    with camera.control:
        camera.stop()
    return {'status': 'stopped'}


@app.get('/api/camera/status')
def camera_status():
    return {'running': bool(camera.thread and camera.thread.is_alive()), 'error': camera.error,
            'result': camera.result, 'sequence': camera.sequence}


@app.get('/api/camera/feed')
async def camera_feed():
    async def frames():
        last = -1
        while camera.thread and camera.thread.is_alive():
            if camera.jpeg and camera.sequence != last:
                last = camera.sequence
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + camera.jpeg + b'\r\n'
            await asyncio.sleep(.03)
    return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/')
def index():
    return FileResponse(APP_DIR / 'static/index.html')


app.mount('/static', StaticFiles(directory=APP_DIR / 'static'), name='static')
app.mount('/outputs', StaticFiles(directory=OUTPUTS), name='outputs')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.server:app', host='127.0.0.1', port=8000)
