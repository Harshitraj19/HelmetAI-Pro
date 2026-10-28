"""
HelmetAI-Pro — FastAPI backend (built on the DIP_PROJECT base, v3)

Kept from the base (same paths, compatible responses)
  GET  /                              health
  POST /api/detect/image              annotated JPEG  (+ X-Detections / X-Violation-Count headers)
  GET  /api/stream/video?source=      MJPEG live stream (webcam index, RTSP/HTTP URL, YouTube, sample file)
  POST /api/detect/video              upload a video -> MJPEG of annotated frames (+ saved MP4)
  GET  /api/detections  DELETE /api/detections
  GET  /api/stats
  POST /api/train/start  GET /api/train/status  POST /api/evaluate
  WS   /ws/notifications

New in Pro
  POST /api/detect/image.json         same as image detect but JSON (+ base64 image) - easier for clients
  GET  /api/violations                filter by type / plate / status / severity
  GET  /api/violations/{id}           full record
  POST /api/violations/{id}/review    human-in-the-loop confirm / reject (audited, does not break chain)
  GET  /api/violations/{id}/challan.pdf   auto e-challan with QR + hash
  GET  /api/evidence/verify           verify the SHA-256 hash chain (tamper detection)
  GET  /api/evidence/file/{path}      evidence images
  GET  /api/outputs/{name}            processed videos (MP4)
  GET  /api/settings  PUT /api/settings   enhancement / privacy / OCR / mode toggles at runtime
  GET  /api/model                     loaded models, classes, metrics
  POST /api/enhance/preview           before/after of the adaptive enhancement for an image
  GET  /api/samples                   sample media shipped with the project
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import anyio
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from sqlalchemy import func

import config
import store
from models import DetectionRecord, SessionLocal, Violation, init_db, utcnow
from pipeline.challan import build_challan
from pipeline.detector import Detector
from pipeline.engine import Engine
from pipeline.enhance import enhance
from pipeline.plate_ocr import OCR_AVAILABLE
from pipeline.videoio import LatestFrameReader, open_writer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

SETTINGS = dict(config.RUNTIME_DEFAULTS, mode="traffic", show_enhanced=True)
ENHANCE_MODES = ("auto", "off", "night", "fog", "low_contrast", "blurry")
engine: Engine | None = None
ws_clients: set[WebSocket] = set()
main_loop: asyncio.AbstractEventLoop | None = None
_engine_lock = threading.Lock()          # one inference at a time on small CPUs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine, main_loop
    main_loop = asyncio.get_running_loop()
    init_db()
    try:
        engine = build_engine()
        log.info("HelmetAI-Pro ready with models: %s", engine.det.model_names)
    except Exception as e:  # API still serves history / evidence even without a model
        log.error("Detection disabled: %s", e)
    yield


app = FastAPI(title="HelmetAI-Pro API", version="3.1.0", lifespan=lifespan,
              description="Helmet / rider / plate enforcement system — merged & extended from DIP_PROJECT")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_origin_regex=config.CORS_ORIGIN_REGEX or None,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Detections", "X-Violation-Count", "X-Result"])


# --------------------------------------------------------------------------- startup
def parse_stack(spec: str):
    out = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        m = re.match(r"^(.*?)(?::([a-z_|]+))?$", item)
        path, classes = m.group(1), m.group(2)
        out.append((path, tuple(classes.split("|")) if classes else None, None))
    return out


def build_engine() -> Engine:
    det = Detector(parse_stack(config.MODEL_STACK), imgsz=config.IMG_SIZE,
                   class_conf=config.CLASS_CONF, base_conf=config.BASE_CONF)
    return Engine(det, SETTINGS, store=store, notify=notify_threadsafe)


def need_engine() -> Engine:
    if engine is None:
        raise HTTPException(503, "No detection model loaded. Train one (training/run_all.sh) or "
                                 "set MODEL_STACK; see README.")
    return engine


def need_admin(token: str | None):
    if config.ADMIN_TOKEN and token != config.ADMIN_TOKEN:
        raise HTTPException(401, "X-Admin-Token required")


# --------------------------------------------------------------------------- websocket
@app.websocket("/ws/notifications")
async def ws_notifications(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:                       # any exit path (not only a clean disconnect) drops the client
        ws_clients.discard(ws)


async def _send(ws: WebSocket, payload: str):
    try:
        await asyncio.wait_for(ws.send_text(payload), timeout=2)
    except Exception:              # dead or stalled client must not hold up everyone else
        ws_clients.discard(ws)


async def broadcast(message: dict):
    payload = json.dumps(message, default=str)
    await asyncio.gather(*(_send(ws, payload) for ws in list(ws_clients)))


def notify_threadsafe(message: dict):
    if main_loop and ws_clients and not main_loop.is_closed():
        asyncio.run_coroutine_threadsafe(broadcast(message), main_loop)


# --------------------------------------------------------------------------- basics
@app.get("/")
def health():
    return {"status": "ok", "service": "HelmetAI-Pro API", "version": app.version,
            "detection": engine is not None, "models": engine.det.model_names if engine else [],
            "plate_ocr_available": OCR_AVAILABLE}


@app.get("/api/model")
def model_info():
    metrics = config.MODELS_DIR / "metrics.json"
    try:
        m = json.loads(metrics.read_text()) if metrics.exists() else None
    except (OSError, ValueError):
        m = None
    return {"stack": config.MODEL_STACK, "loaded": engine.det.model_names if engine else [],
            "img_size": config.IMG_SIZE, "class_conf": config.CLASS_CONF, "metrics": m,
            "plate_ocr_available": OCR_AVAILABLE}


@app.get("/api/settings")
def get_settings():
    return SETTINGS


SETTING_CHOICES = {"enhancement": ENHANCE_MODES, "mode": ("traffic", "site")}
SETTING_RANGES = {"smoothing_window": (1, 31), "min_track_hits": (1, 30)}


def _coerce_setting(k, v):
    cur = SETTINGS[k]
    if isinstance(cur, bool):
        if isinstance(v, str):     # "false" must not become True
            if v.lower() not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                raise ValueError("expected a boolean")
            return v.lower() in ("1", "true", "yes", "on")
        return bool(v)
    if isinstance(cur, int):
        v = int(v)
        lo, hi = SETTING_RANGES.get(k, (-10**9, 10**9))
        if not lo <= v <= hi:
            raise ValueError(f"must be between {lo} and {hi}")
        return v
    v = str(v).strip()
    if k in SETTING_CHOICES and v not in SETTING_CHOICES[k]:
        raise ValueError(f"must be one of {SETTING_CHOICES[k]}")
    if not v or len(v) > 64:
        raise ValueError("must be 1-64 characters")
    return v


@app.put("/api/settings")
def put_settings(patch: dict):
    new = {}
    for k, v in patch.items():         # validate everything first: a bad field changes nothing
        if k not in SETTINGS:
            raise HTTPException(400, f"unknown setting {k}")
        try:
            new[k] = _coerce_setting(k, v)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"{k}: {e}")
    SETTINGS.update(new)
    return SETTINGS


# --------------------------------------------------------------------------- image
def _decode(contents: bytes):
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img


MAX_IMAGE_SIDE = 2560      # 12-MP phone photos: same accuracy at a fraction of the enhancement / blur cost


def _run_image(img, name, mode=None):
    eng = need_engine()
    if max(img.shape[:2]) > MAX_IMAGE_SIDE:
        r = MAX_IMAGE_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    s = eng.session(name, image_mode=True, mode=mode)
    with _engine_lock:
        out, res = eng.process(img, s)
        res["new_violations"] += eng.finish(s)
    # base-compatible flat detection rows (class names as the base UI expects)
    legacy = []
    for v in res["vehicles"]:
        legacy += [{"class": "With Helmet", "confidence": 1.0, "bbox": v["box"], "track_id": v["track_id"]}] * v["helmets"]
        legacy += [{"class": "Without Helmet", "confidence": 1.0, "bbox": v["box"], "track_id": v["track_id"]}] * v["no_helmet"]
    res["legacy"] = legacy
    # Riders on motorcycles are already logged once per rider by eng.finish() (Engine._log_track);
    # logging `legacy` here as well counted every rider twice in the stats.
    if not res["vehicles"]:     # no motorcycle: still log raw head-level detections like the base did
        store.log_detections([dict(object_class="With Helmet" if d["class"] in ("helmet", "hardhat") else "Without Helmet",
                                   confidence=d["confidence"], image_path=name, source=name,
                                   camera_id=SETTINGS["camera_id"], bbox=",".join(map(str, d["bbox"])))
                              for d in res["detections"] if d["class"] in ("helmet", "head", "hardhat")])
    return out, res


def _slim(res):
    return {k: res[k] for k in ("mode", "detections", "vehicles", "scene", "counters", "latency_ms")} | {
        "violations": [{k: v.get(k) for k in ("id", "track_id", "violation_list", "plate", "severity", "severity_level")}
                       for v in res["new_violations"]]}


IMAGE_MODE = Query(None, pattern="^(auto|traffic|site)$",
                   description="auto = traffic rules if a motorcycle is in the picture, else check every person")


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), mode: str | None = IMAGE_MODE):
    img = _decode(await file.read())
    out, res = await asyncio.to_thread(_run_image, img, file.filename or "upload.jpg", mode)
    _, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    n_viol = sum(1 for v in res["vehicles"] if v["violations"]) or sum(1 for d in res["detections"] if d["class"] == "head")
    detections = [d | {"class": {"helmet": "With Helmet", "head": "Without Helmet"}.get(d["class"], d["class"])}
                  for d in res["detections"]]
    return Response(jpeg.tobytes(), media_type="image/jpeg", headers={
        "X-Detections": json.dumps(detections), "X-Violation-Count": str(n_viol),
        "X-Result": json.dumps(_slim(res), default=str)})


@app.post("/api/detect/image.json")
async def detect_image_json(file: UploadFile = File(...), mode: str | None = IMAGE_MODE):
    img = _decode(await file.read())
    out, res = await asyncio.to_thread(_run_image, img, file.filename or "upload.jpg", mode)
    _, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return _slim(res) | {"image_base64": base64.b64encode(jpeg.tobytes()).decode()}


def _preview(img, mode):
    out, stats = enhance(img, mode)
    h = 360
    a = cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(out, (a.shape[1], h), interpolation=cv2.INTER_AREA)
    combo = np.hstack([a, np.full((h, 6, 3), 255, np.uint8), b])
    return cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], stats


@app.post("/api/enhance/preview")
async def enhance_preview(file: UploadFile = File(...), mode: str = Query("auto")):
    if mode not in ENHANCE_MODES:
        raise HTTPException(400, f"mode must be one of {ENHANCE_MODES}")
    img = _decode(await file.read())
    jpeg, stats = await asyncio.to_thread(_preview, img, mode)    # dehazing a big photo must not block the event loop
    return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"X-Result": json.dumps(stats.as_dict())})


# --------------------------------------------------------------------------- video / stream
def resolve_source(source: str):
    """Security fix vs. the base: only webcams, network streams, YouTube, and files under
    samples/ or uploads/ are allowed (the base accepted any server path)."""
    if source.isdigit():
        return int(source)
    if re.match(r"^(rtsp|rtmp|http|https)://", source):
        if "youtube.com" in source or "youtu.be" in source:
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as y:
                    return y.extract_info(source, download=False)["url"]
            except Exception as e:
                raise HTTPException(400, f"YouTube source failed: {e}")
        return source
    if config.ALLOW_LOCAL_FILE_SOURCES:
        for base in (config.SAMPLES_DIR, config.UPLOAD_DIR):
            for cand in (base / source, base / "media" / source):
                p = cand.resolve()
                if p.is_file() and base.resolve() in p.parents:
                    return str(p)
    raise HTTPException(400, "Source must be a webcam index, rtsp/http(s) URL, YouTube link, or a sample file name")


def _fit_width(frame):
    w = frame.shape[1]
    if config.MAX_FRAME_WIDTH and w > config.MAX_FRAME_WIDTH:
        s = config.MAX_FRAME_WIDTH / w
        return cv2.resize(frame, (config.MAX_FRAME_WIDTH, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return frame


def _cleanup_upload(src):
    if isinstance(src, str) and Path(src).resolve().parent == config.UPLOAD_DIR.resolve():
        try:
            os.remove(src)
        except OSError:
            pass


def frame_generator(src, name: str, save_mp4: bool, kind: str, run_id: str | None = None, mode: str | None = None,
                    stride: int = 1):
    """run_id (chosen by the client) is echoed in every WebSocket message of this session, so a page
    only reacts to ITS video/stream when several run at the same time.
    mode overrides the global setting for this run only: 'traffic' = riders on motorcycles,
    'site' = every visible person must wear a helmet (what a webcam demo needs).
    stride (files only): analyse every n-th frame - n x faster, tracking still holds at 2-3.
    Live sources (webcam, RTSP) always get the NEWEST frame (LatestFrameReader), so the view stays
    real-time even though the CPU processes fewer frames per second than the camera delivers."""
    eng = need_engine()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        _cleanup_upload(src)
        raise HTTPException(400, f"Cannot open video source {name}")
    s = eng.session(name, mode=mode)
    fps = cap.get(cv2.CAP_PROP_FPS)
    live = isinstance(src, int) or bool(re.match(r"^(rtsp|rtmp)://", str(src)))
    stride = 1 if live else max(1, stride)
    out_name = f"{re.sub(r'[^A-Za-z0-9_-]', '_', Path(str(name)).stem)[:40]}_{uuid.uuid4().hex[:6]}.mp4"
    writer = reader = None
    last_push = 0.0
    started = False
    key = run_id or uuid.uuid4().hex
    stop = ACTIVE_RUNS[key] = threading.Event()

    def abort():
        """Cleanup for a run whose generator never started (client left before the first frame):
        a never-started generator does not execute its finally block."""
        ACTIVE_RUNS.pop(key, None)
        if not started:
            cap.release()
            if kind == "video_upload_score":
                _cleanup_upload(src)

    def gen():
        nonlocal writer, last_push, reader, started
        started = True
        error = None
        reader = LatestFrameReader(cap) if live else None
        try:
            while not stop.is_set():
                if reader:
                    ok, frame = reader.read()
                else:
                    for _ in range(stride - 1):      # skipped frames: grab() avoids the colour conversion
                        cap.grab()
                    ok, frame = cap.read()
                if not ok:
                    break
                frame = _fit_width(frame)
                with _engine_lock:
                    out, res = eng.process(frame, s)
                if save_mp4:
                    if writer is None:
                        writer = open_writer(config.OUTPUT_DIR / out_name, (fps or 25) / stride, (out.shape[1], out.shape[0]))
                    # live: repeat the frame for the camera frames dropped meanwhile -> real-time playback
                    for _ in range(min(reader.span, 30) if reader else 1):
                        writer.write(out)
                now = time.time()
                if now - last_push > 0.5:
                    last_push = now
                    c = res["counters"]
                    cur_h = sum(v["helmets"] for v in res["vehicles"])
                    cur_n = sum(v["no_helmet"] for v in res["vehicles"])
                    notify_threadsafe({"type": kind, "run_id": run_id, "mode": s.mode, "helmets": cur_h, "no_helmets": cur_n,
                                       "frame": res["frame"], "counters": c, "scene": res["scene"]["scene"]})
                _, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        except GeneratorExit:      # client went away - normal for streams
            raise
        except Exception as e:     # keep the server alive and tell the UI what happened
            log.exception("processing %s failed", name)
            error = str(e)
        finally:
            ACTIVE_RUNS.pop(key, None)
            if reader:
                reader.release()           # its thread releases the camera
            else:
                cap.release()
            try:
                with _engine_lock:
                    eng.finish(s)
            finally:
                if writer is not None:
                    writer.release()
                c = s.counters()
                # field names the base UI expects (fixes the base's summary mismatch bug) + new unique counts
                notify_threadsafe({"type": f"{kind}_done" if kind != "video_score" else "stream_done", "run_id": run_id,
                                   "total_frames": c["frames"], "total_helmets": c["riders_with_helmet"],
                                   "total_no_helmets": c["unique_riders"] - c["riders_with_helmet"],
                                   "max_helmets": c["riders_with_helmet"], "max_no_helmets": c["unique_riders"] - c["riders_with_helmet"],
                                   "counters": c, "error": error,
                                   "output_video": f"/api/outputs/{out_name}" if save_mp4 and writer is not None else None})
                if kind == "video_upload_score":
                    _cleanup_upload(src)

    return gen(), abort


ACTIVE_RUNS: dict[str, threading.Event] = {}     # run_id -> stop flag of every running video / stream


async def mjpeg(request: Request, run) -> StreamingResponse:
    """Serve a frame generator as MJPEG with GUARANTEED cleanup.
    Starlette only stops *iterating* a sync generator when the client disconnects - it never closes
    it, so the generator's finally block (camera release!) never ran and the webcam stayed on after
    Stop. Here a disconnect, a cancellation or the natural end always closes it."""
    frames, abort = run

    async def body():
        try:
            while not await request.is_disconnected():
                chunk = await anyio.to_thread.run_sync(next, frames, None)
                if chunk is None:
                    break
                yield chunk
        finally:
            with anyio.CancelScope(shield=True):         # must finish even while being cancelled
                await anyio.to_thread.run_sync(frames.close)
                abort()

    return StreamingResponse(body(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/stream/video")
async def stream_video(request: Request, source: str = Query("0"), record: bool = Query(False),
                       run_id: str | None = Query(None, max_length=64),
                       mode: str | None = Query(None, pattern="^(traffic|site)$")):
    src = await anyio.to_thread.run_sync(resolve_source, source)
    run = await anyio.to_thread.run_sync(lambda: frame_generator(src, str(source), record, "video_score", run_id, mode))
    return await mjpeg(request, run)


@app.post("/api/stream/stop")
def stop_stream(run_id: str | None = Query(None, max_length=64)):
    """Stop one run (or EVERY running video/stream when run_id is omitted) and release its camera.
    Works even when the browser keeps the MJPEG connection open."""
    keys = [run_id] if run_id else list(ACTIVE_RUNS)
    stopped = []
    for k in keys:
        ev = ACTIVE_RUNS.get(k)
        if ev is not None:
            ev.set()
            stopped.append(k)
    return {"stopped": stopped, "still_running": len(ACTIVE_RUNS) - len(stopped)}


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


@app.post("/api/detect/video")
async def detect_video(request: Request, file: UploadFile = File(...), run_id: str | None = Query(None, max_length=64),
                       mode: str | None = Query(None, pattern="^(traffic|site)$"),
                       stride: int = Query(1, ge=1, le=5, description="analyse every n-th frame (speed-up)")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        return JSONResponse(status_code=400, content={"error": f"Unsupported video format '{ext}'. Supported: {sorted(VIDEO_EXT)}"})
    need_engine()                       # fail fast, before receiving hundreds of MB
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    size, too_big = 0, False
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB << 20:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
    run = await asyncio.to_thread(frame_generator, str(path), file.filename or path.name, True, "video_upload_score",
                                  run_id, mode, stride)
    return await mjpeg(request, run)


@app.get("/api/outputs/{name}")
def get_output(name: str):
    p = (config.OUTPUT_DIR / name).resolve()
    if config.OUTPUT_DIR.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4", filename=name)


@app.get("/api/samples")
def samples():
    media = config.SAMPLES_DIR / "media"
    return sorted(p.name for p in media.iterdir() if p.is_file()) if media.exists() else []


# --------------------------------------------------------------------------- history & stats
@app.get("/api/detections")
def get_detections(limit: int = Query(50, ge=1, le=500)):
    db = SessionLocal()
    try:
        rows = db.query(DetectionRecord).order_by(DetectionRecord.timestamp.desc()).limit(limit).all()
        return [{"id": r.id, "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                 "object_class": r.object_class, "confidence": r.confidence, "image_path": r.image_path,
                 "camera_id": r.camera_id, "track_id": r.track_id} for r in rows]
    finally:
        db.close()


@app.delete("/api/detections")
def clear_detections(x_admin_token: str | None = Header(None)):
    need_admin(x_admin_token)
    db = SessionLocal()
    try:
        db.query(DetectionRecord).delete()
        db.commit()
        return {"status": "success", "message": "Detection history cleared (violation evidence chain is append-only and kept)."}
    finally:
        db.close()


@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    try:
        total = db.query(func.count(DetectionRecord.id)).scalar() or 0
        helmet = db.query(func.count(DetectionRecord.id)).filter(DetectionRecord.object_class == "With Helmet").scalar() or 0
        no_helmet = db.query(func.count(DetectionRecord.id)).filter(DetectionRecord.object_class == "Without Helmet").scalar() or 0
        since = utcnow() - datetime.timedelta(hours=24)
        hourly = dict(db.query(func.strftime("%H", DetectionRecord.timestamp), func.count(DetectionRecord.id))
                      .filter(DetectionRecord.timestamp >= since).group_by(func.strftime("%H", DetectionRecord.timestamp)).all())
        v_hourly = dict(db.query(func.strftime("%H", Violation.created_at), func.count(Violation.id))
                        .filter(Violation.created_at >= since).group_by(func.strftime("%H", Violation.created_at)).all())
        types = {}
        for (vt,) in db.query(Violation.violation_types).all():
            for t in (vt or "").split(","):
                if t:
                    types[t] = types.get(t, 0) + 1
        sev = dict(db.query(Violation.severity_level, func.count(Violation.id)).group_by(Violation.severity_level).all())
        n_viol = db.query(func.count(Violation.id)).scalar() or 0
        plates = db.query(func.count(Violation.id)).filter(Violation.plate.isnot(None)).scalar() or 0
        repeat = db.query(Violation.plate, func.count(Violation.id)).filter(Violation.plate.isnot(None)) \
            .group_by(Violation.plate).having(func.count(Violation.id) > 1).order_by(func.count(Violation.id).desc()).limit(5).all()
        cams = dict(db.query(Violation.camera_id, func.count(Violation.id)).group_by(Violation.camera_id).all())
        return {
            "total_detections": total, "with_helmet": helmet, "without_helmet": no_helmet,
            "compliance_rate": round(helmet / (helmet + no_helmet) * 100, 1) if helmet + no_helmet else 0,
            "hourly_breakdown": hourly, "violation_hourly": v_hourly,
            "violations_total": n_viol, "violation_types": types, "severity": sev,
            "plates_read": plates, "plate_read_rate": round(plates / n_viol * 100, 1) if n_viol else 0,
            "repeat_offenders": [{"plate": p, "count": c} for p, c in repeat], "by_camera": cams,
        }
    finally:
        db.close()


# --------------------------------------------------------------------------- violations & evidence
@app.get("/api/violations")
def list_violations(limit: int = Query(100, ge=1, le=1000), type: str | None = None, plate: str | None = None,
                    status: str | None = None, min_severity: int = 0):
    db = SessionLocal()
    try:
        q = db.query(Violation).filter(Violation.severity >= min_severity)
        if type:
            q = q.filter(Violation.violation_types.contains(type))
        if plate:
            q = q.filter(Violation.plate.ilike(f"%{plate}%"))
        if status:
            q = q.filter(Violation.status == status)
        return [store.row_to_dict(v) for v in q.order_by(Violation.id.desc()).limit(limit).all()]
    finally:
        db.close()


@app.get("/api/violations/{vid}")
def get_violation(vid: int):
    db = SessionLocal()
    try:
        v = db.get(Violation, vid)
        if not v:
            raise HTTPException(404)
        return store.row_to_dict(v)
    finally:
        db.close()


@app.post("/api/violations/{vid}/review")
def review_violation(vid: int, body: dict):
    action = body.get("action")
    if action not in ("confirmed", "rejected", "pending"):
        raise HTTPException(400, "action must be confirmed | rejected | pending")
    r = store.review(vid, action, body.get("note"))
    if not r:
        raise HTTPException(404)
    return r


@app.get("/api/violations/{vid}/challan.pdf")
def challan(vid: int):
    v = get_violation(vid)
    pdf = build_challan(v, config.EVIDENCE_DIR, f"{config.PUBLIC_URL}/api/evidence/verify")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="challan_HAI-{vid:06d}.pdf"'})


@app.get("/api/evidence/verify")
def verify_chain(full: bool = Query(True, description="false = reuse hashes of unchanged image files (fast polling)")):
    return store.verify(full=full)


@app.get("/api/evidence/file/{path:path}")
def evidence_file(path: str):
    p = (config.EVIDENCE_DIR / path).resolve()
    if config.EVIDENCE_DIR.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


# --------------------------------------------------------------------------- training (kept from base, fixed)
training_process: subprocess.Popen | None = None
TRAIN_LOG = config.DATA_DIR / "training.log"


@app.post("/api/train/start")
async def start_training(cfg: dict, x_admin_token: str | None = Header(None)):
    """Runs the full recipe (merge -> pseudo-label -> train -> evaluate -> export).
    Body: {"profile": "cpu"|"gpu", "epochs": int?}.  The base's Roboflow key is no longer passed on
    the command line (it leaked into `ps`); Roboflow datasets can still be merged via env vars."""
    global training_process
    need_admin(x_admin_token)
    if training_process and training_process.poll() is None:
        return {"status": "error", "message": "Training already in progress."}
    profile = cfg.get("profile", "cpu")
    if profile not in ("cpu", "gpu", "max"):
        raise HTTPException(400, "profile must be cpu | gpu | max")
    bash = shutil.which("bash")
    if not bash:
        raise HTTPException(400, "bash not found - on Windows install Git for Windows, or use the GPU notebook in training/")
    cmd = [bash, str(config.ROOT / "training/run_all.sh"), profile]
    with open(TRAIN_LOG, "w") as log_f:      # the child keeps its own handle; ours must not leak
        training_process = subprocess.Popen(cmd, cwd=config.ROOT, stdout=log_f, stderr=subprocess.STDOUT)
    return {"status": "success", "message": f"Training ({profile}) started in background."}


@app.get("/api/train/status")
def get_train_status():
    tail = TRAIN_LOG.read_text(errors="replace")[-3000:] if TRAIN_LOG.exists() else ""
    if not training_process:
        return {"status": "idle", "log": tail}
    rc = training_process.poll()
    st = "running" if rc is None else "completed" if rc == 0 else "failed"
    return {"status": st, "exit_code": rc, "log": tail}


@app.post("/api/model/reload")
def reload_model(x_admin_token: str | None = Header(None)):
    """Hot-swap to a freshly trained model (the base never reloaded its weights)."""
    global engine
    need_admin(x_admin_token)
    old_stack = config.MODEL_STACK
    config.MODEL_STACK = os.environ.get("MODEL_STACK") or config.default_stack()
    try:
        new = build_engine()
    except Exception as e:             # keep serving with the previous model
        config.MODEL_STACK = old_stack
        raise HTTPException(500, f"reload failed, previous model kept: {e}")
    with _engine_lock:                 # swap between frames, never in the middle of one
        engine = new
    return {"loaded": engine.det.model_names}


@app.post("/api/evaluate")
async def run_evaluation(cfg: dict, x_admin_token: str | None = Header(None)):
    need_admin(x_admin_token)          # minutes of CPU - same protection as training
    weights = cfg.get("model_path") or str(config.MODELS_DIR / "helmet_unified.pt")
    if not Path(weights).is_file():
        return {"status": "error", "message": f"weights not found: {weights}"}
    r = await asyncio.to_thread(subprocess.run, [sys.executable, str(config.ROOT / "training/evaluate.py"),
                                "--weights", weights], capture_output=True, text=True, cwd=config.ROOT)
    if r.returncode != 0:
        return {"status": "error", "message": (r.stderr or r.stdout)[-2000:]}
    try:
        m = json.loads((config.MODELS_DIR / "metrics.json").read_text())["per_class_eval"]
    except (OSError, ValueError, KeyError) as e:
        return {"status": "error", "message": f"evaluation finished but metrics.json is unreadable: {e}"}
    pc = list(m["per_class"].values()) or [{"precision": 0, "recall": 0}]
    report = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
              "precision": round(sum(x["precision"] for x in pc) / len(pc), 4),
              "recall": round(sum(x["recall"] for x in pc) / len(pc), 4), "per_class": m["per_class"]}
    return {"status": "success", "report": report}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
