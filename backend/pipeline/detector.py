"""
Multi-backend, multi-model detector.

Backends (auto-chosen by file extension):
  .pt    -> Ultralytics (PyTorch)            training/dev machines
  .onnx  -> onnxruntime (no PyTorch needed)  edge / lightweight deploys
  cloud  -> Roboflow hosted workflow         kept from the DIP_PROJECT base (optional)

Several models can be ENSEMBLED (e.g. a COCO model for person/motorcycle + a
helmet model). All class names are normalised to one vocabulary:

  helmet | head | motorcycle | license_plate | hardhat | person
"""
from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("detector")

CANON = ("helmet", "head", "motorcycle", "license_plate", "hardhat", "person")


def canonical(name: str) -> str | None:
    """Map any dataset's label to the unified vocabulary (None = ignore)."""
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    if n in ("without helmet", "no helmet", "nohelmet", "head", "human head", "bare head", "without hardhat", "no hardhat"):
        return "head"
    if n in ("hardhat", "hard hat", "safety helmet", "construction helmet"):
        return "hardhat"
    if "helmet" in n:                       # "with helmet", "helmet", "rider helmet" ...
        return "helmet"
    if n in ("motorcycle", "motorbike", "bike", "scooter", "two wheeler"):
        return "motorcycle"
    if "plate" in n or "license" in n or "licence" in n or "registration" in n:
        return "license_plate"
    if n in ("person", "rider", "pedestrian"):
        return "person"
    return None


@dataclass
class Det:
    cls: str
    conf: float
    box: tuple[int, int, int, int]       # x1 y1 x2 y2 (pixels)
    source: str = ""

    @property
    def cx(self): return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self): return (self.box[1] + self.box[3]) / 2

    @property
    def w(self): return self.box[2] - self.box[0]

    @property
    def h(self): return self.box[3] - self.box[1]

    def as_dict(self):
        return {"class": self.cls, "confidence": round(self.conf, 3), "bbox": list(self.box)}


def iou(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(dets: list[Det], thr: float = 0.5) -> list[Det]:
    """Class-aware NMS used to merge ensemble outputs."""
    out: list[Det] = []
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        if all(not (o.cls == d.cls and iou(o.box, d.box) > thr) for o in out):
            out.append(d)
    return out


# --------------------------------------------------------------------------- backends
class UltralyticsBackend:
    def __init__(self, path: str, imgsz: int):
        from ultralytics import YOLO  # imported lazily so ONNX-only installs work
        self.model = YOLO(path)
        self.names = self.model.names
        self.imgsz = imgsz

    def __call__(self, bgr, conf):
        r = self.model(bgr, conf=conf, imgsz=self.imgsz, verbose=False)[0]
        for b in r.boxes:
            yield self.names[int(b.cls)], float(b.conf), tuple(map(int, b.xyxy[0].tolist()))


class OnnxBackend:
    """Runs Ultralytics-exported ONNX (YOLOv8/11: [1,4+nc,N]) or YOLOv5 ([1,N,5+nc])."""

    def __init__(self, path: str, imgsz: int, names: list[str] | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's default uses the PHYSICAL cores. Forcing os.cpu_count() (logical cores, x2 with
        # hyper-threading, x2 again for an ensemble) oversubscribes the CPU and was ~4x slower.
        threads = int(os.environ.get("ORT_THREADS", "0"))
        if threads > 0:
            so.intra_op_num_threads = threads
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail] + ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, so, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        shape = inp.shape
        self.imgsz = shape[2] if isinstance(shape[2], int) else imgsz
        meta = self.sess.get_modelmeta().custom_metadata_map
        if names is None and "names" in meta:          # Ultralytics stores names in metadata
            import ast
            names = list(ast.literal_eval(meta["names"]).values())
        self.names = dict(enumerate(names or []))

    def __call__(self, bgr, conf):
        h0, w0 = bgr.shape[:2]
        s = self.imgsz
        r = min(s / h0, s / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input: blob})[0][0]
        if out.shape[0] < out.shape[1]:          # v8/v11: (4+nc, N) -> (N, 4+nc)
            out = out.T
            boxes, scores = out[:, :4], out[:, 4:]
        else:                                    # v5: (N, 5+nc)
            boxes, scores = out[:, :4], out[:, 5:] * out[:, 4:5]
        cls = scores.argmax(1)
        cf = scores[np.arange(len(scores)), cls]
        keep = cf >= conf
        boxes, cls, cf = boxes[keep], cls[keep], cf[keep]
        if not len(cf):
            return
        xy = np.empty_like(boxes)
        xy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - px) / r
        xy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - py) / r
        xy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - px) / r
        xy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - py) / r
        # per-class NMS via coordinate offset trick (offset > frame size, so classes never overlap)
        off = (cls * (max(w0, h0) + 1))[:, None]
        rects = [(float(a), float(b), float(c - a), float(d - b)) for a, b, c, d in (xy + off)]
        idx = cv2.dnn.NMSBoxes(rects, cf.tolist(), conf, 0.5)
        for i in np.array(idx).flatten():
            x1, y1, x2, y2 = xy[i]
            yield (self.names.get(int(cls[i]), str(cls[i])), float(cf[i]),
                   (int(max(0, x1)), int(max(0, y1)), int(min(w0 - 1, x2)), int(min(h0 - 1, y2))))


class RoboflowBackend:
    """Hosted inference (from the DIP_PROJECT base). Credentials come ONLY from env vars."""

    def __init__(self):
        import requests
        self.requests = requests
        self.url = (f"https://detect.roboflow.com/{os.environ['ROBOFLOW_WORKSPACE']}/workflows/"
                    f"{os.environ['ROBOFLOW_WORKFLOW_ID']}")
        self.key = os.environ["ROBOFLOW_API_KEY"]

    def __call__(self, bgr, conf):
        _, buf = cv2.imencode(".jpg", bgr)
        payload = {"api_key": self.key, "inputs": {"image": {"type": "base64", "value": base64.b64encode(buf).decode()}}}
        res = self.requests.post(self.url, json=payload, timeout=5).json()
        preds = []
        for out in (res.get("outputs") or [res]):
            for v in (out.values() if isinstance(out, dict) else []):
                if isinstance(v, dict) and "predictions" in v:
                    preds = v["predictions"]
        for p in preds:
            if p["confidence"] < conf:
                continue
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            yield p["class"], float(p["confidence"]), (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))


# --------------------------------------------------------------------------- ensemble
class Detector:
    """
    models: list of (path, allowed_canonical_classes or None, per_model_conf or None)
    class_conf: per-class confidence thresholds (from the PPE repo's idea)
    """

    def __init__(self, models: list[tuple[str, tuple | None, float | None]], imgsz: int = 640,
                 class_conf: dict[str, float] | None = None, base_conf: float = 0.25):
        self.backends = []
        for path, allowed, mconf in models:
            if path == "roboflow":
                be = RoboflowBackend()
            elif not Path(path).exists() and not path.endswith(".pt"):
                log.warning("model %s missing, skipped", path)
                continue
            elif path.endswith(".onnx"):
                be = OnnxBackend(path, imgsz)
            else:
                try:
                    be = UltralyticsBackend(path, imgsz)
                except Exception as e:  # torch missing / bad weights
                    log.warning("could not load %s (%s), skipped", path, e)
                    continue
            self.backends.append((Path(path).name, be, set(allowed) if allowed else None, mconf))
            log.info("loaded model %s", path)
        if not self.backends:
            raise RuntimeError("No detection model could be loaded. Train one (training/train.py) "
                               "or put models/helmet_unified.onnx in place.")
        self.base_conf = base_conf
        self.class_conf = class_conf or {}
        # ensemble members run concurrently (onnxruntime / torch release the GIL): ~20 % faster
        # per frame than one after the other, because small YOLO models don't use every core alone
        self._pool = ThreadPoolExecutor(len(self.backends), thread_name_prefix="det") if len(self.backends) > 1 else None

    @property
    def model_names(self):
        return [n for n, *_ in self.backends]

    def _run(self, member, bgr) -> list[Det]:
        name, be, allowed, mconf = member
        out = []
        try:
            for label, conf, box in be(bgr, mconf or self.base_conf):
                c = canonical(label)
                if c is None or (allowed and c not in allowed):
                    continue
                if conf < self.class_conf.get(c, self.base_conf):
                    continue
                if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                out.append(Det(c, conf, box, name))
        except Exception as e:
            log.error("backend %s failed: %s", name, e)
        return out

    def __call__(self, bgr: np.ndarray) -> list[Det]:
        return self.detect(bgr)

    def detect(self, bgr: np.ndarray, need: set[str] | None = None) -> list[Det]:
        """need: canonical classes the caller will use. Models that can't produce any of them are
        skipped (helmet-check mode needs no motorcycle/person model: ~2x faster per frame)."""
        members = [m for m in self.backends if need is None or m[2] is None or m[2] & need] or self.backends
        if self._pool and len(members) > 1:
            parts = list(self._pool.map(lambda m: self._run(m, bgr), members))
        else:
            parts = [self._run(m, bgr) for m in members]
        return nms([d for p in parts for d in p])
