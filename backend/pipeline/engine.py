"""
HelmetAI-Pro processing engine — glues every module together.

    frame ─► enhance (adaptive DIP) ─► detect (ensemble) ─► track motorcycles ─► associate riders
          ─► temporal smoothing ─► rule engine ─► plate fusion ─► evidence + hash chain ─► privacy blur
          ─► annotation / HUD ─► (jpeg, result-json)

Two operating modes
    traffic  (default)  motorcycle-centric rules: no helmet, hard-hat, helmet carried, triple riding
    site                 workplace PPE (from the PPE repo): every tracked head must wear a helmet/hard-hat

One `Session` per image / video / stream keeps its own tracker and counters, so several
sources can be processed at the same time without mixing IDs.

For videos / streams a finished violation is turned into evidence (plate OCR, blurred evidence
images, SHA-256 chain, DB row) on ONE background worker: the stream never stalls on it, and a
single worker keeps records in submission order, so the hash chain stays linear.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import privacy, severity
from .detector import Det, Detector
from .enhance import ENHANCE_FOR_DETECTION, enhance
from .plate_ocr import PlateReader
from .riders import BikeScene, associate, rider_zone, smooth
from .tracker import Tracker

log = logging.getLogger("engine")

GREEN, RED, ORANGE, YELLOW, BLUE, WHITE, GREY = (60, 200, 80), (40, 40, 230), (0, 150, 255), (0, 215, 255), (230, 160, 40), (255, 255, 255), (150, 150, 150)
STATUS_COLOR = {"HELMET": GREEN, "NO_HELMET": RED, "INVALID_HELMET": ORANGE}
VIOLATION_TEXT = {"NO_HELMET": "NO HELMET", "INVALID_HELMET": "HARD-HAT", "HELMET_NOT_WORN": "HELMET NOT WORN",
                  "TRIPLE_RIDING": "TRIPLE RIDING", "NO_HEADGEAR": "NO HEADGEAR"}


@dataclass
class Session:
    source: str
    camera_id: str
    image_mode: bool = False
    mode: str = "traffic"
    tracker: Tracker = None
    frames: int = 0
    started: float = field(default_factory=time.time)
    seen_tracks: set = field(default_factory=set)
    violators: set = field(default_factory=set)
    riders_by_track: dict = field(default_factory=dict)      # tid -> stable rider summary
    violation_records: list = field(default_factory=list)
    scene_counts: dict = field(default_factory=dict)
    fps: float = 0.0
    pending: list = field(default_factory=list)             # evidence jobs still running (futures)

    def counters(self) -> dict:
        riders = sum(r["riders"] for r in self.riders_by_track.values())
        helmets = sum(r["helmets"] for r in self.riders_by_track.values())
        return {
            "frames": self.frames, "unique_motorcycles": len(self.seen_tracks),
            "unique_riders": riders, "riders_with_helmet": helmets,
            "violating_vehicles": len(self.violators), "violations_recorded": len(self.violation_records),
            "compliance_rate": round(100 * helmets / riders, 1) if riders else None,
            "fps": round(self.fps, 1), "scenes": self.scene_counts,
        }


class Engine:
    def __init__(self, detector: Detector, settings: dict, store=None, notify=None):
        self.det = detector
        self.settings = settings          # live dict, editable from the API
        self.store = store                # module with add_violation / log_detections / repeat_count
        self.notify = notify or (lambda msg: None)
        self.plates = PlateReader()
        self._evidence_pool = ThreadPoolExecutor(1, thread_name_prefix="evidence")

    # ------------------------------------------------------------------ sessions
    def session(self, source: str, image_mode=False, mode: str | None = None, camera_id: str | None = None) -> Session:
        """mode: traffic | site | auto. 'auto' (images only) picks traffic when a motorcycle is in the
        picture and the every-person helmet check otherwise."""
        mode = mode or self.settings.get("mode", "traffic")
        if mode == "auto" and not image_mode:
            mode = "traffic"              # a stream's mode can't flip per frame (tracks are bikes vs heads)
        s = Session(source=source, camera_id=camera_id or self.settings["camera_id"], image_mode=image_mode,
                    mode=mode)
        s.tracker = Tracker(min_hits=1 if image_mode else int(self.settings["min_track_hits"]))
        return s

    # ------------------------------------------------------------------ main entry
    def process(self, frame: np.ndarray, s: Session) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        s.frames += 1
        H, W = frame.shape[:2]
        enh, scene = enhance(frame, self.settings["enhancement"])
        s.scene_counts[scene.scene] = s.scene_counts.get(scene.scene, 0) + 1
        # validated policy: enhance-for-detection only where it helps (see enhance.py)
        det_in = enh if scene.scene in ENHANCE_FOR_DETECTION else frame
        detect = getattr(self.det, "detect", None)
        if detect and s.mode == "site":        # helmet check needs no motorcycle / person model
            dets = detect(det_in, {"head", "helmet", "hardhat"})
        else:
            dets = detect(det_in) if detect else self.det(det_in)
        if s.mode == "auto":
            s.mode = "traffic" if any(d.cls == "motorcycle" for d in dets) else "site"

        if s.mode == "site":
            subjects = [d for d in dets if d.cls in ("head", "helmet", "hardhat")]
        else:
            subjects = [d for d in dets if d.cls == "motorcycle"]

        before = dict(s.tracker.tracks)
        if s.image_mode:
            s.tracker.reset()
        live = s.tracker.update(subjects, W, H)
        ended = [t for tid, t in before.items() if tid not in s.tracker.tracks]

        if s.mode == "site":
            scenes, loose = self._site_scenes(live), []
        else:
            sc, loose = associate(dets, [t.det for t in live])
            scenes = list(zip([t.tid for t in live], sc))
        by_id = {t.tid: t for t in live}

        # ---- per-track temporal reasoning
        frame_bikes = []
        for tid, bs in scenes:
            t = by_id[tid]
            summ = bs.summary()
            t.votes = t.votes if t.votes.maxlen == self.settings["smoothing_window"] else \
                type(t.votes)(t.votes, maxlen=int(self.settings["smoothing_window"]))
            t.votes.append(summ)
            stable = summ if s.image_mode else smooth(t.votes)
            confirmed = s.tracker.confirmed(t)
            done = getattr(t, "finalised", False)      # evidence already taken: no more crops / frame copies
            if bs.plate is not None and self.settings["plate_ocr"] and not done:
                self.plates.observe(t, enh, bs.plate.box)
            if confirmed:
                s.seen_tracks.add(tid)
                prev = s.riders_by_track.get(tid)
                if prev is None or stable.get("riders", 0) >= prev["riders"]:
                    s.riders_by_track[tid] = {k: stable.get(k, 0) for k in ("riders", "helmets", "no_helmet")}
            violations = stable.get("violations", []) if confirmed else []
            if violations:
                s.violators.add(tid)
                if not done:
                    others = [(o_tid, o_bs) for o_tid, o_bs in scenes if o_tid != tid]
                    self._remember_best(t, frame, bs, summ, violations, scene, others, loose)
                if not hasattr(t, "pending_since"):
                    t.pending_since = s.frames
            frame_bikes.append({"track_id": tid, "riders": summ["riders"], "helmets": summ["helmets"],
                                "no_helmet": summ["no_helmet"], "violations": violations,
                                "box": list(bs.bike.box)})

        # ---- finalise violations: after enough frames to collect plate crops, or when the track ends
        new_records = []
        for t in list(live) + ended:
            if not hasattr(t, "pending_since") or getattr(t, "finalised", False):
                continue
            if s.image_mode or t in ended or s.frames - t.pending_since >= 10:
                self._submit(t, s, new_records)
        new_records += self._collect(s)
        for t in ended:
            self._log_track(t, s)

        # ---- output frame: privacy first, then drawings
        out = enh.copy() if self.settings.get("show_enhanced", True) else frame.copy()
        violating = {tid for tid, _ in scenes if tid in s.violators}
        if self.settings["privacy_blur"]:
            privacy.apply(out, scenes, loose if s.mode == "traffic" else [], violating)
        self._draw(out, scenes, by_id, violating, loose, s, scene)

        dt = time.time() - t0
        s.fps = 0.9 * s.fps + 0.1 * (1 / max(dt, 1e-3)) if s.fps else 1 / max(dt, 1e-3)
        result = {
            "frame": s.frames,
            "mode": s.mode,
            "detections": [d.as_dict() for d in dets],
            "vehicles": frame_bikes,
            "scene": scene.as_dict(),
            "new_violations": new_records,
            "counters": s.counters(),
            "latency_ms": round(dt * 1000, 1),
        }
        return out, result

    def finish(self, s: Session):
        """Flush tracks that are still open at the end of a video/stream and wait for every
        evidence job of the session, so the final counters are complete."""
        recs = []
        for t in list(s.tracker.tracks.values()):
            if hasattr(t, "pending_since") and not getattr(t, "finalised", False):
                self._submit(t, s, recs)
            self._log_track(t, s)
        return recs + self._collect(s, wait=True)

    def _submit(self, t, s: Session, out: list):
        job = self._snapshot(t)
        if s.image_mode:                        # the HTTP response needs the record right away
            out.append(self._finalise(job, s))
        else:
            s.pending.append(self._evidence_pool.submit(self._finalise, job, s))

    @staticmethod
    def _collect(s: Session, wait: bool = False) -> list:
        done, keep = [], []
        for f in s.pending:
            if wait or f.done():
                try:
                    done.append(f.result())
                except Exception:
                    log.exception("evidence job failed")
            else:
                keep.append(f)
        s.pending = keep
        return done

    # ------------------------------------------------------------------ helpers
    def _site_scenes(self, live):
        """Workplace mode: each tracked head-level object is its own 'scene'."""
        from .riders import Rider
        out = []
        for t in live:
            bs = BikeScene(t.det)
            status = {"helmet": "HELMET", "hardhat": "HELMET", "head": "NO_HELMET"}[t.det.cls]
            bs.riders.append(Rider(status, t.det))
            out.append((t.tid, bs))
        return out

    def _remember_best(self, t, frame, bs: BikeScene, summ, violations, scene, others, loose):
        score = sum(r.det.conf for r in bs.riders) + bs.bike.conf + (0.5 if bs.plate else 0)
        if score <= getattr(t, "best_score", -1):
            return
        t.best_score = score
        t.best_scene, t.best_summary, t.best_violations, t.best_env = bs, summ, list(violations), scene.scene
        t.best_frame = frame.copy()
        t.best_others, t.best_loose = others, list(loose)

    @staticmethod
    def _snapshot(t) -> dict:
        """Everything the evidence job needs, detached from the live track (which keeps being
        updated by the processing thread). The track drops its frame copy right away."""
        t.finalised = True
        job = dict(tid=t.tid, scene=t.best_scene, summary=t.best_summary, violations=list(t.best_violations),
                   env=t.best_env, frame=t.best_frame, others=t.best_others, loose=t.best_loose,
                   plate_reads=list(t.plate_reads), plate_crop=t.best_plate_crop, speed=t.speed_px())
        t.best_frame, t.plate_reads = None, []
        return job

    def _finalise(self, job: dict, s: Session):
        bs: BikeScene = job["scene"]
        summary = job["summary"]
        violations = job["violations"]
        if s.mode == "site":
            violations = ["NO_HEADGEAR" if v == "NO_HELMET" else v for v in violations]
        plate = {"plate": "", "raw_reads": [], "agreement": 0.0, "validity": 0.0}
        if self.settings["plate_ocr"] and job["plate_reads"]:
            plate = self.plates.read_crops(job["plate_reads"])
        plate_conf = round(plate["agreement"] * 0.5 + plate["validity"] * 0.5, 3) if plate["plate"] else None
        unprotected = summary["no_helmet"] + summary.get("invalid_helmet", 0)
        repeats = self.store.repeat_count(plate["plate"]) if (self.store and plate["plate"]) else 0
        sev = severity.score(violations, unprotected, job["speed"], repeats)

        # evidence images: violator clear, everyone else blurred
        frame = job["frame"]
        ev = frame.copy()
        if self.settings["privacy_blur"]:   # only the subject of this record stays identifiable
            privacy.apply(ev, job["others"], job["loose"], set())
        self._draw_evidence(ev, bs, violations, plate["plate"])
        zx1, zy1, zx2, zy2 = rider_zone(bs.bike)
        H, W = frame.shape[:2]
        crop = frame[max(0, int(zy1)):min(H, int(zy2)), max(0, int(zx1)):min(W, int(zx2))].copy()
        fields = dict(source=s.source, camera_id=s.camera_id, track_id=job["tid"], violation_types=violations,
                      riders=summary["riders"], helmets=summary["helmets"],
                      plate=plate["plate"] or None, plate_confidence=plate_conf, plate_reads=plate["raw_reads"],
                      severity=sev, severity_level=severity.level(sev), scene=job["env"])
        s.violation_records.append(fields)
        if not (self.store and self.settings["save_evidence"]):
            return dict(fields, id=None)
        rec = self.store.add_violation(fields, ev, crop, job["plate_crop"])
        self.notify({"type": "violation", "record": {k: rec[k] for k in (
            "id", "created_at", "violation_list", "plate", "riders", "severity", "severity_level", "camera_id", "source")},
            # kept for the base dashboard's alert widget
            "timestamp": rec["created_at"], "violations": [{"class": v} for v in rec["violation_list"]]})
        return rec

    def _log_track(self, t, s: Session):
        """One detection row per rider of a finished track -> unique-rider statistics."""
        if not self.store or getattr(t, "logged", False) or not s.tracker.confirmed(t):
            return
        t.logged = True
        stable = smooth(t.votes) if not s.image_mode else (t.votes[-1] if t.votes else {})
        rows = []
        n_h, n_n = stable.get("helmets", 0), stable.get("no_helmet", 0)
        for cls, n in (("helmet", n_h), ("head", n_n)):
            for _ in range(n):
                rows.append(dict(object_class="With Helmet" if cls == "helmet" else "Without Helmet",
                                 confidence=round(t.det.conf, 3), image_path=s.source, source=s.source,
                                 camera_id=s.camera_id, track_id=t.tid, bbox=",".join(map(str, t.det.box))))
        self.store.log_detections(rows)

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _label(img, text, x, y, color, scale=0.5):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y = max(th + 6, y)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)

    def _draw(self, img, scenes, by_id, violating, loose, s: Session, scene):
        for tid, bs in scenes:
            bad = tid in violating
            col = RED if bad else GREEN
            x1, y1, x2, y2 = bs.bike.box
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            stable = smooth(by_id[tid].votes) if not s.image_mode else bs.summary()
            tag = f"#{tid} riders:{bs.n_riders}"
            if stable.get("violations"):
                tag += " | " + ", ".join(VIOLATION_TEXT.get(v, v) for v in stable["violations"])
            self._label(img, tag, x1, y1 - 2, col)
            for r in bs.riders:
                c = STATUS_COLOR[r.status]
                a1, b1, a2, b2 = r.det.box
                cv2.rectangle(img, (a1, b1), (a2, b2), c, 2)
                self._label(img, {"HELMET": "helmet", "NO_HELMET": "no helmet", "INVALID_HELMET": "hard-hat"}[r.status]
                            + f" {r.det.conf:.2f}", a1, b1 - 2, c, 0.42)
            for h in bs.carried_helmets:
                a1, b1, a2, b2 = h.box
                cv2.rectangle(img, (a1, b1), (a2, b2), YELLOW, 2)
                self._label(img, "helmet carried", a1, b2 + 16, YELLOW, 0.42)
            if bs.plate is not None:
                a1, b1, a2, b2 = bs.plate.box
                cv2.rectangle(img, (a1, b1), (a2, b2), BLUE, 2)
        for h in loose:
            if s.mode == "traffic":
                a1, b1, a2, b2 = h.box
                cv2.rectangle(img, (a1, b1), (a2, b2), GREY, 1)
        # HUD (kept from the base project's LIVE SCORE banner, extended; adapts to narrow phone videos)
        c = s.counters()
        unit = "people" if s.mode == "site" else "bikes"
        parts = ["HELMET CHECK" if s.mode == "site" else "TRAFFIC", f"scene:{scene.scene}", f"{unit}:{c['unique_motorcycles']} riders:{c['unique_riders']}"
                 if s.mode == "traffic" else f"{unit}:{c['unique_motorcycles']}", f"violators:{c['violating_vehicles']}"]
        if not s.image_mode:
            parts.append(f"fps:{c['fps']}")
        W = img.shape[1]
        scale = 0.55 if W >= 640 else 0.38
        lines, cur = [], ""
        for p in parts:                       # wrap onto a second line when the frame is narrow
            trial = f"{cur} | {p}" if cur else p
            if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > W - 16 and cur:
                lines.append(cur); cur = p
            else:
                cur = trial
        lines.append(cur)
        lh = int(26 * scale / 0.55) + 4
        cv2.rectangle(img, (0, 0), (W, 8 + lh * len(lines)), (20, 20, 20), -1)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (8, lh * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)
        if s.mode == "site":
            self._status_banner(img, scenes)

    @staticmethod
    def _status_banner(img, scenes):
        """Helmet-check mode: one big verdict for the current frame, burnt into the frame (and so
        into recorded MP4s too): red NO HELMET / green HELMET ON / grey nobody in view."""
        bad = sum(r.status != "HELMET" for _, bs in scenes for r in bs.riders)
        good = sum(r.status == "HELMET" for _, bs in scenes for r in bs.riders)
        if bad:
            text, col = f"NO HELMET - {bad} person(s)" + (f", {good} with helmet" if good else ""), RED
        elif good:
            text, col = f"HELMET ON - {good} person(s)", GREEN
        else:
            text, col = "No person in view", GREY
        H, W = img.shape[:2]
        scale = max(0.6, min(1.4, W / 900))
        th = max(2, int(scale * 2))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        pad = int(12 * scale)
        y2 = H - pad
        y1 = y2 - tht - 2 * pad
        x1 = max(0, (W - tw) // 2 - pad)
        cv2.rectangle(img, (x1, y1), (min(W, x1 + tw + 2 * pad), y2), col, -1)
        cv2.putText(img, text, (x1 + pad, y2 - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, th, cv2.LINE_AA)

    def _draw_evidence(self, img, bs: BikeScene, violations, plate_text):
        x1, y1, x2, y2 = map(int, rider_zone(bs.bike))
        H, W = img.shape[:2]
        cv2.rectangle(img, (max(0, x1), max(0, y1)), (min(W - 1, x2), min(H - 1, y2)), RED, 3)
        for r in bs.riders:
            a1, b1, a2, b2 = r.det.box
            cv2.rectangle(img, (a1, b1), (a2, b2), STATUS_COLOR[r.status], 2)
        if bs.plate is not None:
            a1, b1, a2, b2 = bs.plate.box
            cv2.rectangle(img, (a1, b1), (a2, b2), BLUE, 2)
        text = " + ".join(VIOLATION_TEXT.get(v, v) for v in violations)
        if plate_text:
            text += f"  [{plate_text}]"
        self._label(img, text, max(0, x1), max(20, y1), RED, 0.6)
