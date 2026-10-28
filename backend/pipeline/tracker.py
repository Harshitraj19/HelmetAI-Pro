"""
Motorcycle tracker — greedy IoU + centre-distance matching with persistent IDs.

Adapted from the PPE repo's IoUTracker and extended with:
  * centre-distance fallback (fast bikes have low frame-to-frame IoU)
  * frame-edge cleanup (from PPE) so IDs are not wasted on half-visible boxes
  * per-track memory used by the rest of the pipeline:
        - status votes for temporal smoothing (majority over a window, from PPE)
        - motion history -> apparent speed (severity scoring)
        - plate crops / OCR reads for multi-frame plate fusion
        - which violations were already reported (one evidence record per track)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detector import Det, iou


@dataclass
class Track:
    tid: int
    det: Det
    first_frame: int
    last_frame: int
    hits: int = 1
    centers: deque = field(default_factory=lambda: deque(maxlen=15))
    votes: deque = field(default_factory=lambda: deque(maxlen=7))     # per-frame rider summaries
    plate_reads: list = field(default_factory=list)                   # (text, score, crop)
    best_plate_crop: object = None
    best_plate_sharpness: float = 0.0
    reported: set = field(default_factory=set)
    evidence_ids: list = field(default_factory=list)

    @property
    def box(self):
        return self.det.box

    def speed_px(self) -> float:
        """Mean centre displacement per frame, normalised by bike height (scale-invariant)."""
        if len(self.centers) < 3:
            return 0.0
        pts = list(self.centers)
        d = sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5 for i in range(1, len(pts)))
        return d / (len(pts) - 1) / max(1.0, self.det.h)


class Tracker:
    def __init__(self, iou_th: float = 0.25, max_age: int = 20, dist_th: float = 0.6, min_hits: int = 2):
        self.iou_th, self.max_age, self.dist_th, self.min_hits = iou_th, max_age, dist_th, min_hits
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.frame = 0

    def reset(self):
        self.__init__(self.iou_th, self.max_age, self.dist_th, self.min_hits)

    @staticmethod
    def _valid(d: Det, W: int, H: int) -> bool:
        x1, y1, x2, y2 = d.box
        if d.w * d.h < 400:
            return False
        inside = (min(W, x2) - max(0, x1)) * (min(H, y2) - max(0, y1)) / max(1, d.w * d.h)
        return inside >= 0.4

    def update(self, dets: list[Det], W: int, H: int) -> list[Track]:
        self.frame += 1
        unmatched = set(range(len(dets)))
        # 1) IoU matching, best pairs first
        pairs = sorted(((iou(t.box, dets[j].box), tid, j) for tid, t in self.tracks.items() for j in unmatched),
                       reverse=True)
        used_t = set()
        for score, tid, j in pairs:
            if score < self.iou_th:
                break
            if tid in used_t or j not in unmatched:
                continue
            self._assign(tid, dets[j]); used_t.add(tid); unmatched.discard(j)
        # 2) centre-distance fallback for fast movers
        for tid, t in self.tracks.items():
            if tid in used_t or self.frame - t.last_frame > 5:
                continue
            best, bj = None, None
            for j in unmatched:
                d = dets[j]
                dist = ((d.cx - t.det.cx) ** 2 + (d.cy - t.det.cy) ** 2) ** 0.5 / max(1, t.det.h)
                size_ok = 0.5 < (d.w * d.h) / max(1, t.det.w * t.det.h) < 2.0
                if dist < self.dist_th and size_ok and (best is None or dist < best):
                    best, bj = dist, j
            if bj is not None:
                self._assign(tid, dets[bj]); used_t.add(tid); unmatched.discard(bj)
        # 3) new tracks
        for j in unmatched:
            if self._valid(dets[j], W, H):
                t = Track(self.next_id, dets[j], self.frame, self.frame)
                t.centers.append((dets[j].cx, dets[j].cy))
                self.tracks[self.next_id] = t
                self.next_id += 1
        # 4) prune
        for tid in [k for k, t in self.tracks.items()
                    if self.frame - t.last_frame > self.max_age or not self._valid(t.det, W, H)]:
            del self.tracks[tid]
        return [t for t in self.tracks.values() if t.last_frame == self.frame]

    def _assign(self, tid, d):
        t = self.tracks[tid]
        t.det, t.last_frame = d, self.frame
        t.hits += 1
        t.centers.append((d.cx, d.cy))

    def confirmed(self, t: Track) -> bool:
        return t.hits >= self.min_hits
