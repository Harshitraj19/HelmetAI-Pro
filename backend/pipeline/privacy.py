"""
Privacy by design  (Novelty #6, aligned with India's DPDP Act 2023: data minimisation)

Only violators are kept identifiable. Applied to every OUTPUT frame (stream, video, evidence):
  * heads / helmets of compliant riders              -> blurred
  * heads of people not on a motorcycle (pedestrians) -> blurred
  * number plates of compliant motorcycles           -> blurred
  * violators' heads and plates                       -> left clear (they are the evidence)
"""
from __future__ import annotations

import cv2
import numpy as np


def blur_region(img: np.ndarray, box, pad: float = 0.15, strength: int = 31):
    x1, y1, x2, y2 = box
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    roi = img[y1:y2, x1:x2]
    # pixelate + blur: irreversible and cheap
    small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_LINEAR)
    roi[:] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST),
                              (strength | 1, strength | 1), 0)


def apply(img: np.ndarray, scenes, loose_heads, violating_ids: set[int]) -> int:
    """scenes: list of (track_id, BikeScene). Returns number of regions anonymised."""
    n = 0
    for tid, s in scenes:
        if tid in violating_ids:
            continue
        for r in s.riders:
            blur_region(img, r.det.box); n += 1
        if s.plate is not None:
            blur_region(img, s.plate.box, pad=0.05); n += 1
    for h in loose_heads:
        blur_region(img, h.box); n += 1
    return n
