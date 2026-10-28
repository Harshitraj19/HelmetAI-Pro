"""
Number-plate recognition with multi-frame fusion  (Novelty #4)

Siddhesh's notebook did: crop -> gray -> Otsu -> resize h=32 -> Tesseract psm 6, on ONE frame
(it read "MHOZEP 1543" for "MH02EP1543"). We extend that into a 4-stage pipeline:

  1. Frame selection   keep the sharpest plate crops of a tracked bike (Laplacian variance)
  2. DIP restoration   upscale (cubic) -> CLAHE -> bilateral denoise -> deskew (minAreaRect)
                       -> several binarisations (Otsu, inverted Otsu, adaptive Gaussian)
  3. OCR               Tesseract LSTM, single-line mode, A-Z0-9 whitelist, best variant by confidence
  4. Syntax repair     align to Indian plate grammar and fix look-alikes BY POSITION
                       (letter slot: 0->O 1->I 8->B 5->S 2->Z 6->G, digit slot: O->0 I->1 B->8 ...)
  5. Fusion            character-wise weighted vote across all reads of the same track

Formats handled: standard  SS DD L{0,3} DDDD   (e.g. MH 02 EP 1543, DL 3C AB 1234 variants)
                 Bharat    DD BH DDDD L{1,2}   (e.g. 22 BH 1234 AA)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np

log = logging.getLogger("plate_ocr")


def _find_tesseract() -> bool:
    """pytesseract only wraps the tesseract binary: check it really exists ONCE, so a missing
    binary costs nothing per crop. Looks at TESSERACT_CMD, PATH and the default Windows installs."""
    try:
        import pytesseract
    except ImportError:
        log.warning("pytesseract not installed - plate OCR disabled")
        return False
    cands = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
             r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
             os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")]
    for c in cands:
        if c and os.path.isfile(c):
            pytesseract.pytesseract.tesseract_cmd = c
            try:
                log.info("plate OCR: tesseract %s (%s)", pytesseract.get_tesseract_version(), c)
                return True
            except Exception:
                continue
    log.warning("tesseract binary not found - plate OCR disabled (install it or set TESSERACT_CMD)")
    return False


_TESS = _find_tesseract()
if _TESS:
    import pytesseract

OCR_AVAILABLE = _TESS

STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
    "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UP", "WB",
}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "7", "Z": "2", "S": "5",
            "B": "8", "G": "6", "A": "4"}

# templates: L = letter, D = digit
TEMPLATES = [
    "LLDDLLDDDD", "LLDDLDDDD", "LLDDLLLDDDD", "LLDLLLDDDD", "LLDDDDDD", "LLDLLDDDD", "LLDLDDDD",
    "DDLLDDDDLL", "DDLLDDDDL",          # BH series
]
SPACING = {"LLDDLLDDDD": (2, 4, 6), "LLDLLLDDDD": (2, 4, 6), "LLDDLDDDD": (2, 4, 5), "LLDDLLLDDDD": (2, 4, 7), "LLDDDDDD": (2, 4),
           "LLDLLDDDD": (2, 3, 5), "LLDLDDDD": (2, 3, 4), "DDLLDDDDLL": (2, 4, 8), "DDLLDDDDL": (2, 4, 8)}


# --------------------------------------------------------------------------- DIP helpers
def sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def crop_plate(frame, box, margin: float = 0.12):
    x1, y1, x2, y2 = box
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    H, W = frame.shape[:2]
    return frame[max(0, y1 - mh):min(H, y2 + mh), max(0, x1 - mw):min(W, x2 + mw)].copy()


def _deskew(gray):
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(thr)
    if pts is None or len(pts) < 30:
        return gray
    angle = cv2.minAreaRect(pts)[-1]
    angle = angle - 90 if angle > 45 else angle
    if abs(angle) < 1 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def refine_plate(crop):
    """Cut the crop down to the bright plate rectangle (drops bike body, bolts, frame)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    H, W = g.shape
    thr = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, W // 12), max(3, H // 4))))
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar, area = w / max(1, h), w * h
        if 1.8 <= ar <= 8 and area >= 0.25 * W * H and (best is None or area > best[2] * best[3]):
            best = (x, y, w, h)
    if best is None:
        return crop
    x, y, w, h = best
    return crop[y:y + h, x:x + w]


def keep_characters(binary):
    """binary: dark text (0) on white (255). Keep only character-like connected components."""
    inv = 255 - binary
    H, W = inv.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = np.full_like(binary, 255)
    kept = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        if touches or not (0.35 * H <= h <= 0.95 * H) or w > 0.2 * W or a < 0.02 * h * h:
            continue
        out[lab == i] = 0
        kept += 1
    return out if kept >= 4 else binary


def _binarise(g):
    otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if (otsu == 0).mean() > 0.5:          # light text on dark plate -> make text dark
        otsu = 255 - otsu
    return otsu


def preprocess_variants(crop) -> list[np.ndarray]:
    """Multi-scale, multi-binarisation variants; the OCR keeps the most confident valid read."""
    crop = refine_plate(crop)
    g0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    pad = lambda im: cv2.copyMakeBorder(im, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    out = []
    for target in (48, 72):
        scale = target / max(1, g0.shape[0])
        g = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
        g = cv2.bilateralFilter(g, 7, 40, 40)
        g = _deskew(g)
        b = _binarise(g)
        adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        out += [pad(keep_characters(b)), pad(keep_characters(adapt))]
    out.append(pad(g))
    return out


# --------------------------------------------------------------------------- grammar
def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def repair(raw: str) -> tuple[str, float]:
    """Fit raw OCR text to the best plate template; returns (plate, validity 0..1)."""
    s = clean(raw)
    if len(s) < 6:
        return s, 0.0
    best, best_score = s, 0.0
    for t in TEMPLATES:
        for start in range(0, max(1, len(s) - len(t) + 1)):     # tolerate 1-2 junk chars at the ends
            seg = s[start:start + len(t)]
            if len(seg) != len(t):
                continue
            out, ok = [], 0
            for ch, slot in zip(seg, t):
                if slot == "L":
                    c = ch if ch.isalpha() else TO_LETTER.get(ch)
                else:
                    c = ch if ch.isdigit() else TO_DIGIT.get(ch)
                if c is None:
                    break
                ok += c == ch
                out.append(c)
            else:
                cand = "".join(out)
                single = t[2] == "D" and t[3] == "L"          # single-digit district (DL, old GJ plates...)
                if t.startswith("DD") and cand[2:4] != "BH":
                    continue                                     # Bharat series must literally read 'BH'
                if t == "LLDLLLDDDD" and cand[:2] != "DL":
                    continue                                     # 3-letter series after 1 digit: Delhi only
                score = ok / len(t) * 0.7 + 0.3 * (cand[:2] in STATES or t.startswith("DD")) - 0.06 * (len(s) - len(t))
                if single and cand[:2] != "DL":
                    score -= 0.08
                score *= min(1.0, len(t) / 10)
                if score > best_score:
                    best, best_score = cand, score
    return best, round(max(0.0, best_score), 3)


def pretty(plate: str) -> str:
    for t, cuts in SPACING.items():
        if len(t) == len(plate) and all((c.isalpha() if s == "L" else c.isdigit()) for c, s in zip(plate, t)):
            parts, prev = [], 0
            for c in cuts:
                parts.append(plate[prev:c]); prev = c
            parts.append(plate[prev:])
            return " ".join(p for p in parts if p)
    return plate


# --------------------------------------------------------------------------- OCR
def ocr_crop(crop) -> tuple[str, float, float]:
    """Returns (plate, ocr_confidence 0..1, grammar validity 0..1) for ONE crop."""
    if not _TESS or crop is None or crop.size == 0:
        return "", 0.0, 0.0
    best = ("", 0.0, 0.0)
    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for im in preprocess_variants(crop):
        try:
            d = pytesseract.image_to_data(im, config=cfg, output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            continue
        text = "".join(w for w, _ in words)
        conf = sum(c for _, c in words) / len(words) / 100.0
        plate, valid = repair(text)
        if conf * 0.5 + valid > best[1] * 0.5 + best[2]:
            best = (plate, conf, valid)
        if valid >= 0.95 and conf >= 0.85:      # grammar-perfect, confident read: other variants can't beat it
            break
    return best


def fuse(reads: list[tuple[str, float]]) -> tuple[str, float]:
    """Character-level weighted vote across frames. reads = [(plate, weight)]."""
    reads = [(p, w) for p, w in reads if len(p) >= 6]
    if not reads:
        return "", 0.0
    by_len = defaultdict(float)
    for p, w in reads:
        by_len[len(p)] += w
    length = max(by_len, key=by_len.get)
    same = [(p, w) for p, w in reads if len(p) == length]
    votes = [defaultdict(float) for _ in range(length)]
    for p, w in same:
        for i, ch in enumerate(p):
            votes[i][ch] += w
    plate = "".join(max(v, key=v.get) for v in votes)
    agreement = float(np.mean([max(v.values()) / sum(v.values()) for v in votes]))
    return plate, round(agreement, 3)


class PlateReader:
    """Keeps the K sharpest crops per track and OCRs lazily (OCR is the slow part)."""

    def __init__(self, keep: int = 5, min_height: int = 12):
        self.keep, self.min_height = keep, min_height

    def observe(self, track, frame, plate_box):
        crop = crop_plate(frame, plate_box)
        if crop.shape[0] < self.min_height:
            return
        s = sharpness(crop) * min(1.0, crop.shape[0] / 40)      # prefer sharp AND big crops
        track.plate_reads.append((s, crop))
        track.plate_reads.sort(key=lambda r: r[0], reverse=True)
        del track.plate_reads[self.keep:]
        if s > track.best_plate_sharpness:
            track.best_plate_sharpness, track.best_plate_crop = s, crop

    def read(self, track) -> dict:
        return self.read_crops(track.plate_reads)

    def read_crops(self, reads) -> dict:
        """reads: [(sharpness, crop)] - a snapshot, so OCR can run off the processing thread."""
        results = []
        for s, crop in reads:
            plate, conf, valid = ocr_crop(crop)
            if plate:
                results.append((plate, (conf + 0.05) * (0.3 + valid)))
        plate, agreement = fuse(results)
        _, validity = repair(plate) if plate else ("", 0.0)
        return {"plate": pretty(plate) if plate else "", "raw_reads": [p for p, _ in results],
                "frames_used": len(results), "agreement": agreement, "validity": validity}
