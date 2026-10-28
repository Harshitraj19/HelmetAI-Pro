"""
Adaptive scene-aware image enhancement  (Novelty #3 — the DIP core of the project)

Each frame is analysed with cheap global statistics and routed to the right
classical DIP operator *before* it reaches the detector:

  scene        test (on luminance / dark channel)              operator
  -----------  ----------------------------------------------  ----------------------------------
  night        mean L < 70                                     adaptive gamma + bilateral denoise + CLAHE
  fog / haze   dark-channel mean > 0.55 and contrast low       dark-channel-prior dehazing (He et al.)
  low-contrast std L < 38                                      CLAHE on L (LAB space)
  blurry       Laplacian variance < 60 or edge anisotropy < 0.5 unsharp masking   (motion blur kills
                                                               gradients along the motion direction)
  normal       otherwise                                       pass-through

Everything is pure OpenCV/NumPy and runs in a few ms at 640 px.

Detection policy (decided on the VALIDATION split, see docs/metrics.md):
  fog / blur / low-contrast frames are enhanced BEFORE detection (fog: +25 % mAP50).
  Night frames go to the detector RAW - it was trained with strong brightness augmentation and
  night enhancement cost motorcycle AP - but the enhanced night frame is still used for display,
  evidence images and plate OCR, where human legibility matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


ENHANCE_FOR_DETECTION = {"fog", "blurry", "low_contrast"}


@dataclass
class SceneStats:
    brightness: float
    contrast: float
    dark_channel: float
    sharpness: float
    anisotropy: float = 1.0
    scene: str = "normal"
    ops: list[str] = field(default_factory=list)

    def as_dict(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def content_region(bgr: np.ndarray) -> np.ndarray:
    """Drop letterbox / pillarbox bars (phone videos) so they don't fake a 'night' scene.
    Bars are uniform, so the bounds are found on a ~320 px strided view (full-res percentiles
    cost ~60 ms per 720p frame - more than the detector) and mapped back to full resolution."""
    H, W = bgr.shape[:2]
    step = max(1, min(H, W) // 320)
    g = bgr[::step, ::step].max(axis=2)
    rows = np.where(g.max(axis=1) > 24)[0]
    cols = np.where(g.max(axis=0) > 24)[0]
    if len(rows) * step < 16 or len(cols) * step < 16:
        return bgr
    # a bar must be uniformly dark along its whole length
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    p98 = np.percentile(g, 98, axis=0)
    inner = np.where(p98 > 24)[0]
    # only trust the bar test if real content is left (a dark night scene can fail it everywhere)
    if len(inner) * step >= 16 and len(inner) < len(p98):
        c0, c1 = inner[0], inner[-1] + 1
    return bgr[r0 * step:min(H, r1 * step), c0 * step:min(W, c1 * step)]


def analyse(bgr: np.ndarray) -> SceneStats:
    bgr = content_region(bgr)
    small = cv2.resize(bgr, (320, max(16, int(320 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    dark = cv2.erode(small.min(axis=2), np.ones((15, 15), np.uint8)).astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    # motion blur removes gradients ALONG the motion direction -> strongly anisotropic edge energy
    gx = float((cv2.Sobel(gray, cv2.CV_64F, 1, 0) ** 2).mean())
    gy = float((cv2.Sobel(gray, cv2.CV_64F, 0, 1) ** 2).mean())
    aniso = min(gx, gy) / max(gx, gy, 1e-6)
    s = SceneStats(float(lum.mean()), float(lum.std()), float(dark.mean()), float(sharp), anisotropy=round(aniso, 3))
    if s.brightness < 70:
        s.scene = "night"
    elif s.dark_channel > 0.55 and s.contrast < 50:
        s.scene = "fog"
    elif s.contrast < 38:
        s.scene = "low_contrast"
    elif s.sharpness < 60 or s.anisotropy < 0.5:
        s.scene = "blurry"
    return s


# ----------------------------------------------------------------- operators
def clahe_lab(bgr: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def adaptive_gamma(bgr: np.ndarray, target: float = 115.0) -> np.ndarray:
    mean = float(np.clip(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean(), 1.0, 254.0))   # log(1) = 0 guard
    gamma = float(np.clip(np.log(target / 255.0) / np.log(mean / 255.0), 0.35, 1.0))
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def dehaze_dcp(bgr: np.ndarray, omega: float = 0.9, t0: float = 0.15, patch: int = 15,
               work_side: int = 640) -> np.ndarray:
    """Single-image dehazing with the Dark Channel Prior + guided-filter refinement.
    The transmission map is smooth by construction, so it is estimated at <= work_side px and
    upsampled ("fast guided filter", He & Sun 2015): ~8x faster on 720p+ frames, same look.
    patch / filter radius are in pixels of a 640 px image and scale with the working size."""
    H, W = bgr.shape[:2]
    s = min(1.0, work_side / max(H, W))
    small = cv2.resize(bgr, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    k = max(3, int(round(patch * max(small.shape[:2]) / 640)) | 1)
    kernel = np.ones((k, k), np.uint8)
    simg = small.astype(np.float32) / 255.0
    dark = cv2.erode(simg.min(axis=2), kernel)
    # atmospheric light = brightest pixels among the top 0.1% of the dark channel
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = simg.reshape(-1, 3)[idx].max(axis=0)
    t = 1.0 - omega * cv2.erode((simg / np.maximum(A, 1e-3)).min(axis=2), kernel)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    r = max(4, int(round(40 * max(small.shape[:2]) / 640)))
    t = _guided_filter(gray, t.astype(np.float32), r=r, eps=1e-3)
    if s < 1:
        t = cv2.resize(t, (W, H), interpolation=cv2.INTER_LINEAR)
    t = np.clip(t, t0, 1.0)[..., None]
    out = (bgr.astype(np.float32) / 255.0 - A) / t + A
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _guided_filter(I, p, r, eps):
    box = lambda x: cv2.boxFilter(x, -1, (r, r))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def unsharp(bgr: np.ndarray, amount: float = 1.2, sigma: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


# ----------------------------------------------------------------- router
def enhance(bgr: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, SceneStats]:
    """mode: 'auto' | 'off' | a forced scene name ('night','fog','low_contrast','blurry')."""
    stats = analyse(bgr)
    if mode == "off":
        stats.ops = []
        return bgr, stats
    scene = stats.scene if mode == "auto" else mode
    out = bgr
    if scene == "night":
        out = cv2.bilateralFilter(adaptive_gamma(out), 5, 40, 40)    # gamma lifts noise -> edge-preserving denoise
        out = clahe_lab(out, clip=2.5)
        stats.ops = ["adaptive_gamma", "bilateral_denoise", "clahe"]
    elif scene == "fog":
        out = clahe_lab(dehaze_dcp(out), clip=1.8)
        stats.ops = ["dark_channel_dehaze", "clahe"]
    elif scene == "low_contrast":
        out = clahe_lab(out)
        stats.ops = ["clahe"]
    elif scene == "blurry":
        out = unsharp(out)
        stats.ops = ["unsharp_mask"]
    stats.scene = scene
    return out, stats


# ----------------------------------------------------------------- degradation (for ablation)
def degrade(bgr: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Synthesise night / fog / blur versions of a clean image for the robustness ablation."""
    rng = np.random.default_rng(seed)
    img = bgr.astype(np.float32)
    if kind == "night":
        img = 255 * np.power(img / 255, 2.2) * 0.45
        img += rng.normal(0, 6, img.shape)
    elif kind == "fog":
        h, w = img.shape[:2]
        depth = np.linspace(0.45, 0.85, h, dtype=np.float32)[:, None, None]
        img = img * (1 - depth) + 225 * depth
    elif kind == "blur":
        k = np.zeros((9, 9), np.float32)
        k[4, :] = 1 / 9
        img = cv2.filter2D(img, -1, k)
    return np.clip(img, 0, 255).astype(np.uint8)
