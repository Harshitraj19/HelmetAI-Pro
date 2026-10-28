"""
Central configuration — every value can be overridden with an environment variable
(or a backend/.env file, which is git-ignored; see .env.example).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent


def _env(key, default):
    v = os.environ.get(key)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


MODELS_DIR = Path(_env("MODELS_DIR", str(ROOT / "models")))
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
EVIDENCE_DIR = DATA_DIR / "evidence"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = ROOT / "samples"
for d in (EVIDENCE_DIR, OUTPUT_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = _env("DATABASE_URL", f"sqlite:///{DATA_DIR / 'helmetai.db'}")

# ---- model selection --------------------------------------------------------
# MODEL_STACK: comma-separated list. Each item  path[:classes] , classes separated by '|'.
# Default (chosen from per-class TEST results, docs/metrics.md):
#   unified model  -> helmet, head, license_plate, hardhat   (2x the base model on helmet / head)
#   COCO YOLO11    -> motorcycle (0.756 n / 0.819 s vs 0.593 unified) + person (rider gating)
# ACCURACY_PROFILE=high uses COCO YOLO11s (slower on CPU). Fallback if nothing is trained yet:
# the base DIP_PROJECT helmet model + COCO motorcycles.
ACCURACY_PROFILE = _env("ACCURACY_PROFILE", "fast")


def default_stack() -> str:
    coco = MODELS_DIR / ("coco_motorcycle_s.onnx" if ACCURACY_PROFILE == "high" else "coco_motorcycle_n.onnx")
    coco_item = f"{coco}:motorcycle|person" if coco.exists() else "yolo11n.pt:motorcycle|person"
    for name in ("helmet_unified.onnx", "helmet_unified.pt"):
        if (MODELS_DIR / name).exists():
            return f"{MODELS_DIR / name}:helmet|head|license_plate|hardhat,{coco_item}"
    return f"{MODELS_DIR / 'teacher_helmet_dip.pt'}:helmet|head,{coco_item}"


MODEL_STACK = _env("MODEL_STACK", "") or default_stack()
IMG_SIZE = _env("IMG_SIZE", 640)
BASE_CONF = _env("BASE_CONF", 0.25)
CLASS_CONF = {                      # per-class thresholds (PPE repo idea)
    "helmet": _env("CONF_HELMET", 0.40),
    "head": _env("CONF_HEAD", 0.40),
    "motorcycle": _env("CONF_MOTORCYCLE", 0.35),
    "license_plate": _env("CONF_PLATE", 0.30),
    "hardhat": _env("CONF_HARDHAT", 0.45),
    "person": _env("CONF_PERSON", 0.40),
}

# ---- runtime defaults (also editable live from the UI Settings page) -------------
RUNTIME_DEFAULTS = {
    "enhancement": _env("ENHANCEMENT", "auto"),     # auto | off | night | fog | low_contrast | blurry
    "privacy_blur": _env("PRIVACY_BLUR", True),
    "plate_ocr": _env("PLATE_OCR", True),
    "save_evidence": _env("SAVE_EVIDENCE", True),
    "camera_id": _env("CAMERA_ID", "CAM-01"),
    "smoothing_window": _env("SMOOTHING_WINDOW", 7),
    "min_track_hits": _env("MIN_TRACK_HITS", 3),
}

# ---- security -----------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")]
# any port on this machine is allowed too (Vite moves to 5174, 5175 ... when 5173 is busy)
CORS_ORIGIN_REGEX = _env("CORS_ORIGIN_REGEX", r"https?://(localhost|127\.0\.0\.1)(:\d+)?")
ADMIN_TOKEN = _env("ADMIN_TOKEN", "")              # if set, destructive endpoints need X-Admin-Token
ALLOW_LOCAL_FILE_SOURCES = _env("ALLOW_LOCAL_FILE_SOURCES", True)   # only files under samples/ & uploads/
PUBLIC_URL = _env("PUBLIC_URL", "http://localhost:8000")
MAX_UPLOAD_MB = _env("MAX_UPLOAD_MB", 300)
# videos / streams wider than this are downscaled before processing (4K at full size is ~6x slower
# on CPU and the detector runs at 640 px anyway). 0 = never downscale.
MAX_FRAME_WIDTH = _env("MAX_FRAME_WIDTH", 1280)
