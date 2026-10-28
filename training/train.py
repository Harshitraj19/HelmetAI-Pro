"""
train.py — train the unified HelmetAI-Pro detector (helmet / head / motorcycle / license_plate / hardhat).

Profiles
--------
  cpu   : YOLO11n, 416 px, small epochs  -> fits a laptop / 2-core box (hours)
  gpu   : YOLO11s, 640 px, 120 epochs   -> Colab T4 / Kaggle P100 (~1-1.5 h)
  max   : YOLO11m, 640 px, 200 epochs   -> best accuracy, bigger GPU

After training it
  * evaluates on the held-out TEST split (per-class P / R / mAP50 / mAP50-95)
  * exports ONNX (backend runs it with onnxruntime, no PyTorch needed at inference)
  * copies weights to models/helmet_unified.pt / .onnx and writes models/metrics.json

Usage
-----
  python training/train.py --profile gpu
  python training/train.py --profile cpu --epochs 15 --resume   # continue an interrupted run
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "cpu": dict(model="yolo11n.pt", imgsz=416, epochs=25, batch=16, device="cpu", workers=2),
    "gpu": dict(model="yolo11s.pt", imgsz=640, epochs=120, batch=32, device=0, workers=4),
    "max": dict(model="yolo11m.pt", imgsz=640, epochs=200, batch=24, device=0, workers=8),
}

# Augmentations chosen for traffic cameras: night, rain, motion, far-away riders
AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,   # strong brightness jitter -> night/day robustness
    degrees=3.0, translate=0.1, scale=0.6, shear=1.0, perspective=0.0005,
    fliplr=0.5, flipud=0.0,               # no vertical flips: riders are never upside-down
    mosaic=1.0, mixup=0.1, copy_paste=0.1, close_mosaic=10,
    erasing=0.3,                          # occlusion robustness (other vehicles)
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="cpu")
    ap.add_argument("--data", default=str(ROOT / "datasets/helmet_unified/data.yaml"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--model", help="override base model, e.g. yolo11n.pt or a previous best.pt")
    ap.add_argument("--name", default="unified")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of train set (quick runs)")
    ap.add_argument("--cache", default="", help="ram | disk | '' (cache decoded images for speed)")
    ap.add_argument("--no-export", action="store_true", help="skip test eval + ONNX export (intermediate rounds)")
    a = ap.parse_args()

    cfg = dict(PROFILES[a.profile])
    for k in ("epochs", "imgsz", "batch", "model"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)

    run_dir = ROOT / "runs" / a.name
    if a.resume and (run_dir / "weights/last.pt").exists():
        model = YOLO(str(run_dir / "weights/last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(cfg.pop("model"))
        model.train(
            data=a.data, project=str(ROOT / "runs"), name=a.name, exist_ok=True,
            patience=30, cos_lr=True, optimizer="auto", seed=42, plots=True,
            fraction=a.fraction, cache=a.cache or False, **cfg, **AUG,
        )

    best = run_dir / "weights/best.pt"
    model = YOLO(str(best))
    if a.no_export:
        print("round finished:", best)
        return

    # ---- held-out test evaluation
    m = model.val(data=a.data, split="test", imgsz=cfg.get("imgsz", 640), plots=True,
                  project=str(ROOT / "runs"), name=f"{a.name}_test", exist_ok=True)
    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = dict(precision=round(p, 4), recall=round(r, 4),
                                        mAP50=round(ap50, 4), mAP50_95=round(ap, 4))
    metrics = dict(
        split="test", profile=a.profile, imgsz=cfg.get("imgsz"),
        mAP50=round(float(m.box.map50), 4), mAP50_95=round(float(m.box.map), 4),
        precision=round(float(m.box.mp), 4), recall=round(float(m.box.mr), 4),
        per_class=per_class,
    )

    # ---- export + install into the app
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models/helmet_unified.pt")
    onnx = model.export(format="onnx", imgsz=cfg.get("imgsz", 640), simplify=True, dynamic=False)
    shutil.copy2(onnx, ROOT / "models/helmet_unified.onnx")
    (ROOT / "models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
