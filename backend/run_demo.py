"""
run_demo.py — process every sample image and video with the real model and produce the deliverable outputs.

    python run_demo.py [--max-frames 300] [--out ../samples/outputs/demo]

Outputs: annotated images, annotated MP4s, evidence records (DB + images), e-Challan PDFs,
evidence-chain verification and summary.json / summary.md.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default=str(HERE.parent / "samples/outputs/demo"))
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame (CPU speed-up)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DATA_DIR", str(out / "data"))

    import config
    import store
    from main import SETTINGS, build_engine
    from models import init_db
    from pipeline.challan import build_challan
    from pipeline.videoio import open_writer

    init_db()
    eng = build_engine()
    print("models:", eng.det.model_names)
    media = sorted((config.SAMPLES_DIR / "media").iterdir())
    summary = {"models": eng.det.model_names, "images": [], "videos": []}

    for p in media:
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = cv2.imread(str(p))
            s = eng.session(p.name, image_mode=True)
            t = time.time()
            vis, res = eng.process(img, s)
            res["new_violations"] += eng.finish(s)
            cv2.imwrite(str(out / f"{p.stem}_result.jpg"), vis)
            item = {"file": p.name, "latency_ms": round((time.time() - t) * 1000), "scene": res["scene"]["scene"],
                    "detections": {}, "vehicles": res["vehicles"],
                    "violations": [{k: v.get(k) for k in ("id", "violation_list", "plate", "severity")} for v in res["new_violations"]]}
            for d in res["detections"]:
                item["detections"][d["class"]] = item["detections"].get(d["class"], 0) + 1
            summary["images"].append(item)
            print(f"[img] {p.name:24} {item['detections']}  vehicles={[(v['riders'], v['violations']) for v in res['vehicles']]}")
        elif p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            s = eng.session(p.name)
            writer, n, t = None, 0, time.time()
            while n < a.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if (n - 1) % a.stride:
                    continue
                if frame.shape[1] > 1280:   # keep CPU time sane
                    frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
                vis, _ = eng.process(frame, s)
                if writer is None:
                    writer = open_writer(out / f"{p.stem}_result.mp4", fps / a.stride, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            eng.finish(s)
            cap.release()
            if writer:
                writer.release()
            c = s.counters()
            c["wall_seconds"] = round(time.time() - t, 1)
            summary["videos"].append({"file": p.name, **c})
            print(f"[vid] {p.name:24} {c}")

    # challans for every violation + chain verification
    from models import SessionLocal, Violation
    db = SessionLocal()
    rows = [store.row_to_dict(v) for v in db.query(Violation).order_by(Violation.id).all()]
    db.close()
    (out / "challans").mkdir(exist_ok=True)
    for r in rows:
        (out / "challans" / f"challan_HAI-{r['id']:06d}.pdf").write_bytes(
            build_challan(r, config.EVIDENCE_DIR, f"{config.PUBLIC_URL}/api/evidence/verify"))
    summary["violations_total"] = len(rows)
    summary["plates_read"] = [r["plate"] for r in rows if r["plate"]]
    summary["chain"] = store.verify()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("violations_total", "plates_read", "chain")}, indent=2))


if __name__ == "__main__":
    main()
