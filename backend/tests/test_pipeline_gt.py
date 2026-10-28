"""
End-to-end pipeline test on hand-annotated real photos (no model needed).

Feeds the TEST split's human labels to the engine instead of a model, so the rule engine,
plate OCR, evidence chain, tamper detection, challan and privacy code are verified
independently of model accuracy. Run from backend/:

    python tests/test_pipeline_gt.py [--n 40]
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "backend"))
TMP = Path(tempfile.mkdtemp(prefix="hai_test_"))
os.environ["DATA_DIR"] = str(TMP)                     # isolated DB + evidence folder

import config  # noqa: E402
import store  # noqa: E402
from models import init_db  # noqa: E402
from pipeline.challan import build_challan  # noqa: E402
from pipeline.detector import Det  # noqa: E402
from pipeline.engine import Engine  # noqa: E402
from pipeline.videoio import open_writer  # noqa: E402

from tests.test_rules import INDIA_1, RIDERS_1  # noqa: E402

MEDIA = config.ROOT / "samples/media"


class GTDetector:
    """Pretends to be a detector: returns hand-annotated boxes (shifted by `offset` for video)."""
    dets, offset = [], (0, 0)

    def __call__(self, bgr):
        dx, dy = self.offset
        return [Det(d.cls, d.conf, (d.box[0] + dx, d.box[1] + dy, d.box[2] + dx, d.box[3] + dy), "gt") for d in self.dets]


def main():
    init_db()
    det = GTDetector()
    settings = dict(config.RUNTIME_DEFAULTS, mode="traffic", enhancement="auto")
    eng = Engine(det, settings, store=store)
    out_dir = config.ROOT / "samples/outputs/pipeline_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, "bikes": 0, "riders": 0, "violations": 0, "plates_read": []}
    # --- still images
    for name, ann in (("riders_1.jpg", RIDERS_1), ("india_1.jpg", INDIA_1)):
        det.dets, det.offset = ann, (0, 0)
        img = cv2.imread(str(MEDIA / name))
        s = eng.session(name, image_mode=True)
        out, res = eng.process(img, s)
        eng.finish(s)
        totals["images"] += 1
        totals["bikes"] += len(res["vehicles"])
        totals["riders"] += sum(v["riders"] for v in res["vehicles"])
        totals["violations"] += len(res["new_violations"])
        cv2.imwrite(str(out_dir / name), out)
        print(name, [ (v["track_id"], v["riders"], v["violations"]) for v in res["vehicles"]])

    # --- synthetic video: india_1 panned across 24 frames (tests tracking, smoothing, plate fusion)
    base = cv2.imread(str(MEDIA / "india_1.jpg"))
    s = eng.session("india_1_pan.mp4")
    writer = open_writer(out_dir / "india_1_pan.mp4", 8, (base.shape[1], base.shape[0]))
    for k in range(24):
        dx = -2 * k
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        frame = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        det.dets, det.offset = INDIA_1, (dx, 0)
        out, res = eng.process(frame, s)
        writer.write(out)
        totals["violations"] += len(res["new_violations"])
    totals["violations"] += len(eng.finish(s))
    writer.release()
    print("video session counters:", s.counters())
    assert s.counters()["unique_motorcycles"] == 1, "tracker must keep ONE id for the panning bike"
    assert s.counters()["violations_recorded"] == 1, "one evidence record per violating track"

    # --- plate OCR on the real, readable plates of riders_1 (multi-crop fusion)
    from pipeline.plate_ocr import PlateReader
    from pipeline.tracker import Track
    img = cv2.imread(str(MEDIA / "riders_1.jpg"))
    for d in [d for d in RIDERS_1 if d.cls == "license_plate"]:
        t = Track(0, d, 0, 0)
        pr = PlateReader()
        for scale in (1.0, 0.9, 1.1):              # simulate slightly different frames
            fr = cv2.resize(img, None, fx=scale, fy=scale)
            b = tuple(int(v * scale) for v in d.box)
            pr.observe(t, fr, b)
        r = pr.read(t)
        totals["plates_read"].append(r["plate"] or "-")
        print("plate", d.box, "->", r)

    print("pipeline totals:", totals)
    ver = store.verify()
    print("chain verification (clean):", ver)
    assert ver["valid"], ver

    # tamper test: modify one evidence image -> chain must break exactly there
    from models import SessionLocal, Violation
    db = SessionLocal()
    rows = db.query(Violation).order_by(Violation.id).all()
    db.close()
    assert rows, "expected at least one violation"
    victim = rows[len(rows) // 2]
    f = config.EVIDENCE_DIR / victim.frame_path
    backup = f.read_bytes()
    im = cv2.imread(str(f)); im[0:10, 0:10] = 0; cv2.imwrite(str(f), im)
    ver2 = store.verify()
    print("chain verification (after tampering record %d):" % victim.id, ver2)
    assert not ver2["valid"] and ver2["broken_id"] == victim.id
    f.write_bytes(backup)
    assert store.verify()["valid"]

    # challan for the most severe record
    worst = max(rows, key=lambda r: r.severity)
    pdf = build_challan(store.row_to_dict(worst), config.EVIDENCE_DIR, config.PUBLIC_URL + "/verify")
    (out_dir / "sample_challan.pdf").write_bytes(pdf)
    print(f"challan: {len(pdf)} bytes for violation {worst.id} ({worst.violation_types}, severity {worst.severity})")
    print("ALL GT PIPELINE CHECKS PASSED  ->", out_dir)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
