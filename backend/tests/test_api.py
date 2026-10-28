"""
HTTP-level tests: the real FastAPI app with a stand-in detector (hand-annotated boxes, like
serve_demo.py), an isolated DATA_DIR, and FastAPI's TestClient. No model or server needed.

Run from backend/:   python tests/test_api.py      (or: python -m pytest tests/test_api.py)
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="hai_api_")
os.environ.setdefault("ADMIN_TOKEN", "")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from pipeline.engine import Engine  # noqa: E402
from tests.test_rules import INDIA_1  # noqa: E402

MEDIA = main.config.ROOT / "samples/media"


class Fixed:
    model_names = ["annotated-demo"]

    def __call__(self, bgr):
        return list(INDIA_1)


main.build_engine = lambda: Engine(Fixed(), main.SETTINGS, store=main.store, notify=main.notify_threadsafe)


def _client():
    return TestClient(main.app)


def test_health_and_model():
    with _client() as c:
        h = c.get("/").json()
        assert h["status"] == "ok" and h["detection"] and h["models"] == ["annotated-demo"]
        assert "plate_ocr_available" in c.get("/api/model").json()


def test_settings_validation():
    with _client() as c:
        before = c.get("/api/settings").json()
        assert c.put("/api/settings", json={"privacy_blur": "false"}).json()["privacy_blur"] is False
        assert c.put("/api/settings", json={"privacy_blur": True}).json()["privacy_blur"] is True
        for bad in ({"smoothing_window": "abc"}, {"smoothing_window": 0}, {"mode": "space"},
                    {"enhancement": "sepia"}, {"nope": 1}, {"camera_id": ""}, {"plate_ocr": "maybe"}):
            r = c.put("/api/settings", json=bad)
            assert r.status_code == 400, (bad, r.status_code)
        # a rejected patch must not half-apply
        assert c.put("/api/settings", json={"camera_id": "X", "mode": "bad"}).status_code == 400
        assert c.get("/api/settings").json()["camera_id"] == before["camera_id"]


def test_image_detect_counts_each_rider_once():
    with _client() as c:
        before = c.get("/api/stats").json()
        img = (MEDIA / "india_1.jpg").read_bytes()
        r = c.post("/api/detect/image.json", files={"file": ("india_1.jpg", img, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["vehicles"][0]["riders"] == 2 and body["vehicles"][0]["violations"] == ["NO_HELMET"]
        v = body["violations"][0]
        assert v["id"] and v["track_id"] == body["vehicles"][0]["track_id"]
        after = c.get("/api/stats").json()
        # 2 riders (1 helmet + 1 bare head) -> exactly 2 new detection rows (was 4: double logging)
        assert after["total_detections"] - before["total_detections"] == 2, (before, after)
        assert after["with_helmet"] - before["with_helmet"] == 1
        assert after["violations_total"] - before["violations_total"] == 1


def test_annotated_jpeg_endpoint_and_challan_and_chain():
    with _client() as c:
        img = (MEDIA / "india_1.jpg").read_bytes()
        r = c.post("/api/detect/image", files={"file": ("x.jpg", img, "image/jpeg")})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is not None
        rows = c.get("/api/violations").json()
        assert rows
        pdf = c.get(f"/api/violations/{rows[0]['id']}/challan.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        assert c.get("/api/evidence/verify").json()["valid"]
        assert c.get("/api/evidence/verify?full=false").json()["valid"]
        rv = c.post(f"/api/violations/{rows[0]['id']}/review", json={"action": "confirmed"}).json()
        assert rv["status"] == "confirmed"
        assert c.get("/api/evidence/verify").json()["valid"], "review must not break the chain"
        assert c.get(f"/api/evidence/file/{rows[0]['frame_path']}").status_code == 200
        assert c.get("/api/evidence/file/../../config.py").status_code == 404


def test_bad_inputs():
    with _client() as c:
        assert c.post("/api/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
        assert c.post("/api/detect/video", files={"file": ("x.txt", b"x", "text/plain")}).status_code == 400
        assert c.get("/api/stream/video", params={"source": "../../backend/config.py"}).status_code == 400
        assert c.get("/api/violations/999999").status_code == 404
        assert c.post("/api/violations/1/review", json={"action": "delete"}).status_code == 400


def test_enhance_preview():
    with _client() as c:
        img = (MEDIA / "riders_1.jpg").read_bytes()
        r = c.post("/api/enhance/preview?mode=fog", files={"file": ("a.jpg", img, "image/jpeg")})
        assert r.status_code == 200 and '"fog"' in r.headers["x-result"]
        assert c.post("/api/enhance/preview?mode=sepia", files={"file": ("a.jpg", img, "image/jpeg")}).status_code == 400


def test_video_upload_produces_h264_mp4():
    # 12-frame clip with an odd frame size (H.264 needs even sizes - the writer must cope)
    src = Path(os.environ["DATA_DIR"]) / "clip.avi"
    base = cv2.imread(str(MEDIA / "india_1.jpg"))
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (base.shape[1], base.shape[0]))
    for k in range(12):
        w.write(np.roll(base, -2 * k, axis=1))
    w.release()
    with _client() as c:
        with c.stream("POST", "/api/detect/video?run_id=t1", files={"file": ("clip.avi", src.read_bytes(), "video/avi")}) as r:
            assert r.status_code == 200
            body = b"".join(r.iter_bytes())
        assert body.count(b"--frame") == 12
        outs = list(main.config.OUTPUT_DIR.glob("clip_*.mp4"))
        assert outs, "annotated MP4 not written"
        cap = cv2.VideoCapture(str(outs[0]))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 12
        served = c.get(f"/api/outputs/{outs[0].name}")
        assert served.status_code == 200 and served.headers["content-type"] == "video/mp4"
        assert not list(main.config.UPLOAD_DIR.iterdir()), "uploaded file must be deleted after processing"


def test_stream_mode_parameter():
    with _client() as c:
        assert c.get("/api/stream/video", params={"source": "123.mp4", "mode": "space"}).status_code == 422
    # helmet-check ("site") mode: every head is checked even with no motorcycle in view
    from tests.test_rules import Det
    eng = main.build_engine()
    eng.det = lambda bgr: [Det("head", .9, (300, 100, 380, 190))]
    s = eng.session("cam", mode="site")
    img = cv2.imread(str(MEDIA / "riders_1.jpg"))
    recs = []
    for _ in range(14):
        _, res = eng.process(img, s)
        recs += res["new_violations"]
    recs += eng.finish(s)                       # evidence is written by a background worker
    assert s.mode == "site" and res["vehicles"][0]["no_helmet"] == 1
    assert [r["violation_list"] for r in recs] == [["NO_HEADGEAR"]], recs


def test_image_auto_mode_checks_people_without_a_bike():
    from tests.test_rules import Det
    with _client() as c:
        img = (MEDIA / "riders_1.jpg").read_bytes()
        main.engine.det = lambda bgr: [Det("head", .9, (300, 100, 380, 190)), Det("helmet", .9, (100, 100, 170, 180))]
        r = c.post("/api/detect/image.json?mode=auto", files={"file": ("a.jpg", img, "image/jpeg")}).json()
        assert r["mode"] == "site" and sum(v["helmets"] for v in r["vehicles"]) == 1
        assert [v["violation_list"] for v in r["violations"]] == [["NO_HEADGEAR"]]
        r = c.post("/api/detect/image.json?mode=traffic", files={"file": ("a.jpg", img, "image/jpeg")}).json()
        assert r["mode"] == "traffic" and r["vehicles"] == []          # no bike -> nothing to judge
        main.engine.det = Fixed()                                       # INDIA_1 has a motorcycle
        r = c.post("/api/detect/image.json?mode=auto", files={"file": ("a.jpg", img, "image/jpeg")}).json()
        assert r["mode"] == "traffic" and r["vehicles"][0]["riders"] == 2


def test_video_stride_processes_fewer_frames():
    src = Path(os.environ["DATA_DIR"]) / "clip2.avi"
    base = cv2.imread(str(MEDIA / "india_1.jpg"))
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (base.shape[1], base.shape[0]))
    for k in range(12):
        w.write(np.roll(base, -2 * k, axis=1))
    w.release()
    with _client() as c:
        with c.stream("POST", "/api/detect/video?stride=3", files={"file": ("clip2.avi", src.read_bytes(), "video/avi")}) as r:
            body = b"".join(r.iter_bytes())
        assert body.count(b"--frame") == 4
        assert c.post("/api/detect/video?stride=9", files={"file": ("c.avi", b"x", "video/avi")}).status_code == 422


class FakeCamera:
    """Stands in for cv2.VideoCapture(0): 30 fps, never ends, records whether it was released."""
    instances = []

    def __init__(self, *_):
        self.released = False
        FakeCamera.instances.append(self)

    def isOpened(self):
        return True

    def get(self, _):
        return 30.0

    def read(self):
        import time
        time.sleep(1 / 30)
        return (not self.released), cv2.imread(str(MEDIA / "riders_1.jpg"))

    def release(self):
        self.released = True


def _with_fake_camera(fn):
    real = main.cv2.VideoCapture
    main.cv2.VideoCapture = FakeCamera
    FakeCamera.instances.clear()
    try:
        with _client():                       # starts the engine
            fn()
    finally:
        main.cv2.VideoCapture = real


def test_webcam_released_on_disconnect_stop_and_early_exit():
    def check():
        # 1) client disconnects mid-stream -> the generator is closed -> camera released
        frames, abort = main.frame_generator(0, "0", False, "video_score", "cam-a", "site")
        next(frames); next(frames)
        frames.close(); abort()
        assert FakeCamera.instances[-1].released, "camera still on after disconnect"
        # 2) Stop endpoint while the client stays connected
        frames, abort = main.frame_generator(0, "0", False, "video_score", "cam-b", "site")
        next(frames)
        assert main.stop_stream("cam-b")["stopped"] == ["cam-b"]
        assert len(list(frames)) <= 1          # the loop ends at its next stop check, not later
        assert FakeCamera.instances[-1].released, "camera still on after Stop"
        # 3) client gone before the first frame (a never-started generator skips its finally)
        frames, abort = main.frame_generator(0, "0", False, "video_score", "cam-c", "site")
        frames.close(); abort()
        assert FakeCamera.instances[-1].released, "camera still on after early disconnect"
        assert not main.ACTIVE_RUNS, main.ACTIVE_RUNS
    _with_fake_camera(check)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
