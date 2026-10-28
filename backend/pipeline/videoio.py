"""
MP4 writing that browsers can actually play.

OpenCV's usual 'mp4v' (MPEG-4 Part 2) downloads fine but will not play in a <video> tag.
H.264 ('avc1') does; it is available through Media Foundation on Windows and through
FFmpeg/OpenH264 on most Linux builds. We try H.264 first and fall back to 'mp4v'.

H.264 needs even frame sizes (Media Foundation even hangs on release() with odd ones), so the
writer is opened at the nearest even size and frames are cropped by at most one pixel.
"""
from __future__ import annotations

import logging
import sys
import threading

import cv2

log = logging.getLogger("videoio")
# (backend, fourcc) in order of preference. On Windows, Media Foundation has a built-in H.264
# encoder (FFmpeg would first try - and loudly fail - to load the OpenH264 DLL).
_CANDIDATES = ([(cv2.CAP_MSMF, "avc1")] if sys.platform == "win32" else []) + \
              [(cv2.CAP_ANY, "avc1"), (cv2.CAP_ANY, "mp4v")]
_working: tuple[int, str] | None = None      # first combination that opened, remembered


def safe_fps(fps: float, default: float = 25.0) -> float:
    """Webcams / some streams report 0, NaN or 1000 fps."""
    return fps if fps and 1 <= fps <= 120 else default


class Mp4Writer:
    def __init__(self, writer: cv2.VideoWriter, size: tuple[int, int]):
        self._w, (self.width, self.height) = writer, size

    def write(self, frame):
        self._w.write(frame[:self.height, :self.width])

    def release(self):
        self._w.release()


def open_writer(path, fps: float, size: tuple[int, int]) -> Mp4Writer:
    global _working
    even = (max(2, size[0] & ~1), max(2, size[1] & ~1))
    for api, cc in ([_working] if _working else []) + _CANDIDATES:
        w = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*cc), safe_fps(fps), even)
        if w.isOpened():
            if _working != (api, cc):
                log.info("video writer: %s via backend %s", cc, api)
                _working = (api, cc)
            return Mp4Writer(w, even)
        w.release()
    raise RuntimeError(f"no usable MP4 codec for {path}")


class LatestFrameReader:
    """For LIVE sources (webcam, RTSP): a thread keeps reading and only the newest frame is handed
    out. When processing (~8 fps on CPU) is slower than the camera (25-30 fps), stale frames are
    dropped instead of queueing up, so the picture stays real-time instead of drifting seconds
    behind. The thread is the only user of the capture and releases it itself (no cross-thread
    release while a read is in progress)."""

    def __init__(self, cap):
        self._cap = cap
        self._cond = threading.Condition()
        self._frame, self._seq, self._taken, self._alive, self._stop = None, 0, 0, True, False
        self.span = 1          # source frames covered by the last read (1 + frames dropped before it)
        self._thread = threading.Thread(target=self._run, name="frame-grabber", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while not self._stop:
                ok, frame = self._cap.read()
                with self._cond:
                    if not ok:
                        break
                    self._frame, self._seq = frame, self._seq + 1
                    self._cond.notify_all()
        finally:
            self._cap.release()
            with self._cond:
                self._alive = False
                self._cond.notify_all()

    def read(self, timeout: float = 10.0):
        """(ok, newest frame not returned before). ok=False when the source ended or stalled."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._taken or not self._alive, timeout)
            if self._seq == self._taken:
                return False, None
            self.span, self._taken = self._seq - self._taken, self._seq
            return True, self._frame

    def release(self):
        self._stop = True
        self._thread.join(timeout=3)
