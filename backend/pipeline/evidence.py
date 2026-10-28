"""
Tamper-evident evidence store  (Novelty #5)

Every violation produces an evidence bundle:
    frame.jpg   full frame (privacy-filtered: bystanders blurred, violator kept clear)
    bike.jpg    crop of the offending motorcycle + riders
    plate.jpg   sharpest plate crop across the track (if any)

record_hash = SHA-256( canonical JSON of the record fields  +  SHA-256 of every evidence file )
chain_hash  = SHA-256( previous chain_hash  +  record_hash )          (genesis = 64 zeros)

Changing ANY byte of an image, the plate text, the time or the violation type - or deleting /
re-ordering a record - breaks every later chain_hash. `verify_chain()` pinpoints the first
broken record. This is a blockchain-style append-only log without needing a blockchain.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2

GENESIS = "0" * 64
HASHED_FIELDS = ("uid", "created_at", "source", "camera_id", "track_id", "violation_types",
                 "riders", "plate", "severity", "frame_path", "bike_path", "plate_path")


# (path, mtime_ns, size) -> digest. Lets the dashboard poll the chain cheaply; any rewrite of a file
# changes its mtime/size and forces a re-hash. verify_chain(full=True) bypasses the cache entirely.
_hash_cache: dict[tuple, str] = {}


def sha256_file(path: str | Path, use_cache: bool = False) -> str:
    key = None
    if use_cache:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key in _hash_cache:
            return _hash_cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if key:
        if len(_hash_cache) > 50_000:
            _hash_cache.clear()
        _hash_cache[key] = digest
    return digest


def record_hash(fields: dict, root: Path, use_cache: bool = False) -> str:
    payload = {k: fields.get(k) for k in HASHED_FIELDS}
    files = [fields.get(k) for k in ("frame_path", "bike_path", "plate_path") if fields.get(k)]
    payload["file_hashes"] = [sha256_file(root / f, use_cache) if (root / f).exists() else "MISSING" for f in files]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def chain(prev: str, rec: str) -> str:
    return hashlib.sha256((prev + rec).encode()).hexdigest()


def save_images(root: Path, uid: str, day: str, frame, bike_crop, plate_crop) -> dict:
    folder = Path(day)
    (root / folder).mkdir(parents=True, exist_ok=True)
    out = {}
    for key, img in (("frame_path", frame), ("bike_path", bike_crop), ("plate_path", plate_crop)):
        if img is None or getattr(img, "size", 0) == 0:
            out[key] = None
            continue
        rel = folder / f"{uid}_{key.split('_')[0]}.jpg"
        cv2.imwrite(str(root / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out[key] = rel.as_posix()
    return out


def verify_chain(records: list[dict], root: Path, full: bool = True) -> dict:
    """records must be ordered by id. Returns a report usable by the API/UI.
    full=False reuses cached image hashes for unchanged files (fast dashboard polling)."""
    prev = GENESIS
    for i, r in enumerate(records):
        rh = record_hash(r, root, use_cache=not full)
        if rh != r["record_hash"]:
            return {"valid": False, "checked": i, "broken_id": r["id"],
                    "reason": "record content or evidence image was modified"}
        if r["prev_hash"] != prev or chain(prev, rh) != r["chain_hash"]:
            return {"valid": False, "checked": i, "broken_id": r["id"],
                    "reason": "chain link broken (record deleted, inserted or re-ordered)"}
        prev = r["chain_hash"]
    return {"valid": True, "checked": len(records), "head": prev}
