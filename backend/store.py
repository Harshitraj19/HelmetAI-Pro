"""
Persistence service: detections + hash-chained violations.

A process-wide lock serialises violation inserts so the chain stays linear even when
several streams/uploads are processed concurrently.
"""
from __future__ import annotations

import json
import threading
import uuid

from sqlalchemy import func

from config import EVIDENCE_DIR
from models import AuditLog, DetectionRecord, SessionLocal, Violation, utcnow
from pipeline import evidence

_chain_lock = threading.Lock()


def row_to_dict(v: Violation) -> dict:
    d = {c.name: getattr(v, c.name) for c in Violation.__table__.columns}
    d["created_at"] = v.created_at.isoformat() if v.created_at else None
    d["reviewed_at"] = v.reviewed_at.isoformat() if v.reviewed_at else None
    d["violation_list"] = [x for x in (v.violation_types or "").split(",") if x]
    d["plate_reads"] = json.loads(v.plate_reads) if v.plate_reads else []
    return d


def add_violation(fields: dict, frame, bike_crop, plate_crop) -> dict:
    """fields: source, camera_id, track_id, violation_types(list), riders, helmets, plate,
    plate_confidence, plate_reads(list), severity, severity_level, scene"""
    uid = uuid.uuid4().hex[:16]
    created = utcnow()
    files = evidence.save_images(EVIDENCE_DIR, uid, created.strftime("%Y%m%d"), frame, bike_crop, plate_crop)
    rec = dict(fields, uid=uid, created_at=created.isoformat(), **files)
    rec["violation_types"] = ",".join(fields["violation_types"])
    rec["plate_reads"] = json.dumps(fields.get("plate_reads", []))
    with _chain_lock:
        db = SessionLocal()
        try:
            last = db.query(Violation).order_by(Violation.id.desc()).first()
            prev = last.chain_hash if last else evidence.GENESIS
            rh = evidence.record_hash(rec, EVIDENCE_DIR)
            v = Violation(**{k: rec[k] for k in rec if k in Violation.__table__.columns and k != "created_at"},
                          created_at=created, record_hash=rh, prev_hash=prev, chain_hash=evidence.chain(prev, rh))
            db.add(v)
            db.commit()
            db.refresh(v)
            return row_to_dict(v)
        finally:
            db.close()


def log_detections(rows: list[dict]):
    if not rows:
        return
    db = SessionLocal()
    try:
        db.add_all(DetectionRecord(timestamp=utcnow(), **r) for r in rows)
        db.commit()
    finally:
        db.close()


def repeat_count(plate: str) -> int:
    if not plate:
        return 0
    db = SessionLocal()
    try:
        return db.query(func.count(Violation.id)).filter(Violation.plate == plate).scalar() or 0
    finally:
        db.close()


def verify(full: bool = True) -> dict:
    db = SessionLocal()
    try:
        rows = [row_to_dict(v) for v in db.query(Violation).order_by(Violation.id).all()]
    finally:
        db.close()
    for r in rows:   # hash uses the stored string form
        r["violation_types"] = ",".join(r["violation_list"])
        r["plate_reads"] = None
    return evidence.verify_chain(rows, EVIDENCE_DIR, full=full)


def review(vid: int, action: str, note: str | None):
    db = SessionLocal()
    try:
        v = db.get(Violation, vid)
        if not v:
            return None
        # status lives outside the hashed fields, so review never breaks the chain
        v.status, v.reviewed_at = action, utcnow()
        db.add(AuditLog(violation_id=vid, action=action, note=note))
        db.commit()
        return row_to_dict(v)
    finally:
        db.close()
