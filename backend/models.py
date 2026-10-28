"""
Database models (SQLAlchemy). Extends the DIP_PROJECT base schema:

  detections   - every detection (kept from the base, + bbox, track id, camera, source)
  violations   - one row per violating MOTORCYCLE TRACK, with evidence files, plate,
                 severity and the SHA-256 hash chain fields
"""
import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class DetectionRecord(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    object_class = Column(String, index=True)       # helmet | head | motorcycle | license_plate | hardhat
    confidence = Column(Float)
    image_path = Column(String, nullable=True)      # kept for base compatibility (= source name)
    source = Column(String, nullable=True)
    camera_id = Column(String, nullable=True)
    track_id = Column(Integer, nullable=True)
    bbox = Column(String, nullable=True)            # "x1,y1,x2,y2"


class Violation(Base):
    __tablename__ = "violations"
    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=utcnow, index=True)
    source = Column(String)
    camera_id = Column(String, index=True)
    track_id = Column(Integer)
    violation_types = Column(String, index=True)    # comma list: NO_HELMET,TRIPLE_RIDING,...
    riders = Column(Integer)
    helmets = Column(Integer)
    plate = Column(String, index=True, nullable=True)
    plate_confidence = Column(Float, nullable=True)
    plate_reads = Column(Text, nullable=True)       # JSON list of raw per-frame reads
    severity = Column(Integer)
    severity_level = Column(String)
    scene = Column(String, nullable=True)           # enhancement scene detected
    frame_path = Column(String, nullable=True)
    bike_path = Column(String, nullable=True)
    plate_path = Column(String, nullable=True)
    record_hash = Column(String(64))
    prev_hash = Column(String(64))
    chain_hash = Column(String(64), index=True)
    status = Column(String, default="pending")      # pending | confirmed | rejected (human review)
    reviewed_at = Column(DateTime, nullable=True)


class AuditLog(Base):
    """Human-review actions are logged separately so the evidence chain itself stays append-only."""
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    at = Column(DateTime, default=utcnow)
    violation_id = Column(Integer, ForeignKey("violations.id"))
    action = Column(String)
    note = Column(Text, nullable=True)


_SQLITE = DATABASE_URL.startswith("sqlite")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 15} if _SQLITE else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

if _SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(conn, _):
        # WAL: dashboard/history reads no longer block (or fail) while a stream is writing
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


def init_db():
    Base.metadata.create_all(bind=engine)
