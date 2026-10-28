"""
Rider association & rule engine  (Novelties #1 worn-vs-carried, #2 rider counting / triple riding,
                                  plus fake-helmet (hard-hat) check)

Idea from KrishArul26 (detect the motorcycle first, then the riders on it) +
geometry gating from the PPE repo (PPE must sit in the right body region),
re-designed for two-wheelers:

    rider zone of a bike =  the bike box widened by 15 % and extended upward by 1.6 x its height
    a head/helmet/hardhat belongs to the bike whose zone contains its centre with the best score
    plate belongs to the bike whose (slightly enlarged) box contains it

Per head-level object we decide the rider status:
    helmet  in the upper part of the zone          -> HELMET (compliant)
    head    (bare)                                  -> NO_HELMET
    hardhat on a rider                              -> INVALID_HELMET   (construction hard-hat on a bike)
    helmet  sitting LOW (handlebar / seat / hand)   -> CARRIED          (helmet present but not worn)
            and a bare head is also on the bike         -> that rider is NO_HELMET + helmet "carried"

Bike-level rules:
    riders > 2                                      -> TRIPLE_RIDING
    any NO_HELMET / INVALID_HELMET                  -> HELMET violation
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Det

HEADWEAR = ("helmet", "head", "hardhat")


@dataclass
class Rider:
    status: str                  # HELMET | NO_HELMET | INVALID_HELMET
    det: Det
    position: str = "rider"      # rider | pillion


@dataclass
class BikeScene:
    bike: Det
    riders: list[Rider] = field(default_factory=list)
    carried_helmets: list[Det] = field(default_factory=list)
    plate: Det | None = None

    @property
    def n_riders(self):
        return len(self.riders)

    @property
    def violations(self) -> list[str]:
        v = []
        if any(r.status == "NO_HELMET" for r in self.riders):
            v.append("NO_HELMET")
        if any(r.status == "INVALID_HELMET" for r in self.riders):
            v.append("INVALID_HELMET")
        if self.carried_helmets and any(r.status != "HELMET" for r in self.riders):
            v.append("HELMET_NOT_WORN")
        if self.n_riders > 2:
            v.append("TRIPLE_RIDING")
        return v

    def summary(self) -> dict:
        return {
            "riders": self.n_riders,
            "helmets": sum(r.status == "HELMET" for r in self.riders),
            "no_helmet": sum(r.status == "NO_HELMET" for r in self.riders),
            "invalid_helmet": sum(r.status == "INVALID_HELMET" for r in self.riders),
            "carried": len(self.carried_helmets),
            "violations": self.violations,
        }


def rider_zone(b: Det) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = b.box
    w, h = x2 - x1, y2 - y1
    return x1 - 0.15 * w, y1 - 1.6 * h, x2 + 0.15 * w, y2


def _zone_score(d: Det, bike: Det) -> float:
    zx1, zy1, zx2, zy2 = rider_zone(bike)
    if not (zx1 <= d.cx <= zx2 and zy1 <= d.cy <= zy2):
        return 0.0
    # prefer heads horizontally centred on the bike and above it
    hx = 1 - abs(d.cx - bike.cx) / max(1, (zx2 - zx1) / 2)
    # head size should be plausible relative to bike (0.08 .. 0.7 of bike height)
    rel = d.h / max(1, bike.h)
    size_ok = 1.0 if 0.06 <= rel <= 0.8 else 0.3
    return max(0.05, hx) * size_ok


def associate(dets: list[Det], bikes: list[Det] | None = None) -> tuple[list[BikeScene], list[Det]]:
    """Returns (bike scenes, unassigned head-level detections such as pedestrians)."""
    bikes = bikes if bikes is not None else [d for d in dets if d.cls == "motorcycle"]
    scenes = [BikeScene(b) for b in bikes]
    heads = [d for d in dets if d.cls in HEADWEAR]
    plates = [d for d in dets if d.cls == "license_plate"]
    loose: list[Det] = []

    # each head-level object -> best bike
    assigned: dict[int, list[Det]] = {i: [] for i in range(len(scenes))}
    for h in heads:
        scores = [(_zone_score(h, s.bike), i) for i, s in enumerate(scenes)]
        best = max(scores, default=(0.0, -1))
        if best[0] > 0:
            assigned[best[1]].append(h)
        else:
            loose.append(h)

    persons = [d for d in dets if d.cls == "person"]
    for i, s in enumerate(scenes):
        objs = _dedupe_heads(assigned[i])
        objs, standing = _person_gate(objs, persons, s.bike)
        loose.extend(standing)
        objs, far = _depth_filter(objs)
        loose.extend(far)
        bx1, by1, bx2, by2 = s.bike.box
        low_line = by1 + 0.2 * (by2 - by1)         # below this a helmet is on the handlebar / seat
        bare = [o for o in objs if o.cls == "head"]
        for o in objs:
            if o.cls == "helmet" and o.cy > low_line and (bare or len(objs) == 1 and o.cy > by1 + 0.5 * (by2 - by1)):
                s.carried_helmets.append(o)        # present but NOT on a head
            elif o.cls == "helmet":
                s.riders.append(Rider("HELMET", o))
            elif o.cls == "head":
                s.riders.append(Rider("NO_HELMET", o))
            elif o.cls == "hardhat":
                s.riders.append(Rider("INVALID_HELMET", o))
        # order front -> back: the rider closest to the handlebar side is the leftmost/rightmost
        # head; we simply mark the first by x as "rider", others "pillion"
        s.riders.sort(key=lambda r: r.det.cx)
        for k, r in enumerate(s.riders):
            r.position = "rider" if k == 0 else "pillion"

    # plates -> bike that contains them
    for p in plates:
        best, bi = 0.0, -1
        for i, s in enumerate(scenes):
            x1, y1, x2, y2 = s.bike.box
            pad_w, pad_h = 0.15 * (x2 - x1), 0.25 * (y2 - y1)
            if x1 - pad_w <= p.cx <= x2 + pad_w and y1 <= p.cy <= y2 + pad_h:
                sc = p.conf * (1 - abs(p.cx - s.bike.cx) / max(1, (x2 - x1)))
                if sc > best:
                    best, bi = sc, i
        if bi >= 0 and (scenes[bi].plate is None or p.conf > scenes[bi].plate.conf):
            scenes[bi].plate = p
    return scenes, loose


def _person_gate(objs: list[Det], persons: list[Det], bike: Det) -> tuple[list[Det], list[Det]]:
    """Person-based association (PPE-repo idea, adapted to two-wheelers).
    A rider's body box reaches down into the lower half of the motorcycle (legs to the footrest);
    someone standing BEHIND a parked bike only shows the upper body above it, and someone standing
    BESIDE it has a body centred outside the bike. Heads without any person box are kept (detector
    may miss the person), so this only removes heads we have positive evidence against."""
    if not persons:
        return objs, []
    bx1, by1, bx2, by2 = bike.box
    keep, drop = [], []
    for o in objs:
        owners = [p for p in persons
                  if p.box[0] - 4 <= o.cx <= p.box[2] + 4 and p.box[1] - 0.3 * o.h <= o.cy <= p.box[1] + 0.45 * p.h]
        if not owners:
            keep.append(o)
            continue
        # overlapping people (pillion, person behind) -> the head is a rider's if ANY owner qualifies
        is_rider = any(p.box[3] >= by1 + 0.45 * (by2 - by1) and
                       bx1 - 0.05 * (bx2 - bx1) <= p.cx <= bx2 + 0.05 * (bx2 - bx1) for p in owners)
        (keep if is_rider else drop).append(o)
    return keep, drop


def _depth_filter(objs: list[Det]) -> tuple[list[Det], list[Det]]:
    """People standing BEHIND a bike fall inside its zone in 2-D. Real riders' heads are at
    the same depth, so a head much smaller than the largest rider head is background."""
    if len(objs) < 2:
        return objs, []
    ref = max(o.h for o in objs)
    near = [o for o in objs if o.h >= 0.6 * ref]
    return near, [o for o in objs if o.h < 0.6 * ref]


def _dedupe_heads(objs: list[Det]) -> list[Det]:
    """A head and a helmet box on the same person -> keep the stronger one."""
    from .detector import iou
    out: list[Det] = []
    for o in sorted(objs, key=lambda d: d.conf, reverse=True):
        if all(iou(o.box, k.box) < 0.45 for k in out):
            out.append(o)
    return out


def smooth(votes) -> dict:
    """Temporal majority vote over the last N frame summaries of a tracked bike (PPE idea)."""
    if not votes:
        return {}
    n = len(votes)
    need = (n + 1) // 2
    keys = ("NO_HELMET", "INVALID_HELMET", "HELMET_NOT_WORN", "TRIPLE_RIDING")
    stable = [k for k in keys if sum(k in v["violations"] for v in votes) >= need]
    riders = sorted(v["riders"] for v in votes)[n // 2]
    return {"violations": stable, "riders": riders,
            "helmets": sorted(v["helmets"] for v in votes)[n // 2],
            "no_helmet": sorted(v["no_helmet"] for v in votes)[n // 2]}
