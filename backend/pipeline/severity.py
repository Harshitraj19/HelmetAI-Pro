"""
Risk / severity score 0-100 for prioritising violations.

  base points per violation type  +  per extra unprotected rider  +  apparent speed bonus
  + repeat-offender bonus (same plate seen violating before)
"""
WEIGHTS = {"NO_HELMET": 40, "INVALID_HELMET": 30, "HELMET_NOT_WORN": 30, "TRIPLE_RIDING": 30}


def score(violations: list[str], unprotected: int, speed: float, repeat_count: int = 0) -> int:
    s = sum(WEIGHTS.get(v, 0) for v in violations)
    s += 10 * max(0, unprotected - 1)           # every extra unprotected person
    s += min(20, int(speed * 100))              # speed = bike-heights per frame
    s += min(15, 5 * repeat_count)
    return int(max(0, min(100, s)))


def level(s: int) -> str:
    return "critical" if s >= 75 else "high" if s >= 50 else "medium" if s >= 25 else "low"
