"""
Unit tests for the rider-association rule engine.

Real cases use hand-annotated boxes on the sample photos shipped with the base repo
(samples/media); synthetic cases cover rules that need specific geometry.
Run: python -m pytest tests/test_rules.py   (or simply: python tests/test_rules.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.detector import Det  # noqa: E402
from pipeline.riders import associate  # noqa: E402

# ---- hand annotations (original pixel coordinates) --------------------------------------
RIDERS_1 = [  # samples/media/riders_1.jpg : 3 bikes, every rider helmeted, pedestrians behind
    Det("motorcycle", .9, (72, 101, 242, 397)), Det("helmet", .9, (134, 27, 191, 94)),
    Det("license_plate", .8, (117, 208, 186, 222)),
    Det("head", .8, (87, 36, 117, 72)),            # yellow-shirt pedestrian standing behind bike A
    Det("head", .7, (54, 50, 72, 72)),             # far-away pedestrian
    Det("motorcycle", .9, (245, 115, 440, 419)), Det("helmet", .9, (318, 54, 381, 136)),
    Det("license_plate", .8, (305, 240, 377, 253)),
    Det("motorcycle", .9, (455, 144, 625, 401)), Det("helmet", .9, (491, 27, 547, 87)),
    Det("license_plate", .8, (495, 180, 574, 194)),
]
INDIA_1 = [  # samples/media/india_1.jpg : helmeted rider + bare-headed child pillion, police officer beside
    Det("motorcycle", .9, (98, 217, 727, 549)),
    Det("helmet", .9, (421, 67, 586, 239)),
    Det("head", .9, (163, 114, 258, 222)),         # pillion child - no helmet -> violation
    Det("head", .9, (776, 0, 976, 184)),           # traffic police officer standing beside -> not a rider
]


def test_riders_1_all_compliant_and_pedestrians_ignored():
    scenes, loose = associate(RIDERS_1)
    assert len(scenes) == 3
    assert all(s.violations == [] for s in scenes), [s.summary() for s in scenes]
    assert all(s.plate is not None for s in scenes)
    assert sum(s.n_riders for s in scenes) == 3
    assert len(loose) == 2                          # both bare heads are pedestrians


def test_india_1_pillion_without_helmet():
    scenes, loose = associate(INDIA_1)
    s = scenes[0].summary()
    assert s["riders"] == 2 and s["helmets"] == 1 and s["no_helmet"] == 1
    assert scenes[0].violations == ["NO_HELMET"]
    assert len(loose) == 1                          # police officer not counted as a rider


def test_triple_riding():
    bike = Det("motorcycle", .9, (100, 300, 300, 500))
    heads = [Det("head", .9, (150, 180, 190, 225)), Det("helmet", .9, (195, 182, 237, 228)),
             Det("head", .9, (240, 186, 280, 231))]
    s = associate([bike] + heads)[0][0]
    assert s.n_riders == 3
    assert set(s.violations) == {"NO_HELMET", "TRIPLE_RIDING"}


def test_helmet_carried_not_worn():
    bike = Det("motorcycle", .9, (100, 300, 300, 500))
    head = Det("head", .9, (180, 180, 222, 228))
    hanging = Det("helmet", .8, (110, 340, 150, 382))   # on the handlebar
    s = associate([bike, head, hanging])[0][0]
    assert len(s.carried_helmets) == 1 and s.n_riders == 1
    assert set(s.violations) == {"NO_HELMET", "HELMET_NOT_WORN"}


def test_hardhat_is_not_a_riding_helmet():
    bike = Det("motorcycle", .9, (100, 300, 300, 500))
    s = associate([bike, Det("hardhat", .9, (180, 180, 225, 222))])[0][0]
    assert s.violations == ["INVALID_HELMET"]


def test_head_and_helmet_on_same_person_counted_once():
    bike = Det("motorcycle", .9, (100, 300, 300, 500))
    s = associate([bike, Det("helmet", .9, (180, 180, 225, 225)), Det("head", .5, (182, 183, 224, 226))])[0][0]
    assert s.n_riders == 1 and s.violations == []


def test_two_bikes_riders_not_mixed():
    b1, b2 = Det("motorcycle", .9, (0, 300, 200, 500)), Det("motorcycle", .9, (400, 300, 600, 500))
    s = associate([b1, b2, Det("helmet", .9, (80, 190, 125, 235)), Det("head", .9, (480, 190, 525, 235))])[0]
    assert s[0].violations == [] and s[1].violations == ["NO_HELMET"]


def test_people_standing_behind_parked_bike_are_not_riders():
    bike = Det("motorcycle", .9, (300, 500, 600, 900))
    rider_head = Det("head", .9, (420, 300, 480, 370))
    rider = Det("person", .9, (380, 290, 560, 820))           # legs reach the footrest
    bystander_head = Det("head", .9, (330, 330, 385, 395))
    bystander = Det("person", .9, (300, 320, 420, 560))        # only upper body visible above the bike
    s, loose = associate([bike, rider_head, rider, bystander_head, bystander])
    assert s[0].n_riders == 1 and bystander_head in loose


def test_person_standing_beside_bike_not_counted():
    bike = Det("motorcycle", .9, (300, 500, 600, 900))
    heads = [Det("helmet", .9, (420, 300, 480, 370)), Det("helmet", .9, (610, 280, 670, 350))]
    people = [Det("person", .9, (380, 290, 560, 820)), Det("person", .9, (600, 270, 760, 950))]  # officer beside
    s = associate([bike] + heads + people)[0]
    assert s[0].n_riders == 1 and s[0].violations == []


def test_head_inside_two_person_boxes_kept_if_any_is_rider():
    bike = Det("motorcycle", .9, (436, 134, 626, 401))        # riders_1 right bike, real detections
    helmet = Det("helmet", .86, (490, 25, 551, 93))
    rider = Det("person", .80, (429, 24, 595, 390))
    behind = Det("person", .81, (448, 0, 523, 154))           # higher confidence, but standing behind
    s = associate([bike, helmet, rider, behind])[0]
    assert s[0].n_riders == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
