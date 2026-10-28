"""
pseudo_label.py — cross-source label completion (teacher -> student distillation)

WHY: no source dataset is fully labelled for our 5 classes.
  * Open Images (Siddhesh) was downloaded one class at a time with OIDv4: an image in the
    "Motorcycle" set has ONLY motorcycle boxes even if the rider wears a helmet
    (measured: 0 images with >1 class in every split).
  * PPE only labels hard-hats; bare heads of workers are unlabelled.
Training on that directly teaches the model that unlabelled helmets/heads/bikes are background.

HOW: for every train/val image, teachers predict the classes that image's source did NOT
label, and confident predictions are added (never overlapping existing boxes of the same
class). TEST labels are never modified.

Teachers (round 1):
  models/teacher_helmet_dip.pt   With Helmet -> helmet, Without Helmet -> head
  yolo11n.pt (COCO)              motorcycle
Round 2: the round-1 student itself (adds license_plate + refines everything).

Usage:
  python training/pseudo_label.py --round 1
  python training/pseudo_label.py --round 2 --student runs/unified_r1/weights/best.pt
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ["helmet", "head", "motorcycle", "license_plate", "hardhat"]


def canon(name: str):
    n = name.lower().replace("_", " ").replace("-", " ")
    if n in ("without helmet", "no helmet", "head", "human head"):
        return 1
    if n in ("hardhat", "hard hat", "safety helmet"):
        return 4
    if "helmet" in n:
        return 0
    if n in ("motorcycle", "motorbike"):
        return 2
    if "plate" in n or "registration" in n:
        return 3
    return None


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def contain(a, b):
    """intersection over the SMALLER box - robust when teacher and human box styles differ"""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(1e-9, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def source_classes(label_lines):
    """Classes the human annotator labelled in this image (we trust those completely)."""
    return {int(l.split()[0]) for l in label_lines if l.strip()}


def run(teachers, data: Path, splits, conf: dict, ppe_hardhat_is_labelled=True, imgsz=640):
    added = Counter()
    for split in splits:
        img_dir, lbl_dir = data / split / "images", data / split / "labels"
        imgs = sorted(img_dir.iterdir())
        for path, allowed, model in teachers:
            print(f"[{split}] teacher {Path(path).name} -> {[UNIFIED[c] for c in allowed]}")
            for i in range(0, len(imgs), 32):
                batch = imgs[i:i + 32]
                for img, res in zip(batch, model(list(map(str, batch)), conf=min(conf.values()), imgsz=imgsz, verbose=False)):
                    lbl = lbl_dir / f"{img.stem}.txt"
                    lines = [l for l in (lbl.read_text().splitlines() if lbl.exists() else []) if l.strip()]
                    human = source_classes(lines)
                    if img.name.startswith("ppe_") and ppe_hardhat_is_labelled:
                        human |= {4}
                    boxes = [(int(l.split()[0]), xyxy(*map(float, l.split()[1:]))) for l in lines]
                    for b in res.boxes:
                        c = canon(res.names[int(b.cls)])
                        if c is None or c not in allowed or c in human:
                            continue           # never second-guess a class the human labelled
                        if float(b.conf) < conf.get(UNIFIED[c], 0.5):
                            continue
                        x, y, w, h = b.xywhn[0].tolist()
                        box = xyxy(x, y, w, h)
                        # construction scenes: any helmet is a hard-hat -> never add riding 'helmet' there
                        if img.name.startswith("ppe_") and c == 0:
                            continue
                        # a 'head' that sits on a hard-hat is a teacher error (QA finding)
                        if c == 1 and any(ec == 4 and contain(box, e) > 0.25 for ec, e in boxes):
                            continue
                        # skip duplicates: same class by IoU, head-level classes by containment
                        if any((ec == c and iou(box, e) > 0.4) or
                               ({ec, c} <= {0, 1, 4} and contain(box, e) > 0.5) for ec, e in boxes):
                            continue
                        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                        boxes.append((c, box))
                        added[UNIFIED[c]] += 1
                    lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
                print(f"   {min(i + 32, len(imgs))}/{len(imgs)}  added so far: {dict(added)}", end="\r")
            print()
    for c in data.rglob("*.cache"):
        c.unlink()
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets/helmet_unified"))
    ap.add_argument("--round", type=int, choices=(1, 2), default=1)
    ap.add_argument("--student", help="round-2 teacher (round-1 best.pt)")
    ap.add_argument("--coco", default="yolo11s.pt", help="COCO model used for motorcycles in round 1")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])   # NEVER test
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    # conservative thresholds: a missed pseudo-box hurts less than a wrong one
    conf = {"helmet": 0.55, "head": 0.55, "motorcycle": 0.5, "license_plate": 0.5, "hardhat": 0.6}
    if a.round == 1:
        teachers = [
            (str(ROOT / "models/teacher_helmet_dip.pt"), {0, 1}, YOLO(str(ROOT / "models/teacher_helmet_dip.pt"))),
            (a.coco, {2}, YOLO(a.coco)),
        ]
    else:
        assert a.student, "--student is required for round 2"
        teachers = [(a.student, {0, 1, 2, 3, 4}, YOLO(a.student))]
    added = run(teachers, Path(a.data), a.splits, conf, imgsz=a.imgsz)
    print("pseudo-labels added:", dict(added))


if __name__ == "__main__":
    main()
