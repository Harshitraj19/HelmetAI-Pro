"""
inspect_labels.py — visual QA of pseudo-labels: human boxes solid, pseudo boxes dashed.

    python training/inspect_labels.py --human datasets/labels_train_human_backup --n 12 --out docs/pseudo_label_check.jpg
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

NAMES = ["helmet", "head", "motorcycle", "license_plate", "hardhat"]
COL = [(60, 200, 80), (40, 40, 230), (230, 160, 40), (0, 215, 255), (0, 150, 255)]


def boxes(path, W, H):
    out = []
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                c, x, y, w, h = l.split()
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                out.append((int(c), int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)))
    return out


def dashed(img, p1, p2, col, t=2, d=8):
    x1, y1 = p1; x2, y2 = p2
    for x in range(x1, x2, d * 2):
        cv2.line(img, (x, y1), (min(x + d, x2), y1), col, t); cv2.line(img, (x, y2), (min(x + d, x2), y2), col, t)
    for y in range(y1, y2, d * 2):
        cv2.line(img, (x1, y), (x1, min(y + d, y2)), col, t); cv2.line(img, (x2, y), (x2, min(y + d, y2)), col, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/helmet_unified/train")
    ap.add_argument("--human", default="datasets/labels_train_human_backup")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="docs/pseudo_label_check.jpg")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    data, human = Path(a.data), Path(a.human)
    changed = []
    for lbl in sorted((data / "labels").glob("*.txt")):
        h = human / lbl.name
        if h.exists() and len(lbl.read_text().splitlines()) > len(h.read_text().splitlines()):
            changed.append(lbl)
    random.Random(a.seed).shuffle(changed)
    tiles = []
    for lbl in changed[: a.n]:
        img = cv2.imread(str(next((data / "images").glob(lbl.stem + ".*"))))
        H, W = img.shape[:2]
        hb = boxes(human / lbl.name, W, H)
        for c, *b in boxes(lbl, W, H):
            if (c, *b) in hb:
                cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), COL[c], 2)
            else:
                dashed(img, tuple(b[:2]), tuple(b[2:]), COL[c])
                cv2.putText(img, "pseudo " + NAMES[c], (b[0], max(12, b[1] - 4)), 0, 0.5, COL[c], 2)
        tiles.append(cv2.resize(img, (360, 270)))
    while len(tiles) % 4:
        tiles.append(np.zeros((270, 360, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    Path(a.out).parent.mkdir(exist_ok=True)
    cv2.imwrite(a.out, grid)
    print(f"{len(changed)} images received pseudo-labels; grid -> {a.out}")


if __name__ == "__main__":
    main()
