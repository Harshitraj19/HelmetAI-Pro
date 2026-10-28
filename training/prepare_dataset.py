"""
prepare_dataset.py — build the unified HelmetAI-Pro dataset from all source repos.

Sources merged
--------------
  A. Siddhesh (Open Images v6, YOLO txt)   Helmet / Human head / Motorcycle / Vehicle registration plate
  B. PPE safety (Roboflow, YOLO txt)       Safety-Helmet / Reflective-Jacket
  C. DIP_PROJECT                           no labelled data; its best.pt is used as a *teacher*
                                           in pseudo_label.py to add missing "head" boxes to B
  D. KrishArul26                           no data shipped (TF frozen graphs only) - idea reused, not data

Unified classes
---------------
  0 helmet          riding / sports helmet worn or carried
  1 head            bare head  ->  NO-HELMET evidence
  2 motorcycle
  3 license_plate
  4 hardhat         construction hard-hat  ->  NOT a valid riding helmet (novelty: fake-helmet check)

What it does
------------
  * remaps every source class to the unified ids (drops classes we do not use, e.g. jacket)
  * validates labels: clips to [0,1], drops degenerate / tiny / malformed boxes
  * removes exact and near-duplicate images across sources (64-bit dHash, Hamming <= 4)
  * rebalances: PPE has ~10k construction images vs ~1.9k road images, so it is subsampled
  * writes train / val / test splits + data.yaml + a stats report (JSON + Markdown)

Usage
-----
  python training/prepare_dataset.py \
      --siddhesh  ../repos/Helmet-Detection-and-Number-Plate-Recognition-Using-YOLO/data \
      --ppe       ../repos/ppe-safety-detection-ai/dataset/safety-Helmet-Reflective-Jacket \
      --out       datasets/helmet_unified  --ppe-train 3000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

UNIFIED = ["helmet", "head", "motorcycle", "license_plate", "hardhat"]

# source-class-id -> unified-id (None = drop)
MAP_SIDDHESH = {0: 0, 1: 1, 2: 2, 3: 3}
MAP_PPE = {0: 4, 1: None}          # Safety-Helmet -> hardhat, Reflective-Jacket -> dropped

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIDE = 0.004                   # boxes thinner than 0.4% of the image are noise


# --------------------------------------------------------------------------- helpers
def dhash(path: Path, size: int = 8) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_labels(txt: Path, cmap: dict, stats: Counter) -> list[tuple]:
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            stats["malformed_line"] += 1
            continue
        try:
            c = int(float(p[0]))
            x, y, w, h = map(float, p[1:])
        except ValueError:
            stats["malformed_line"] += 1
            continue
        u = cmap.get(c)
        if u is None:
            stats["dropped_class"] += 1
            continue
        # clip box to image
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        w, h = x2 - x1, y2 - y1
        if w < MIN_SIDE or h < MIN_SIDE:
            stats["degenerate_box"] += 1
            continue
        out.append((u, (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return out


def collect(img_dir: Path, lbl_dir: Path, cmap: dict, source: str, stats: Counter) -> list[dict]:
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        labels = read_labels(lbl_dir / f"{img.stem}.txt", cmap, stats)
        items.append({"img": img, "labels": labels, "source": source})
    return items


def dedupe(items: list[dict], stats: Counter, max_dist: int = 4) -> list[dict]:
    """Remove exact + near duplicates (bucketed by 16-bit prefix for speed)."""
    kept, buckets = [], defaultdict(list)
    for it in items:
        h = dhash(it["img"])
        if h is None:
            stats["unreadable_image"] += 1
            continue
        key = h >> 48
        dup = any(bin(h ^ o).count("1") <= max_dist for o in buckets[key])
        if dup:
            stats["duplicate_image"] += 1
            continue
        buckets[key].append(h)
        kept.append(it)
    return kept


def write_split(items: list[dict], out: Path, split: str, link: bool):
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for it in items:
        name = f"{it['source']}_{it['img'].stem}{it['img'].suffix.lower()}"
        dst = out / split / "images" / name
        if not dst.exists():
            if link:
                try:
                    os.link(it["img"], dst)
                except OSError:
                    shutil.copy2(it["img"], dst)
            else:
                shutil.copy2(it["img"], dst)
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in it["labels"]]
        (out / split / "labels" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siddhesh", required=True, help="…/Helmet-Detection-…/data")
    ap.add_argument("--ppe", required=True, help="…/dataset/safety-Helmet-Reflective-Jacket")
    ap.add_argument("--out", default="datasets/helmet_unified")
    ap.add_argument("--ppe-train", type=int, default=3000, help="PPE images kept for train (rebalancing)")
    ap.add_argument("--ppe-eval", type=int, default=400, help="PPE images kept for val and for test")
    ap.add_argument("--val-frac", type=float, default=0.12, help="fraction of Siddhesh train moved to val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    stats = Counter()
    sid, ppe, out = Path(a.siddhesh), Path(a.ppe), Path(a.out)
    if out.exists():
        shutil.rmtree(out)

    # ---- A. Siddhesh (road scenes, all 4 road classes)
    s_train = collect(sid / "train/images", sid / "train/labels", MAP_SIDDHESH, "oi", stats)
    s_test = collect(sid / "test/images", sid / "test/labels", MAP_SIDDHESH, "oi", stats)
    # ---- B. PPE (construction hard-hats)
    p_train = collect(ppe / "train/images", ppe / "train/labels", MAP_PPE, "ppe", stats)
    p_val = collect(ppe / "valid/images", ppe / "valid/labels", MAP_PPE, "ppe", stats)
    p_test = collect(ppe / "test/images", ppe / "test/labels", MAP_PPE, "ppe", stats)
    stats["raw_images"] = sum(map(len, (s_train, s_test, p_train, p_val, p_test)))

    # global de-duplication (test first so test images win, avoiding train/test leakage)
    ordered = s_test + p_test + p_val + s_train + p_train
    kept = {id(x) for x in dedupe(ordered, stats)}
    f = lambda L: [x for x in L if id(x) in kept]
    s_train, s_test, p_train, p_val, p_test = map(f, (s_train, s_test, p_train, p_val, p_test))

    # drop PPE images that lost all boxes (only jackets) - they add no helmet signal
    nz = lambda L: [x for x in L if x["labels"]]
    p_train, p_val, p_test = nz(p_train), nz(p_val), nz(p_test)

    rng.shuffle(s_train); rng.shuffle(p_train); rng.shuffle(p_val); rng.shuffle(p_test)
    n_val = int(len(s_train) * a.val_frac)
    splits = {
        "train": s_train[n_val:] + p_train[: a.ppe_train],
        "val": s_train[:n_val] + p_val[: a.ppe_eval],
        "test": s_test + p_test[: a.ppe_eval],
    }
    for k, v in splits.items():
        write_split(v, out, k, link=not a.copy)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(UNIFIED)}\nnames: {UNIFIED}\n"
    )

    report = {"classes": UNIFIED, "cleaning": dict(stats), "splits": {}}
    for k, v in splits.items():
        cc = Counter(c for it in v for c, *_ in it["labels"])
        report["splits"][k] = {
            "images": len(v),
            "by_source": dict(Counter(it["source"] for it in v)),
            "boxes": {UNIFIED[c]: cc.get(c, 0) for c in range(len(UNIFIED))},
            "background_images": sum(1 for it in v if not it["labels"]),
        }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))

    md = ["# Unified dataset report", "", f"Classes: {', '.join(UNIFIED)}", "", "## Cleaning", ""]
    md += [f"- {k}: {v}" for k, v in sorted(stats.items())]
    md += ["", "## Splits", "", "| split | images | oi | ppe | " + " | ".join(UNIFIED) + " |",
           "|---|---|---|---|" + "---|" * len(UNIFIED)]
    for k, r in report["splits"].items():
        md.append(f"| {k} | {r['images']} | {r['by_source'].get('oi',0)} | {r['by_source'].get('ppe',0)} | "
                  + " | ".join(str(r["boxes"][c]) for c in UNIFIED) + " |")
    (out / "dataset_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
