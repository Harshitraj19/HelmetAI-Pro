"""
evaluate.py — honest evaluation on partially-labelled test data.

The Open Images test images are labelled for ONE class each. A plain `model.val()` would count
every correct helmet found in a "motorcycle" image as a false positive. So for each class we
evaluate ONLY on test images whose annotator labelled that class (and on PPE images for
hard-hat), and report that class's AP.  Output: models/metrics.json + docs/metrics.md

Also runs the enhancement ablation: mAP on synthetically degraded copies (night / fog / blur)
of the same images, with and without the adaptive DIP enhancement.

Usage: python training/evaluate.py --weights models/helmet_unified.pt [--ablation]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from pipeline.enhance import degrade, enhance  # noqa: E402

UNIFIED = ["helmet", "head", "motorcycle", "license_plate", "hardhat"]


def subset_yaml(test_dir: Path, cls: int, tmp: Path, transform=None) -> Path:
    out = tmp / f"c{cls}_{transform or 'clean'}"
    for sub in ("images", "labels"):
        (out / "val" / sub).mkdir(parents=True, exist_ok=True)
    for lbl in (test_dir / "labels").glob("*.txt"):
        lines = [l for l in lbl.read_text().splitlines() if l.strip()]
        if not any(int(l.split()[0]) == cls for l in lines):
            continue
        img = next((test_dir / "images").glob(lbl.stem + ".*"))
        dst = out / "val/images" / img.name
        if transform:
            kind, enh = transform.split("+") if "+" in transform else (transform, None)
            im = degrade(cv2.imread(str(img)), kind)
            if enh:
                im, _ = enhance(im, "auto")
            cv2.imwrite(str(dst), im)
        else:
            shutil.copy2(img, dst)
        # keep only this class's boxes -> no false "false positives" from unlabelled classes
        (out / "val/labels" / lbl.name).write_text("\n".join(l for l in lines if int(l.split()[0]) == cls) + "\n")
    y = out / "data.yaml"
    y.write_text(f"path: {out}\ntrain: val/images\nval: val/images\nnames: {UNIFIED}\n")
    return y


def eval_class(model, yaml, cls, imgsz):
    m = model.val(data=str(yaml), imgsz=imgsz, verbose=False, plots=False, classes=[cls])
    idx = list(m.box.ap_class_index)
    if cls not in idx:
        return dict(precision=0, recall=0, mAP50=0, mAP50_95=0)
    p, r, ap50, ap = m.box.class_result(idx.index(cls))
    n = len(list((Path(yaml).parent / "val/images").iterdir()))
    return dict(images=n, precision=round(p, 4), recall=round(r, 4), mAP50=round(ap50, 4), mAP50_95=round(ap, 4))


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/helmet_unified.pt"))
    ap.add_argument("--test", default=str(ROOT / "datasets/helmet_unified/test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ablation", action="store_true")
    a = ap.parse_args()
    model = YOLO(a.weights)
    tmp = Path(tempfile.mkdtemp())
    res = {"weights": Path(a.weights).name, "imgsz": a.imgsz, "per_class": {}}
    for c, name in enumerate(UNIFIED):
        res["per_class"][name] = eval_class(model, subset_yaml(Path(a.test), c, tmp), c, a.imgsz)
        print(name, res["per_class"][name])
    pc = res["per_class"].values()
    res["mAP50"] = round(sum(x["mAP50"] for x in pc) / len(UNIFIED), 4)
    res["mAP50_95"] = round(sum(x["mAP50_95"] for x in pc) / len(UNIFIED), 4)

    if a.ablation:   # robustness study: does the DIP enhancement recover accuracy?
        res["ablation"] = {}
        for kind in ("night", "fog", "blur"):
            for variant in (kind, f"{kind}+enh"):
                aps = [eval_class(model, subset_yaml(Path(a.test), c, tmp, variant), c, a.imgsz)["mAP50"]
                       for c in (0, 1, 2)]   # the rider-critical classes
                res["ablation"][variant] = round(sum(aps) / len(aps), 4)
                print("ablation", variant, res["ablation"][variant])

    out = ROOT / "models/metrics.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old["per_class_eval"] = res
    out.write_text(json.dumps(old, indent=2))
    md = ["| class | test images | precision | recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    md += [f"| {k} | {v.get('images','-')} | {v['precision']} | {v['recall']} | {v['mAP50']} | {v['mAP50_95']} |"
           for k, v in res["per_class"].items()]
    md.append(f"| **mean** | | | | **{res['mAP50']}** | **{res['mAP50_95']}** |")
    if "ablation" in res:
        md += ["", "| condition | mAP50 raw | mAP50 with adaptive enhancement |", "|---|---|---|"]
        md += [f"| {k} | {res['ablation'][k]} | {res['ablation'][k + '+enh']} |" for k in ("night", "fog", "blur")]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
