# HelmetAI-Pro — model metrics

Model: YOLO11n, 416 px, CPU-trained (models/helmet_unified.pt / .onnx). Test = held-out split; each class evaluated only on test images annotated for that class (partial-label-safe).

### Per-class test results
| class | test images | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| helmet | 100 | 0.7015 | 0.5748 | 0.6121 | 0.328 |
| head | 98 | 0.8001 | 0.431 | 0.5307 | 0.2986 |
| motorcycle | 84 | 0.71 | 0.6319 | 0.6711 | 0.3859 |
| license_plate | 103 | 0.6917 | 0.5035 | 0.5157 | 0.2272 |
| hardhat | 400 | 0.924 | 0.8494 | 0.923 | 0.6557 |
| **mean** | | | | **0.6505** | **0.3791** |


### Robustness ablation (TEST split, synthetic degradations, rider classes)
| condition | mAP50 raw | mAP50 with adaptive enhancement |
|---|---|---|
| night | 0.4989 | 0.4747 |
| fog | 0.4733 | 0.5908 |
| blur | 0.5648 | 0.5685 |

### Night-enhancement tuning (VALIDATION split, rider classes helmet / head / motorcycle, mAP50)
| variant | mean | helmet | head | motorcycle |
|---|---|---|---|---|
| raw (no enhancement) | **0.565** | 0.694 | 0.457 | **0.545** |
| gamma + bilateral + CLAHE | 0.531 | 0.697 | 0.464 | 0.431 |
| gamma only | 0.541 | 0.698 | 0.467 | 0.458 |
| gamma + CLAHE 1.5 | 0.521 | 0.685 | 0.455 | 0.423 |
| gamma (target 90) | 0.556 | **0.705** | **0.479** | 0.484 |

**Adopted policy:** enhance before detection for fog / blur / low-contrast (fog +0.118 mAP50 on test);
for night frames the detector sees the raw frame (it was trained with strong brightness augmentation),
while the enhanced frame is used for display, evidence images and plate OCR.

### Training run (CPU, 2 cores, YOLO11n @ 416 px)
* Round 1: 7 epochs on teacher-completed labels → val mAP50 0.620
* Round 2: +49 student pseudo-labels (23 plates) → 5 fine-tuning epochs → **val mAP50 0.653, mAP50-95 0.375**
* Per-class test results above. The GPU recipe (YOLO11s @ 640, 120 epochs) is expected to be substantially higher.
