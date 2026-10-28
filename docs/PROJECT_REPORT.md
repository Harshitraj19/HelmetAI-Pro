# HelmetAI-Pro: A Privacy-Preserving, Weather-Robust Two-Wheeler Helmet Enforcement System

*Digital Image Processing project report (draft for reviews).*

## Abstract
Two-wheeler riders account for the largest share of road deaths in India, and helmet non-compliance is the single
biggest modifiable factor. Existing student projects detect "helmet / no helmet" on single images, count the same
rider on every video frame, break down at night or in fog, cannot tell a worn helmet from a carried one, and produce
no usable evidence. HelmetAI-Pro merges four open-source systems into one pipeline and adds six contributions:
(1) worn-vs-carried helmet reasoning, (2) per-motorcycle rider counting with triple-riding detection,
(3) scene-adaptive image enhancement (adaptive gamma, CLAHE, dark-channel-prior dehazing, unsharp masking, bilateral
denoising) placed before detection, (4) multi-frame licence-plate reconstruction with plate-grammar repair,
(5) a SHA-256 hash-chained, tamper-evident evidence log with auto-generated e-Challans, and (6) privacy by design,
blurring everyone except the violator. A unified five-class YOLO11 detector is trained on two merged datasets whose
missing labels are completed by teacher-student pseudo-labelling.

## 1. Problem statement
Given a road image or video stream, decide for every motorcycle (i) how many people ride it, (ii) whether each one wears
a valid riding helmet on the head, (iii) the vehicle number, and produce verifiable evidence for violations only,
while working under poor illumination, haze and motion blur.

## 2. Related systems (merged)
| System | Approach | Limitation addressed here |
|---|---|---|
| DIP_PROJECT (base) | YOLO helmet / no-helmet, FastAPI + React, WebSocket alerts | per-frame counting, no rider/plate logic, leaked API key, summary bug |
| KrishArul26 | Faster R-CNN helmet + SSD motorbike (TF1) | no association between the two models, image only |
| prodbykosta PPE | person tracking + region-gated PPE association + smoothing | construction PPE only, no UI / evidence |
| Siddhesh | YOLOv5 4-class + Tesseract | single-class-per-image labels, one-frame OCR |

## 3. Dataset
* Open Images subset (Helmet, Human head, Motorcycle, Vehicle registration plate) and a Roboflow construction PPE set.
* Unified classes: helmet, head, motorcycle, license_plate, hardhat.
* Cleaning: box clipping, degenerate box removal, 64-bit dHash near-duplicate removal (1,084 images), PPE rebalancing.
* **Partial-label problem:** 0 Open Images pictures carry more than one class. Solved with teacher-student label
  completion: the base project's helmet model and a COCO detector add missing helmet/head/motorcycle boxes
  (conservative thresholds, never overriding a class a human labelled), followed by visual QA that removed 285 faulty
  teacher boxes (hard-hats mislabelled as helmets); a round-2 student adds plates.
* Evaluation per class on test images annotated for that class, so unlabelled-but-correct detections are not
  counted as false positives.

## 4. Digital image processing methods
| Stage | Technique | Purpose |
|---|---|---|
| Scene analysis | mean / σ of CIE-L*, dark-channel mean, Laplacian variance, Sobel gradient anisotropy | classify night / fog / low-contrast / motion-blur |
| Night | adaptive gamma γ = log(0.45)/log(μ/255), bilateral filter, CLAHE on L* | lift shadows without amplifying noise |
| Fog | dark channel prior, atmospheric light from top 0.1 % dark pixels, guided-filter transmission refinement | haze removal |
| Low contrast | CLAHE (LAB) | local contrast |
| Motion blur | unsharp masking | edge recovery |
| Plate | bright-region morphology to refine the plate rectangle, CLAHE, bilateral, min-area-rect deskew, Otsu & adaptive Gaussian binarisation, connected-component character filtering, multi-scale resampling | OCR-ready characters |
| Privacy | pixelation + Gaussian blur | irreversible anonymisation |
| Evidence frame selection | Laplacian-variance sharpness × size | best plate crops |

## 5. Decision logic
* Rider zone of a motorcycle = box widened 15 % and extended upward 1.6× its height; each head-level detection
  goes to the best-scoring bike; heads < 60 % of the largest rider head on that bike are treated as people standing
  behind (depth filter).
* A helmet whose centre lies inside the motorcycle body while a bare head is present = helmet carried, not worn.
* A construction hard-hat on a rider = invalid helmet.
* > 2 riders = triple riding.
* Majority vote over a 7-frame window per tracked bike; a violation must persist ≥ 3 frames.
* Severity = type weights + 10 per extra unprotected rider + speed bonus + repeat-offender bonus (0–100).

## 6. Evidence integrity
record_hash = SHA-256(canonical JSON of fields ‖ SHA-256 of each evidence image);
chain_hash = SHA-256(previous chain_hash ‖ record_hash). Editing any image or field, or deleting / re-ordering a
record, breaks verification at exactly that record (tested). Human review status lives outside the hash.

## 7. Results
See `docs/metrics.md` (per-class test AP, robustness ablation) and `samples/outputs/` (annotated media, challans).

## 8. Limitations & future work
CPU-trained model is a proof of concept; full accuracy needs the GPU recipe. Open Images contains few Indian road
scenes, so a fully-labelled Indian test set (150–300 images) is the most valuable next step. Plate OCR on plates
smaller than ~20 px tall is unreliable even with fusion. Future: pose-based worn-helmet verification, plate
super-resolution, EigenCAM explanations, multi-camera hotspot maps, Jetson / Raspberry Pi INT8 deployment.
