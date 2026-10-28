# HelmetAI-Pro — Project Roadmap

Base: **Rishav1317/DIP_PROJECT (HelmetAI)**, merged with KrishArul26 (rider cascade + alarm),
prodbykosta (tracking, PPE association, smoothing, evaluation) and TheSiddheshFulawade (motorcycle /
plate classes, Open Images data, Tesseract OCR), then extended with six novelties.

Status legend: ✅ done in this release · 🔄 needs the full GPU training run · 🧭 future work

---

## Phase 0 — Audit & clean-up of the base  ✅
| # | Item | Status |
|---|------|--------|
| 0.1 | Read all 4 repos line-by-line, inventory every feature | ✅ |
| 0.2 | **Remove the leaked Roboflow API key** (`backend/.env` was committed) → `.env.example`, `.env` git-ignored | ✅ (revoke the old key on Roboflow) |
| 0.3 | Fix video summary bug (server sent `max_helmets`, UI read `total_helmets` → always "compliant") | ✅ |
| 0.4 | Per-frame counting inflated compliance → count **unique tracked riders** | ✅ |
| 0.5 | `/api/stream/video?source=` accepted any server path → whitelist (webcam, rtsp/http, YouTube, samples/uploads) | ✅ |
| 0.6 | Hard-coded `localhost:8000` in every page → `VITE_API_URL` | ✅ |
| 0.7 | Training passed the API key on the command line, never reloaded the model → admin-token protected training + hot reload | ✅ |
| 0.8 | Substring violation check (`"no" in label`) → canonical class mapping | ✅ |

## Phase 1 — Unified dataset  ✅
| # | Item | Status |
|---|------|--------|
| 1.1 | 5 unified classes: `helmet, head, motorcycle, license_plate, hardhat` | ✅ |
| 1.2 | Merge Open Images (Siddhesh, 1,923 imgs) + PPE (10,500 imgs); drop jackets | ✅ |
| 1.3 | Label validation (clip, degenerate, malformed) + **dHash de-duplication** (1,084 duplicates removed) | ✅ |
| 1.4 | Rebalancing (PPE subsampled to 3,000 train) → 4,354 / 584 / 785 images | ✅ |
| 1.5 | **Finding:** Open Images subset is single-class per image (0 co-occurrences) → partial labels | ✅ documented |
| 1.6 | **Teacher-student label completion**: R1 764 boxes, visual QA removed 285 faulty (hard-hat→helmet), R2 +49 (23 plates) | ✅ |

## Phase 2 — Model  🔄
| # | Item | Status |
|---|------|--------|
| 2.1 | Training recipe `training/run_all.sh` (merge → pseudo-label → train R1 → pseudo-label R2 → train → eval → ONNX) | ✅ |
| 2.2 | Traffic-specific augmentation (brightness, occlusion/erasing, no vertical flip, mosaic, copy-paste) | ✅ |
| 2.3 | CPU profile (YOLO11n/416): R1 7 ep + R2 5 ep → val mAP50 0.653, test 0.651 | ✅ |
| 2.4 | **GPU profile (YOLO11s/640, 120 ep) on Colab/Kaggle** — `training/HelmetAI_Pro_GPU_Training.ipynb` | 🔄 run it |
| 2.5 | Honest per-class evaluation on partially-labelled test data + robustness ablation | ✅ code |
| 2.6 | ONNX export → backend runs without PyTorch (edge-friendly) | ✅ |

## Phase 3 — Processing pipeline  ✅
| Module | Origin | What it does |
|---|---|---|
| `enhance.py` | **Novelty #3** | Scene analysis (L*, σ, dark channel, Laplacian) → gamma+CLAHE / DCP dehaze / CLAHE / unsharp |
| `detector.py` | base + PPE | .pt / ONNX / Roboflow backends, ensembles, per-class thresholds, canonical labels |
| `tracker.py` | PPE | IoU + centre-distance tracker, edge cleanup, speed estimate |
| `riders.py` | KrishArul idea + PPE gating, **Novelties #1/#2** | rider zones, depth filter, helmet-carried, hard-hat, triple riding, temporal majority vote |
| `plate_ocr.py` | Siddhesh + **Novelty #4** | plate refinement, character-component filter, multi-scale Tesseract, Indian grammar repair, multi-frame vote |
| `evidence.py` | **Novelty #5** | SHA-256 record hash + hash chain, tamper detection |
| `challan.py` | **Novelty #5** | e-Challan PDF with fines, evidence, hashes, QR |
| `privacy.py` | **Novelty #6** | blur all non-violators (faces + plates) |
| `severity.py` | extra | 0-100 risk score (type, unprotected riders, speed, repeat offender) |
| `engine.py` | — | per-source sessions, unique counting, one evidence record per violating track |

## Phase 4 — API & UI  ✅
- All base endpoints kept; new: violations, review, challan, evidence verify, settings, model info/reload, enhancement preview, samples.
- UI: Dashboard (unique riders, types, severity, repeat offenders, chain status), Image, Video (fixed),
  Live (sample chooser, record MP4), **Violations & e-Challan**, **Enhancement Lab**, Model & Training, History, Settings.
- Shared WebSocket + audible alarm (WebAudio).

## Phase 4b — Model selection & policy (data-driven)  ✅
- Per-class comparison on test subsets → ensemble (unified for helmet/head/plate/hard-hat, COCO for motorcycle/person).
- Night enhancement hurt motorcycle AP on validation → detector sees raw night frames; fog/blur/low-contrast enhanced.
- Real-media review found & fixed: bystanders behind parked bikes / officers beside bikes counted as riders
  (person-based gating), letterboxed daylight videos classified as night (border-aware scene analysis).

## Phase 5 — Verification  ✅
- `tests/test_rules.py` — 7 rule tests incl. hand-annotated real photos (pedestrian & police officer correctly ignored, pillion without helmet caught).
- `tests/test_pipeline_gt.py` — end-to-end: still images, 24-frame panning video (1 track ID, 1 evidence record),
  real plate OCR with fusion, hash-chain verify, **tamper detection**, challan.
- API smoke test of every endpoint (`tests/serve_demo.py`).

## Phase 6 — Future work  🧭
1. Annotate 150–300 **fully-labelled Indian road images** (CVAT / Roboflow) → unbiased test set + fine-tune.
2. Pose-based worn-helmet verification (YOLO-pose keypoints inside helmet box) as a second opinion to the geometry rule.
3. Plate super-resolution (Real-ESRGAN) for far plates; EasyOCR/PaddleOCR as alternative recognisers.
4. Explainability: EigenCAM heat-maps on the evidence page.
5. Multi-camera hotspot map, repeat-offender alerts, SMS / e-mail notice.
6. Edge deployment benchmark: ONNX INT8 on Raspberry Pi 5 / Jetson Nano.
7. Active-learning loop: rejected (false-positive) reviews exported as hard negatives for re-training.
