# 🛡️ HelmetAI-Pro — Helmet, Rider & Number-Plate Enforcement System

> DIP Project · built on **Rishav1317/DIP_PROJECT** and merged with three other open-source helmet projects ·
> YOLO11 + adaptive image enhancement + tracking + plate OCR + tamper-proof e-Challans · FastAPI + React

**One line:** an end-to-end, privacy-preserving traffic-enforcement system that works in poor light and fog,
checks that the helmet is actually *worn*, counts riders per motorcycle (triple riding), reads the number plate
across several frames, and issues hash-chained, tamper-evident e-Challans.

---

## What came from where

| Source repo | Taken into HelmetAI-Pro |
|---|---|
| **Rishav1317/DIP_PROJECT** (base) | FastAPI + React app, image / video / live MJPEG, WebSocket alerts, SQLite history, analytics dashboard, Roboflow training page, label normalisation, cloud fallback |
| **KrishArul26/Motorbike-Helmet_detection** | "find the motorcycle first, then the riders" cascade idea, audible violation alarm |
| **prodbykosta/ppe-safety-detection-ai** | IoU tracker with IDs + edge cleanup, region-gated PPE association, temporal majority smoothing, per-class thresholds, box filters, ONNX backend, evaluation mode, MP4 export, workplace (site) mode, 10.5k-image hard-hat dataset |
| **TheSiddheshFulawade/Helmet-…-Number-Plate** | Motorcycle & plate classes, Open Images dataset (1.9k images), "bare head = violation" idea, Otsu + Tesseract plate OCR |

## Novelties (not in any of the four repos)

1. **Helmet worn vs carried** — a helmet on the handlebar/seat with a bare head above it is flagged `HELMET_NOT_WORN`.
2. **Rider counting & triple riding** — heads are assigned to *their* motorcycle (rider zone + depth filter), pillions included.
3. **Adaptive DIP enhancement** — per-frame scene analysis routes to gamma+CLAHE (night), dark-channel-prior dehazing (fog),
   CLAHE (low contrast) or unsharp masking (blur) *before* detection; measured with a night/fog/blur ablation.
4. **Multi-frame plate reconstruction** — sharpest crops per track → plate refinement → character-component filter →
   multi-scale Tesseract → Indian-plate grammar repair (position-aware O↔0, I↔1, B↔8 …) → character-wise vote across frames.
5. **Tamper-evident evidence + e-Challan** — SHA-256 record hash over fields *and* images, hash-chained records,
   one-click chain verification, PDF challan with fines, evidence and QR.
6. **Privacy by design** — faces and plates of everyone except the violator are blurred in every output.

Extras: **fake-helmet check** (construction hard-hat on a rider = invalid), severity score 0–100, repeat-offender detection,
human-in-the-loop review (audited, chain-safe), runtime settings, traffic ↔ site (workplace PPE) modes, ONNX runtime.

---

## Architecture

```
frame ─► Adaptive enhancement ─► Detector (YOLO11 .pt / ONNX / ensemble / Roboflow)
      ─► Motorcycle tracker (IDs) ─► Rider association (zones, depth filter, carried helmet, hard-hat)
      ─► Temporal majority vote ─► Rule engine (NO_HELMET, INVALID_HELMET, HELMET_NOT_WORN, TRIPLE_RIDING)
      ─► Plate fusion OCR ─► Severity ─► Evidence store (SHA-256 chain) ─► e-Challan PDF
      ─► Privacy blur ─► Annotated frame (MJPEG / MP4)  +  WebSocket events ─► React dashboard
```

```
HelmetAI-Pro/
├── backend/            FastAPI app (main.py), config, models (DB), store (hash chain)
│   ├── pipeline/       enhance · detector · tracker · riders · plate_ocr · evidence · challan · privacy · severity · engine
│   └── tests/          test_rules.py · test_pipeline_gt.py · serve_demo.py
├── frontend/           React 19 + Vite (9 pages)
├── training/           prepare_dataset · pseudo_label · train · evaluate · run_all.sh · GPU notebook
├── models/             teacher weights from the source repos + helmet_unified.(pt|onnx) after training
├── datasets/           built by training/prepare_dataset.py (git-ignored)
├── samples/            demo media + generated outputs
└── docs/               ROADMAP.md · metrics.md
```

---

## Quick start

### 1. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Tesseract for plate OCR:  Ubuntu: sudo apt install tesseract-ocr   ·  Windows: UB-Mannheim installer, add to PATH
cp .env.example .env                                     # optional settings
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
API docs: http://localhost:8000/docs

**Which model is used?** `models/helmet_unified.onnx` (or `.pt`) if present; otherwise the backend falls back to an
ensemble of the base project's helmet model + a COCO motorcycle model (auto-downloaded), so it runs out of the box.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev            # http://localhost:5173   (set VITE_API_URL in frontend/.env for another backend)
```

### 3. Train the unified model (recommended on a GPU)
Trained weights are already included (`models/helmet_unified.onnx`). To train the stronger GPU version:
* **Colab / Kaggle:** open `training/HelmetAI_Pro_GPU_Training.ipynb`, upload the project zip, *Run all* (~1.5–2 h on a T4).
  Copy `models/helmet_unified.onnx` back and press **Hot-reload model** on the *Model & Training* page.
* **Locally:** `bash training/run_all.sh gpu` (or `cpu` for a slow smoke test).

### 4. Tests
```bash
cd backend
python tests/test_rules.py           # rule engine (real annotated photos + synthetic cases)
python tests/test_pipeline_gt.py     # end-to-end: tracking, OCR fusion, evidence chain, tamper detection, challan
python tests/test_api.py             # HTTP API: settings validation, image/video endpoints, H.264 output, challan, chain
```

### Notes for Windows
* **Plate OCR** needs the Tesseract binary (UB-Mannheim installer). It is found automatically in
  `C:\Program Files\Tesseract-OCR`, on `PATH`, or via `TESSERACT_CMD`; without it everything else works and
  plates are reported as "not read" (the *Model & Training* page shows the OCR status).
* Annotated MP4s are written as **H.264** (Media Foundation), so they play directly in the browser.

### Performance knobs (`backend/.env`)
| variable | default | effect |
|---|---|---|
| `MAX_FRAME_WIDTH` | 1280 | videos/streams wider than this are downscaled before processing (0 = off) |
| `ORT_THREADS` | auto | ONNX Runtime threads per model (auto = physical cores; don't set it to the logical-core count) |
| `ACCURACY_PROFILE` | fast | `high` uses COCO YOLO11s for motorcycles (more accurate, slower) |

---

## Dataset

| split | images | Open Images | PPE | helmet | head | motorcycle | plate | hard-hat |
|---|---|---|---|---|---|---|---|---|
| train | 4,354 | 1,354 | 3,000 | 827 | 1,487 | 563 | 500 | 6,917 |
| val | 584 | 184 | 400 | 66 | 187 | 89 | 76 | 927 |
| test | 785 | 385 | 400 | 254 | 413 | 144 | 143 | 973 |

(before pseudo-labelling; 1,084 duplicate images removed.)

**Important finding:** the Open Images part is labelled one class per image (an image from the *Motorcycle* set has
no helmet boxes even when the rider wears one). Training on it naively teaches the model that helmets are background.
HelmetAI-Pro fixes this with teacher-student label completion (`training/pseudo_label.py`) and evaluates each class only
on images where that class was annotated (`training/evaluate.py`). Results: `docs/metrics.md`.


## Results (this release — YOLO11n trained on 2 CPU cores)

| class | test mAP50 | vs. base DIP_PROJECT model |
|---|---|---|
| helmet | **0.649** @640 (0.612 @416) | 0.307 → **2.1×** |
| head (no helmet) | **0.583** @640 (0.531 @416) | 0.246 → **2.4×** |
| motorcycle | 0.671 (COCO YOLO11n 0.756 / 11s 0.819 used in the ensemble) | – |
| license plate | 0.516 | – |
| hard-hat | **0.923** | – |

* Production detector = **ensemble chosen per class from these numbers**: unified model (helmet, head, plate, hard-hat)
  + COCO YOLO11 (motorcycle, person). All ONNX → runs without PyTorch, ~5 FPS on a 2-core CPU.
* **Fog enhancement: +25 % mAP50** (0.473 → 0.591). Night: the detector uses the raw frame (decided on validation —
  see `docs/metrics.md`), enhancement is used for display / evidence / OCR.
* Plate OCR fusion recovered `GJ 1 NC 9343` exactly from a 16-px-tall plate where single reads failed.
* Tests: 10 rule tests + end-to-end pipeline test (tracking, fusion, hash chain, tamper detection, challan) all pass.
* Sample outputs: `samples/outputs/demo/` (annotated images & videos, `challans/`, `summary.json`).
* Full numbers, ablation and training curves: `docs/metrics.md`, `docs/training/`.

**Known limitations (CPU model):** misses some bare heads in crowded scenes; police caps / costume crowns can be
confused with helmets / hard-hats; plates < 20 px tall are rarely readable. Run the GPU notebook for the full model —
all violations are also marked "pending review" for a human officer.

## Main API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/detect/image` · `/api/detect/image.json` | detect on an image (`?mode=auto\|traffic\|site`; auto = traffic rules if a motorcycle is visible, else every person is helmet-checked) |
| POST | `/api/stream/stop?run_id=` | stop a stream and release its camera (no `run_id` = stop every running stream) |
| POST | `/api/detect/video` | upload video → MJPEG + saved H.264 MP4 (`?mode=traffic\|site&stride=1..5`) |
| GET | `/api/stream/video?source=0&record=true&mode=site` | webcam / RTSP / YouTube / sample live stream (live sources always process the newest frame) |
| GET | `/api/violations` · `/api/violations/{id}` | evidence records (filter by type, plate, status, severity) |
| POST | `/api/violations/{id}/review` | confirm / reject (audited) |
| GET | `/api/violations/{id}/challan.pdf` | e-Challan |
| GET | `/api/evidence/verify` | verify the hash chain |
| GET/PUT | `/api/settings` | enhancement, privacy, OCR, mode … live |
| GET | `/api/stats` | dashboard analytics |
| WS | `/ws/notifications` | live violations, per-frame counters |

## Security notes
* The base repo committed a Roboflow API key in `backend/.env` — **revoke it**. Keys now live only in an untracked `.env`.
* Set `ADMIN_TOKEN` to protect delete / training / model-reload endpoints.
* e-Challans are marked "pending officer review"; fines are indicative (MV Amendment Act 2019 headline values).

See **docs/ROADMAP.md** for the phase-by-phase plan and status.
