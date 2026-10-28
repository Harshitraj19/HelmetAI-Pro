#!/usr/bin/env bash
# Full HelmetAI-Pro training recipe — one command.
#   bash training/run_all.sh gpu      (Colab / Kaggle / any CUDA box, ~1.5-2 h on a T4)
#   bash training/run_all.sh cpu      (laptop; slow, use for a smoke test)
set -euo pipefail
PROFILE=${1:-gpu}
cd "$(dirname "$0")/.."

# 0. source datasets (the original repos)
mkdir -p ../repos
[ -d ../repos/Helmet-Detection-and-Number-Plate-Recognition-Using-YOLO ] || \
  git clone --depth 1 https://github.com/TheSiddheshFulawade/Helmet-Detection-and-Number-Plate-Recognition-Using-YOLO ../repos/Helmet-Detection-and-Number-Plate-Recognition-Using-YOLO
[ -d ../repos/ppe-safety-detection-ai ] || \
  git clone --depth 1 https://github.com/prodbykosta/ppe-safety-detection-ai ../repos/ppe-safety-detection-ai

# 1. merge + clean + dedupe + rebalance
python training/prepare_dataset.py \
  --siddhesh ../repos/Helmet-Detection-and-Number-Plate-Recognition-Using-YOLO/data \
  --ppe ../repos/ppe-safety-detection-ai/dataset/safety-Helmet-Reflective-Jacket \
  --out datasets/helmet_unified --ppe-train 3000

# 2. round-1 pseudo-labels from teachers (DIP helmet model + COCO motorcycles)
python training/pseudo_label.py --round 1

# 3. train student round 1
python training/train.py --profile "$PROFILE" --name unified_r1 --no-export \
  $( [ "$PROFILE" = gpu ] && echo "--epochs 60" || echo "--epochs 10" )

# 4. round-2 pseudo-labels from the student (adds plates, completes the rest)
python training/pseudo_label.py --round 2 --student runs/unified_r1/weights/best.pt

# 5. final training, initialised from round 1
python training/train.py --profile "$PROFILE" --name unified --model runs/unified_r1/weights/best.pt

# 6. honest per-class evaluation + enhancement ablation
python training/evaluate.py --weights models/helmet_unified.pt --ablation \
  --imgsz $( [ "$PROFILE" = cpu ] && echo 416 || echo 640 )

echo "Done. Model: models/helmet_unified.onnx (+ .pt)   Metrics: docs/metrics.md"
