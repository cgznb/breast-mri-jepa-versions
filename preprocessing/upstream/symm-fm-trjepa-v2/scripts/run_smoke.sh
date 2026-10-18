#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/smoke}"
export OMP_NUM_THREADS=1
python -m trjepa.cli synthetic --config configs/smoke_v2.yaml --output "$OUT/data"
PREV=""
for STAGE in B C D E; do
  ARGS=()
  if [[ -n "$PREV" ]]; then ARGS=(--init "$PREV"); fi
  python -m trjepa.cli train --config configs/smoke_v2.yaml --root "$OUT/data" \
    --codec "$OUT/data/synthetic_codec.pt" --stage "$STAGE" --output "$OUT/$STAGE" --device cpu "${ARGS[@]}"
  PREV="$OUT/$STAGE/best.pt"
done
python -m trjepa.cli sample --checkpoint "$OUT/E/best.pt" --codec "$OUT/data/synthetic_codec.pt" \
  --root "$OUT/data" --case "$OUT/data/example_source_only.json" --output "$OUT/prediction.npz" --steps 2 --samples 2
python -m trjepa.cli evaluate --checkpoint "$OUT/E/best.pt" --codec "$OUT/data/synthetic_codec.pt" \
  --root "$OUT/data" --output "$OUT/evaluation" --steps 2 --samples 2 --limit 2
