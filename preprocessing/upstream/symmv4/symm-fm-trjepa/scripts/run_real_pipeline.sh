#!/usr/bin/env bash
# CACHE must be a NEW source-grid cache with raw images and verified spatial QC.
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "Usage: $0 CACHE VQ_CHECKPOINT OUTPUT [CONFIG] [DEVICE] [--trusted-legacy]" >&2; exit 2; fi
cd "$(dirname "$0")/.."
CACHE="$1"; CODEC="$2"; OUT="$3"; CFG="${4:-configs/three_phase.yaml}"; DEVICE="${5:-cuda:0}"
LEGACY=()
if [[ "${6:-}" == "--trusted-legacy" ]]; then LEGACY=(--trusted-legacy); elif [[ $# -gt 5 ]]; then echo "Unknown sixth argument" >&2; exit 2; fi
PREV=""
for STAGE in B C D E; do
  ARGS=()
  if [[ -n "$PREV" ]]; then ARGS=(--init "$PREV"); fi
  python -m trjepa.cli train --config "$CFG" --root "$CACHE" --codec "$CODEC" \
    --output "$OUT/$STAGE" --stage "$STAGE" --device "$DEVICE" "${ARGS[@]}" "${LEGACY[@]}"
  PREV="$OUT/$STAGE/best.pt"
done
