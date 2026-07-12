#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f scripts/server/.env ]]; then
  # shellcheck disable=SC1091
  source scripts/server/.env
fi

PRED_JSONL="${1:?Usage: bash scripts/server/eval_predictions.sh path/to/pred.jsonl [output_name.json]}"
OUTPUT_NAME="${2:-metrics.json}"
OUT_DIR="${EVAL_ROOT:-metrics/results/full_reproduction_eval}"

mkdir -p "$OUT_DIR"

python metrics/qwen-bench/eval/eval_sgg_metrics_with_qwen.py \
  --pred-jsonl "$PRED_JSONL" \
  --output-dir "$OUT_DIR" \
  --output-name "$OUTPUT_NAME" \
  --iou-thr "${IOU_THR:-0.5}" \
  --batch-size-qwen "${BATCH_SIZE_QWEN:-16}" \
  --max-new-tokens-qwen "${MAX_NEW_TOKENS_QWEN:-128}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"

echo "[eval] output=$OUT_DIR/$OUTPUT_NAME"

