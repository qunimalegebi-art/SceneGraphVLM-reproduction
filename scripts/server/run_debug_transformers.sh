#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f scripts/server/.env ]]; then
  # shellcheck disable=SC1091
  source scripts/server/.env
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

MODEL="${PVSG_MODEL:-checkpoints/PVSG}"
TEST_JSONL="${DEBUG_JSONL:-experiments/smoke_probe/probe.jsonl}"
OUT_DIR="${RESULT_ROOT:-metrics/results/full_reproduction}/debug_transformers"

mkdir -p "$OUT_DIR"

echo "[debug] This uses transformers backend. It is for debugging only, not paper-speed vLLM."

python metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py \
  --model "$MODEL" \
  --test-jsonl "$TEST_JSONL" \
  --output-dir "$OUT_DIR" \
  --run-name "SceneGraphVLM-PVSG-debug-transformers" \
  --infer-backend transformers \
  --batch-size 1 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --torch-dtype bfloat16 \
  --response-prefix $'<answer>\n' \
  --force

