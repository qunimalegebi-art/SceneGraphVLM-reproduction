#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f scripts/server/.env ]]; then
  # shellcheck disable=SC1091
  source scripts/server/.env
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"

MODEL="${AG_MODEL:-checkpoints/AG}"
TEST_JSONL="${AG_TEST_JSONL:?Set AG_TEST_JSONL in scripts/server/.env}"
OUT_DIR="${RESULT_ROOT:-metrics/results/full_reproduction}/AG"

mkdir -p "$OUT_DIR"

echo "[AG] model=$MODEL"
echo "[AG] test_jsonl=$TEST_JSONL"
echo "[AG] backend=vllm"

python metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py \
  --model "$MODEL" \
  --test-jsonl "$TEST_JSONL" \
  --output-dir "$OUT_DIR" \
  --run-name "SceneGraphVLM-AG-GT-vllm" \
  --infer-backend vllm \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --temperature 0.0 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --response-prefix $'<answer>\n' \
  --force

echo "[AG] output=$OUT_DIR/SceneGraphVLM-AG-GT-vllm.jsonl"

