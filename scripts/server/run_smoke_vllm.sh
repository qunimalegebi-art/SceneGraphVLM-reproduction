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
TEST_JSONL="${SMOKE_JSONL:-experiments/smoke_probe/probe.jsonl}"
OUT_DIR="${RESULT_ROOT:-metrics/results/full_reproduction}/smoke"

mkdir -p "$OUT_DIR"

echo "[smoke] model=$MODEL"
echo "[smoke] test_jsonl=$TEST_JSONL"
echo "[smoke] backend=vllm"

python metrics/qwen-bench/infer/GT-prompt/infer_swift_gt_prompt.py \
  --model "$MODEL" \
  --test-jsonl "$TEST_JSONL" \
  --output-dir "$OUT_DIR" \
  --run-name "SceneGraphVLM-PVSG-smoke-vllm" \
  --infer-backend vllm \
  --batch-size 1 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --response-prefix $'<answer>\n' \
  --force

echo "[smoke] output=$OUT_DIR/SceneGraphVLM-PVSG-smoke-vllm.jsonl"

