#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f scripts/server/.env ]]; then
  # shellcheck disable=SC1091
  source scripts/server/.env
fi

echo "[check] pwd=$(pwd)"
echo "[check] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[check] nvidia-smi"
nvidia-smi || true

echo "[check] python"
python --version

echo "[check] torch / cuda"
python - <<'PY'
import importlib.util
print("torch installed:", importlib.util.find_spec("torch") is not None)
if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("total memory GB:", round(props.total_memory / 1024**3, 2))
print("vllm installed:", importlib.util.find_spec("vllm") is not None)
print("swift installed:", importlib.util.find_spec("swift") is not None)
PY

echo "[check] checkpoint paths"
for p in "${AG_MODEL:-checkpoints/AG}" "${PSG_MODEL:-checkpoints/PSG}" "${PVSG_MODEL:-checkpoints/PVSG}"; do
  if [[ -d "$p" ]]; then
    echo "  OK  $p"
  else
    echo "  MISS $p"
  fi
done

echo "[check] dataset jsonl paths"
for p in "${AG_TEST_JSONL:-}" "${PSG_TEST_JSONL:-}" "${PVSG_TEST_JSONL:-}"; do
  if [[ -n "$p" && -f "$p" ]]; then
    echo "  OK  $p"
  else
    echo "  MISS ${p:-unset}"
  fi
done

echo "[check] done"

