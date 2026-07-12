# SceneGraphVLM server reproduction scripts

这些脚本用于 Linux GPU 服务器上的 SceneGraphVLM baseline 复现。

推荐顺序：

```bash
cd /path/to/SceneGraphVLM
cp scripts/server/env.example scripts/server/.env
vim scripts/server/.env

bash scripts/server/check_server_env.sh
bash scripts/server/run_smoke_vllm.sh

# smoke test 通过后再跑全量
bash scripts/server/run_full_pvsg_vllm.sh
bash scripts/server/run_full_psg_vllm.sh
bash scripts/server/run_full_ag_vllm.sh
```

注意：

- `run_*_vllm.sh` 使用 `--infer-backend vllm`，用于服务器加速 baseline。
- `run_debug_transformers.sh` 使用 `--infer-backend transformers`，只用于调试，不代表论文加速效果。
- checkpoint 和 dataset 不应提交到 Git，请在服务器单独准备。

