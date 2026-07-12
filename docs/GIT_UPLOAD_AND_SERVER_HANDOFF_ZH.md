# Git 上传与服务器交接清单

本文档用于把 SceneGraphVLM 复现代码交给师兄/服务器时检查。

---

## 1. Git 里应该包含什么？

应该包含：

```text
代码
推理脚本
评估脚本
服务器运行脚本
数据准备说明
checkpoint 下载说明
本地小样本实验报告
baseline 评估报告
```

本次新增重点文件：

```text
docs/SCENEGRAPHVLM_FULL_REPRODUCTION_ZH.md
docs/GIT_UPLOAD_AND_SERVER_HANDOFF_ZH.md
scripts/server/README.md
scripts/server/env.example
scripts/server/check_server_env.sh
scripts/server/run_smoke_vllm.sh
scripts/server/run_debug_transformers.sh
scripts/server/run_full_pvsg_vllm.sh
scripts/server/run_full_psg_vllm.sh
scripts/server/run_full_ag_vllm.sh
scripts/server/eval_predictions.sh
```

---

## 2. Git 里不应该包含什么？

不要把下面内容直接普通 git push：

```text
checkpoints.zip
checkpoints/AG
checkpoints/PSG
checkpoints/PVSG
*.safetensors
*.pt
原始视频
完整数据集
dumped frames
API key / password / .env
```

这些应该在服务器上单独下载，或用网盘/scp/Git LFS 管理。

---

## 3. 上传前检查命令

在本地运行：

```bash
git status
git log --oneline -3
git ls-files | grep -E "safetensors|checkpoints.zip|\\.pt$|\\.mp4$|\\.avi$|\\.env$"
```

如果最后一条命令有输出，要检查是不是误把大文件或密钥加入 Git。

---

## 4. 服务器 clone 后第一步

师兄 clone 后：

```bash
cd SceneGraphVLM
cp scripts/server/env.example scripts/server/.env
vim scripts/server/.env
```

主要修改：

```text
CHECKPOINT_ROOT
AG_MODEL
PSG_MODEL
PVSG_MODEL
DATA_ROOT
AG_TEST_JSONL
PSG_TEST_JSONL
PVSG_TEST_JSONL
CUDA_VISIBLE_DEVICES
BATCH_SIZE
GPU_MEMORY_UTILIZATION
```

---

## 5. 如何确认不是跑错成慢速版本？

加速 baseline 必须运行：

```bash
bash scripts/server/run_smoke_vllm.sh
bash scripts/server/run_full_pvsg_vllm.sh
```

这些脚本内部使用：

```bash
--infer-backend vllm
```

如果运行的是：

```bash
bash scripts/server/run_debug_transformers.sh
```

那只是调试版，不能代表论文加速结果。

---

## 6. 推荐发给师兄的话

```text
师兄，我会把 SceneGraphVLM 复现代码上传到 Git。

Git 里包含：
1. 本地 baseline 记录；
2. 服务器 vLLM 运行脚本；
3. AG/PSG/PVSG 全量 baseline 脚本；
4. 数据和 checkpoint 准备说明。

Git 里不会放 checkpoint 和完整数据集，这些需要服务器单独下载或我后续传过去。

服务器上建议先跑：
bash scripts/server/check_server_env.sh
bash scripts/server/run_smoke_vllm.sh

确认 vLLM smoke test 成功后，再跑：
bash scripts/server/run_full_pvsg_vllm.sh
bash scripts/server/run_full_psg_vllm.sh
bash scripts/server/run_full_ag_vllm.sh

注意：只有脚本里使用 --infer-backend vllm，才代表尝试论文中的加速推理设置；transformers 版本只是调试 baseline。
```

