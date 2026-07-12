# SceneGraphVLM 论文全量复现说明（服务器 GPU 版）

本文档用于提交到 Git，给远程 GPU 服务器上的师兄/协作者使用。目标是把 SceneGraphVLM 的复现流程讲清楚：需要哪些数据、哪些 checkpoint、如何确认是否使用 vLLM 加速、如何从 smoke test 逐步跑到全量 baseline。

---

## 0. 重要说明

本仓库提供两类运行方式：

1. **本地/调试 baseline**
   - 后端：`transformers`
   - 适合：Windows、本地小样本、流程验证
   - 特点：慢，但容易跑通

2. **服务器/论文加速 baseline**
   - 后端：`vllm`
   - 适合：Linux GPU 服务器
   - 特点：更接近论文设置

判断是否使用了加速后端，不能只看代码是否 clone 成功，而要看运行命令里是否包含：

```bash
--infer-backend vllm
```

如果日志显示：

```bash
--infer-backend transformers
```

那就是调试版，不是论文中的 vLLM 加速推理。

---

## 1. 论文级复现分成哪几层？

这里的“全量复现”建议分层理解。

### 1.1 Released checkpoint baseline

使用作者已经发布的 checkpoint，在标准数据集 test split 上跑推理和评估。

这是最现实、最优先的目标。

需要：

- SceneGraphVLM 代码；
- AG / PSG / PVSG released checkpoints；
- AG / PSG / PVSG 数据；
- Linux GPU 环境；
- vLLM 推理后端。

### 1.2 完整训练复现

从头或从 base VLM 开始，复现：

```text
SFT → hallucination-aware GRPO
```

这一步需要多卡、高显存、完整训练数据和更长时间。当前建议先不作为第一阶段目标。

---

## 2. 数据与 checkpoint 的区别

不要把 dataset 和 checkpoint 混在一起。

| 类型 | 作用 | 例子 |
|---|---|---|
| dataset | 测试/训练输入，是“题目” | PVSG、PSG、AG |
| checkpoint | 已训练模型权重，是“会做题的模型” | checkpoints/PVSG、checkpoints/PSG、checkpoints/AG |

全量 baseline 至少需要：

```text
checkpoints/AG
checkpoints/PSG
checkpoints/PVSG
datasets/...
```

---

## 3. 数据获取入口

### 3.1 PVSG

官方仓库：

https://github.com/LilyDaytoy/OpenPVSG

README 中有 dataset 下载链接。数据包括：

```text
Ego4D videos / masks
EpicKitchen videos / masks
VidOR videos / masks
pvsg.json
```

官方目录示例：

```text
data/
  ego4d/
    frames/
    masks/
    videos/
  epic_kitchen/
    frames/
    masks/
    videos/
  vidor/
    frames/
    masks/
    videos/
  pvsg.json
```

### 3.2 PSG

官方仓库：

https://github.com/Jingkang50/OpenPSG

README 中有 full dataset 下载链接。通常需要：

```text
COCO images / panoptic annotations
PSG annotations
psg_train_val.json
psg_val_test.json
```

### 3.3 AG / Action Genome

官方仓库：

https://github.com/JingweiJ/ActionGenome

需要两部分：

```text
Charades videos，480p 版本
Action Genome annotations
```

官方说明中还需要从视频 dump frames，生成的 frames 约 74GB。

---

## 4. SceneGraphVLM checkpoint

官方 SceneGraphVLM 仓库：

https://github.com/markus0440/SceneGraphVLM

Released checkpoints 包括：

```text
checkpoints/AG
checkpoints/PSG
checkpoints/PVSG
```

本地已验证：

| checkpoint | 本机是否能加载 | smoke test |
|---|---|---|
| AG | 是 | 通过 |
| PSG | 是 | 通过 |
| PVSG | 是 | 通过 |

注意：checkpoint 不应直接放进普通 Git 仓库。应通过服务器下载、网盘、scp 或 Git LFS 管理。

---

## 5. 推荐服务器目录结构

建议师兄在服务器上使用类似目录：

```text
/home/<user>/projects/SceneGraphVLM/
  checkpoints/
    AG/
    PSG/
    PVSG/
  datasets/
    data_playground/
    annotations/
    frames/
  metrics/
  scripts/
  docs/
```

如果 checkpoint 或 dataset 放在别处，也可以通过脚本里的环境变量指定。

---

## 6. 服务器环境要求

最低建议：

```text
Linux
NVIDIA GPU
CUDA 可用
Python / conda
PyTorch
transformers
ms-swift
qwen-vl-utils
```

如果要接近论文速度，建议：

```text
vLLM
FlashAttention / flash-linear-attention
较大显存 GPU，24GB 起步，40GB/80GB 更稳
```

论文速度条件更接近：

```text
A100 80GB + vLLM
```

本地 Windows + Transformers 不代表论文加速效果。

---

## 7. 推荐执行顺序

不要一上来直接跑全量。推荐顺序：

```text
1 张图 smoke test
→ 10 条样本
→ 100 条样本
→ 单数据集全量
→ AG/PSG/PVSG 全部 checkpoint 全量
```

这样能及时发现环境、路径、显存、格式问题。

---

## 8. 服务器脚本

本仓库新增服务器脚本目录：

```text
scripts/server/
```

主要脚本：

```text
check_server_env.sh
run_smoke_vllm.sh
run_full_pvsg_vllm.sh
run_full_psg_vllm.sh
run_full_ag_vllm.sh
run_debug_transformers.sh
eval_predictions.sh
env.example
```

使用方式：

```bash
cd /home/<user>/projects/SceneGraphVLM
cp scripts/server/env.example scripts/server/.env
vim scripts/server/.env
bash scripts/server/check_server_env.sh
bash scripts/server/run_smoke_vllm.sh
```

确认 smoke test 成功后再跑全量。

---

## 9. batch size 建议

根据显存从小到大调：

| GPU 显存 | 建议 batch size |
|---|---:|
| 16GB | 1 |
| 24GB | 1～4 |
| 40GB | 8～16 |
| 80GB | 32～64 |

如果显存爆掉，优先降低：

```bash
BATCH_SIZE
MAX_NEW_TOKENS
GPU_MEMORY_UTILIZATION
```

---

## 10. 输出结果

建议输出统一放在：

```text
metrics/results/full_reproduction/
```

例如：

```text
metrics/results/full_reproduction/PVSG/SceneGraphVLM-PVSG-GEN.jsonl
metrics/results/full_reproduction/PSG/SceneGraphVLM-PSG-GT.jsonl
metrics/results/full_reproduction/AG/SceneGraphVLM-AG-GT.jsonl
```

评估结果放在：

```text
metrics/results/full_reproduction_eval/
```

---

## 11. 如何确认是否跑的是“加速后效果”

需要同时满足：

1. 服务器是 Linux GPU 环境；
2. 已安装 vLLM；
3. 运行脚本使用：

```bash
--infer-backend vllm
```

4. 日志中能看到 vLLM backend 初始化；
5. 推理速度明显快于本地 Transformers。

仅仅：

```text
git clone 代码
下载数据
下载 checkpoint
```

不能说明已经使用了加速后端。

---

## 12. 给师兄的最小执行说明

可以直接发：

```text
师兄，代码里我会放两类脚本：

1. scripts/server/run_debug_transformers.sh
   用于服务器上先调通流程，速度慢。

2. scripts/server/run_*_vllm.sh
   用于服务器 GPU 上跑 vLLM 加速 baseline。

只有运行命令里包含 --infer-backend vllm，才代表尝试论文加速设置。

建议先：
bash scripts/server/check_server_env.sh
bash scripts/server/run_smoke_vllm.sh

确认没问题后，再跑：
bash scripts/server/run_full_pvsg_vllm.sh
bash scripts/server/run_full_psg_vllm.sh
bash scripts/server/run_full_ag_vllm.sh
```

---

## 13. 当前本地已经完成的证明

本地已完成：

- AG / PSG / PVSG checkpoint smoke test；
- PVSG checkpoint 真实视频 4 帧推理；
- constrained relation backend probe；
- 硬件支持评估报告；
- 远程 GPU 使用说明。

相关文件：

```text
experiments/SceneGraphVLM_hardware_baseline_assessment_2026-07-11.md
experiments/GPU_remote_usage_for_SceneGraphVLM_reproduction_2026-07-11.md
experiments/full_checkpoint_smoke/
experiments/real_video_probe/
experiments/constrained_relation_probe/
```

