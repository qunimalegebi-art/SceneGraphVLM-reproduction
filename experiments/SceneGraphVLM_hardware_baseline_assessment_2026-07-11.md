# SceneGraphVLM 全量模型硬件支持与 baseline 实现评估

日期：2026-07-11  
任务：根据 SceneGraphVLM 论文与本机实测，评估全量模型硬件支持、显存需求、推理速度，并给出尽快实现 baseline 的方案。

---

## 1. 结论先行

### 1.1 本机能不能跑 SceneGraphVLM？

可以跑。

本机已经完成：

- 官方 SceneGraphVLM 代码部署；
- 官方 released checkpoints 下载；
- AG / PSG / PVSG 三个 checkpoint 解压；
- 三个 checkpoint 的最小 smoke test；
- PVSG checkpoint 在真实视频帧上的 4 帧推理测试。

本机配置：

| 项目 | 本机情况 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| 显存 | 16GB |
| Driver | 566.07 |
| CUDA Compute Capability | 8.6 |
| 系统 | Windows |
| 推理后端 | Transformers |
| vLLM | Windows 原生环境暂不适合作为主路线 |

判断：

> 本机可以支持 SceneGraphVLM released checkpoints 的 baseline 推理，但速度明显慢于论文中的 A100 + vLLM 设置。

---

### 1.2 能不能复现论文里的“约 1 秒/图”？

短期内，本机 Windows 环境不现实。

论文中的速度条件是：

- 单张 NVIDIA A100 80GB；
- vLLM 加速；
- batch size 1 的后端吞吐测试；
- Qwen3.5-0.8B 作为主干模型；
- TOON 紧凑输出格式。

论文报告 Qwen3.5-0.8B 在 vLLM 后端下吞吐约 371 tokens/s，而 Hugging Face 后端只有约 27 tokens/s。

本机当前是：

- RTX 3080 Ti Laptop 16GB；
- Windows；
- Transformers 后端；
- 缺少 vLLM / Flash-linear-attention 快速路径。

因此本机速度更接近“可跑通 baseline”，不是论文级高速部署。

---

### 1.3 本机实测速度

#### 真实视频 4 帧 PVSG checkpoint 推理

| 指标 | 结果 |
|---|---:|
| 帧数 | 4 |
| 解析成功 | 4 / 4 |
| 平均每帧物体数 | 10.00 |
| 平均每帧关系数 | 1.25 |
| 平均推理时间 | 约 9.01 秒 / 帧 |
| 总关系数 | 5 |

关系分布：

```text
on × 3
walking-on × 1
holding × 1
```

#### 三个 released checkpoint smoke test

同一张图像、同一套本地 Transformers 推理环境下：

| Checkpoint | 是否能加载 | 是否能生成 | 单样本生成时间 |
|---|---|---|---:|
| AG | 是 | 是 | 约 5.23 秒 |
| PSG | 是 | 是 | 约 10.15 秒 |
| PVSG | 是 | 是 | 约 10.22 秒 |

说明：

> AG / PSG / PVSG 三个 released checkpoint 均能在 16GB 显存机器上完成单样本推理。

但 smoke test 只验证“能加载、能生成”，不代表达到论文指标。

---

## 2. 论文中的模型与硬件设定

### 2.1 模型结构

SceneGraphVLM 不是传统的：

```text
detector → relation classifier → post-processing
```

而是：

```text
image / video frame → VLM → TOON scene graph text
```

论文核心设计：

1. 使用 TOON 格式表示 scene graph，减少 JSON 里的重复字段和符号；
2. 使用小型 VLM 作为主干；
3. 先 SFT，让模型学会输出 scene graph 格式；
4. 再 GRPO，用 hallucination-aware reward 减少幻觉和无根据关系；
5. 视频任务中可把上一帧 scene graph 作为上下文。

论文明确提到主模型使用：

```text
Qwen3.5-0.8B
```

本地 checkpoint 配置也对应 Qwen3.5 结构：

| 项目 | 本地 PVSG checkpoint 配置 |
|---|---|
| architecture | Qwen3_5ForConditionalGeneration |
| dtype | bfloat16 |
| text hidden size | 1024 |
| text layers | 24 |
| vision layers | 12 |
| weight size | 约 2.21GB |

---

### 2.2 论文推理硬件

论文中的后端 benchmark 设置：

| 项目 | 论文设定 |
|---|---|
| GPU | NVIDIA A100 80GB |
| batch size | 1 |
| 主要后端 | vLLM |
| 主干 | Qwen3.5-0.8B |
| 输出格式 | TOON |

论文中 Qwen3.5-0.8B 的吞吐：

| 后端 | tokens/s |
|---|---:|
| vLLM | 371 |
| Hugging Face | 27 |
| SGLang | 274 |

这说明 SceneGraphVLM 的“快”高度依赖 vLLM 等推理后端，而不仅仅是模型小。

---

### 2.3 论文训练硬件

论文训练分为：

```text
SFT → GRPO
```

训练部分比推理要求高很多。

论文和官方仓库信息显示：

- SFT 脚本按多 GPU 环境设计；
- 官方示例使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`；
- GRPO 实验使用两张 NVIDIA H200；
- 训练环境依赖 Docker / Linux / vLLM / FlashAttention / DeepSpeed / MSwift。

因此：

> 本机不适合做论文级全量 SFT / GRPO 训练复现。

本机最多适合：

- released checkpoint 推理；
- 小规模真实视频 baseline；
- 小样本 LoRA 适配尝试；
- 后处理和关系后端可行性实验。

---

## 3. 本机硬件支持评估

### 3.1 显存是否够？

够做推理。

本机显存：

```text
16GB
```

released checkpoint 单个模型权重：

```text
约 2.21GB
```

已解压 checkpoint：

| Checkpoint | 大小 |
|---|---:|
| AG | 约 2.08GB |
| PSG | 约 2.08GB |
| PVSG | 约 2.08GB |

单模型 batch size 1 推理可以跑。  
但不建议同时加载多个 checkpoint。

建议推理设置：

```text
batch_size = 1
torch_dtype = bfloat16
max_new_tokens = 512 或 1024
image size = 640 × 480 左右
```

---

### 3.2 推理速度评估

论文速度：

```text
约 1 秒 / graph
```

本机实测：

```text
约 8～10 秒 / frame
```

差距原因：

1. 论文使用 A100 80GB，本机是移动端 3080 Ti 16GB；
2. 论文使用 vLLM，本机使用 Transformers；
3. Windows 下 vLLM / FlashAttention / Flash-linear-attention 支持不如 Linux；
4. 本地日志显示 fast path 不可用，回退到 torch implementation；
5. 当前 batch size 只能稳妥设为 1，无法利用大 batch 提升吞吐。

所以本机推理速度结论：

> 可以做 baseline、demo 和小规模真实视频测试；不适合做大规模 benchmark 或 near-real-time 部署。

---

### 3.3 稳定性评估

本机已经完成：

- PVSG 4 帧真实视频推理：成功；
- constrained relation backend probe：成功运行；
- AG / PSG / PVSG 三 checkpoint smoke test：成功。

已知限制：

- Windows 原生 vLLM 不作为稳定路线；
- fast attention 相关库缺失，速度受影响；
- 目前只做小规模推理验证，还未跑完整 PVSG test set；
- 真实视频效果偏稀疏，关系召回不足；
- constrained backend prompt 实验失败，不能直接接 SAMJAM object table。

---

## 4. 当前 baseline 应该怎么实现？

导师要求“尽快实现基线模型”，我建议 baseline 分两层：

---

### 4.1 Baseline A：本机可交付 baseline

目标：

> 尽快证明 SceneGraphVLM released checkpoint 可以在本机真实视频帧上跑通，并产出可视化和统计结果。

使用：

```text
checkpoint: D:\scene\checkpoints\PVSG
backend: transformers
batch_size: 1
input: 抽帧真实视频
output: TOON / triples / visualization
```

已完成结果：

```text
D:\scene\experiments\real_video_probe
D:\scene\experiments\constrained_relation_probe
D:\scene\experiments\full_checkpoint_smoke
```

这个 baseline 的汇报说法：

> 我们已完成 SceneGraphVLM released checkpoint 的本地 baseline 推理。RTX 3080 Ti Laptop 16GB 可支持 AG / PSG / PVSG 三个 checkpoint 单样本推理；PVSG checkpoint 在真实视频 4 帧上解析成功率 100%，平均约 9.01 秒/帧。

---

### 4.2 Baseline B：论文设置 baseline

目标：

> 尽量接近论文设置，跑官方 PVSG test jsonl，使用 vLLM。

推荐环境：

```text
Linux
CUDA
Docker
vLLM
FlashAttention
DeepSpeed
MSwift
GPU ≥ 24GB 更稳，A100 80GB 最接近论文
```

官方仓库给出的 PVSG GEN-prompt 推理参数包括：

```text
--infer-backend vllm
--batch-size 64
--max-new-tokens 2048
--max-model-len 8192
--gpu-memory-utilization 0.9
```

这套参数不适合直接搬到本机 16GB Windows 上。  
如果要在本机试，需要降级为：

```text
--infer-backend transformers
--batch-size 1
--max-new-tokens 512 / 1024
```

---

## 5. 是否需要“全量模型”？

这里要区分两个概念。

### 5.1 全量 released checkpoints

如果“全量模型”指官方 released checkpoints：

```text
AG
PSG
PVSG
```

那么本机已经支持。

三个 checkpoint 都已解压，并且 smoke test 通过。

---

### 5.2 全量论文复现

如果“全量模型”指：

```text
完整数据集 + SFT + GRPO + 官方指标复现
```

那么本机不支持。

原因：

- 缺少 Linux/Docker/vLLM 标准训练环境；
- 显存只有 16GB；
- 论文训练使用多 GPU / H200；
- 数据集准备量大；
- GRPO rollout 成本高。

所以实际建议是：

> 本周目标应定义为“released checkpoint inference baseline”，不要定义为“完整训练复现 baseline”。

---

## 6. 和 SAMJAM / Gemini baseline 的关系

SceneGraphVLM 当前 baseline 的定位不是替代 SAMJAM，而是作为对照：

| 方法 | 当前定位 |
|---|---|
| SAMJAM | 真实视频 pipeline baseline，关系更多，但噪声和跨帧不稳定 |
| SceneGraphVLM | 专用 SGG checkpoint baseline，格式更规整，但真实视频关系偏少 |
| Gemini / Qwen2-VL | prompt baseline，通用理解强，但结构化和稳定性依赖 prompt |

目前 SceneGraphVLM 的关键发现：

1. 本机能跑 released checkpoint；
2. 论文级速度依赖 A100 + vLLM，本机达不到；
3. 真实视频关系输出偏稀疏；
4. 不能仅靠 prompt 直接作为 SAMJAM / Grounded-SAM-2 的关系后端；
5. 如果要做后端，需要 relation-only SFT / LoRA。

---

## 7. 建议向导师汇报的版本

可以这样说：

> 根据论文，SceneGraphVLM 的主模型是基于 Qwen3.5-0.8B 的小型 VLM，通过 TOON 格式、SFT 和 hallucination-aware GRPO 实现场景图生成。论文中的约 1 秒/图速度是在单张 A100 80GB 上使用 vLLM 加速得到的；Qwen3.5-0.8B 在 vLLM 下约 371 tokens/s，而 Hugging Face 后端只有约 27 tokens/s。因此速度优势主要依赖 vLLM 和高端 GPU。  
>   
> 我们本机为 RTX 3080 Ti Laptop 16GB，已经完成 AG / PSG / PVSG 三个 released checkpoint 的加载和最小推理测试，说明本机可以支持 baseline 推理。PVSG checkpoint 在真实视频 4 帧上解析成功率 100%，平均约 9.01 秒/帧。结论是：本机可以实现本地 baseline 和小规模真实视频测试，但无法复现论文中的 A100 + vLLM 近实时速度，也不适合做完整 SFT / GRPO 训练。下一步建议以 PVSG released checkpoint 作为 SceneGraphVLM baseline，跑更多真实视频小样本，并与 SAMJAM 输出进行对比。

---

## 8. 下一步执行建议

优先级从高到低：

1. **固定本地 baseline 脚本**

   使用：

   ```text
   D:\scene\scripts\run_windows_real_video_probe.ps1
   ```

   或整理一个更清楚的：

   ```text
   run_scenegraphvlm_baseline_windows.ps1
   ```

2. **用 PVSG checkpoint 跑 2～3 个真实视频小样本**

   统计：

   - parse success；
   - objects/frame；
   - relations/frame；
   - time/frame；
   - predicate distribution；
   - qualitative errors。

3. **与 SAMJAM 做同视频对比**

   重点比较：

   - 关系数量；
   - 关系质量；
   - 格式稳定性；
   - 是否适合后端。

4. **如果需要论文级速度**

   转 Linux / Docker / vLLM 环境。  
   本机 Windows 不建议继续硬凑 vLLM。

---

## 9. 参考来源

- SceneGraphVLM arXiv paper: https://arxiv.org/abs/2605.13667
- SceneGraphVLM GitHub repository: https://github.com/markus0440/SceneGraphVLM
- 本地 checkpoint：`D:\scene\checkpoints\AG`, `D:\scene\checkpoints\PSG`, `D:\scene\checkpoints\PVSG`
- 本地实测结果：
  - `D:\scene\experiments\real_video_probe\rendered\summary.json`
  - `D:\scene\experiments\constrained_relation_probe\summary.json`
  - `D:\scene\experiments\full_checkpoint_smoke`

