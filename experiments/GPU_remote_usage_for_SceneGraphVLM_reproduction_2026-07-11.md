# 远程 GPU / 线上 GPU 服务使用调研报告：以 SceneGraphVLM 复现为例

日期：2026-07-11  
对象：刚开始接触研究生科研复现、还不熟悉远程 GPU 使用流程的同学  
项目：SceneGraphVLM 复现与 baseline 实验

---

## 1. 先说结论

导师说“如果你本地环境不允许做实验，可以调他们的 GPU”，通常不是指把 GPU 插到你电脑上，也不是像普通软件一样点一下就能用。

更准确地说，是：

> 你通过网络登录到实验室或服务器上的一台机器，在那台机器上运行代码。你的电脑只是负责远程操作，真正计算发生在服务器 GPU 上。

可以把它理解成：

```text
你的电脑：键盘、屏幕、写代码、传文件
远程服务器：真正跑模型、占用 GPU、保存实验结果
```

对于 SceneGraphVLM 这个项目，本地 RTX 3080 Ti Laptop 16GB 已经能做小规模 baseline 推理，但如果想更接近论文设置，尤其是：

- vLLM 加速；
- Linux / Docker 环境；
- 更大 batch；
- 跑完整 PVSG / PSG / AG 测试集；
- 做 SFT / LoRA / GRPO 后训练；

就更适合用导师提供的远程 GPU。

---

## 2. 为什么本地电脑不一定适合做完整实验？

你现在本地电脑不是不能跑，而是有几个限制。

当前本机情况：

| 项目 | 本机情况 |
|---|---|
| GPU | RTX 3080 Ti Laptop |
| 显存 | 16GB |
| 系统 | Windows |
| 当前后端 | Transformers |
| 已完成 | SceneGraphVLM checkpoint 小规模推理 |

本机可以做：

- 单张图 / 少量视频帧推理；
- 小规模真实视频 baseline；
- 结果可视化；
- 和 SAMJAM 做定性对比；
- 写报告、做汇报。

但不适合做：

- 大规模测试集评估；
- 论文级 1 秒/图速度复现；
- vLLM 高吞吐推理；
- SFT / GRPO 全量训练；
- 多 GPU 训练。

原因主要有三个：

1. **Windows 对深度学习部署不如 Linux 稳**

   SceneGraphVLM 官方环境更偏 Linux / Docker。

2. **vLLM、FlashAttention、DeepSpeed 更适合 Linux**

   论文中的速度优势依赖这些工具。本地 Windows 很难完全复刻。

3. **16GB 显存够推理，但不够完整训练**

   论文训练使用多 GPU，GRPO 阶段甚至使用 H200 级别硬件。

所以导师说“可以调他们的 GPU”，本质上是给你一个更接近论文环境的平台。

---

## 3. 远程 GPU 到底是什么？

远程 GPU 常见有三种形式。

---

### 3.1 实验室服务器

这是导师最可能说的情况。

形式类似：

```text
一台 Linux 服务器
上面有 NVIDIA GPU
你拿到账号和密码 / SSH key
通过 SSH 登录进去跑代码
```

你需要知道的信息通常有：

```text
服务器地址：例如 10.xxx.xxx.xxx 或 gpu.lab.edu
用户名：例如 yourname
登录方式：密码或 SSH key
GPU 情况：例如 1 张 4090 / 2 张 A6000 / 4 张 A100
系统环境：Ubuntu / CentOS
是否有 Docker
是否有 conda
是否需要排队系统：如 Slurm
```

---

### 3.2 学校 / 课题组计算平台

有些学校不是直接给你一台服务器，而是给你一个网页平台。

你可能会看到：

```text
提交任务
选择 GPU
选择镜像
填写运行命令
等待排队
下载结果
```

这种通常背后有 Slurm / Kubernetes / 容器平台。

优点：

- 环境相对规范；
- 不容易和别人抢资源；
- 任务有记录。

缺点：

- 对新手不如直接 SSH 直观；
- 每次运行需要写提交脚本；
- 调试稍麻烦。

---

### 3.3 商业云 GPU

比如 AutoDL、恒源云、阿里云、腾讯云、RunPod 等。

你租一台带 GPU 的机器，按小时计费。

优点：

- 机器随开随用；
- 可以选 4090、A100 等；
- 通常有现成 PyTorch 镜像。

缺点：

- 要花钱；
- 数据上传下载慢；
- 停机后数据可能要额外保存；
- 和导师 GPU 相比，不一定有必要。

如果导师已经说可以用他们的 GPU，优先用导师提供的资源。

---

## 4. 远程 GPU 的基本工作流

最常见流程是：

```text
1. 拿到服务器账号
2. 从自己电脑 SSH 登录服务器
3. 在服务器上检查 GPU
4. 上传 / 克隆代码
5. 准备 Python / conda / Docker 环境
6. 上传模型 checkpoint 和数据
7. 运行实验
8. 用 tmux / screen 保持任务不中断
9. 查看日志和 GPU 使用情况
10. 下载结果到本地写报告
```

下面结合 SceneGraphVLM 具体讲。

---

## 5. 第一步：需要向导师、服务器管理员或项目协作者确认什么？

你可以直接问下面这些问题。

### 5.1 服务器连接信息

需要问：

```text
服务器 IP 或域名是什么？
我的用户名是什么？
用密码登录还是 SSH key 登录？
是否需要 VPN / 校园网？
SSH 端口是不是 22？
```

例子：

```text
ssh username@server_ip
```

如果不是 22 端口：

```text
ssh -p 端口号 username@server_ip
```

---

### 5.2 GPU 情况

需要问：

```text
服务器是什么 GPU？
每张 GPU 显存多少？
我可以用几张？
有没有使用时间限制？
多人共用时怎么避免冲突？
```

SceneGraphVLM 推荐：

| 任务 | 建议 GPU |
|---|---|
| 小规模推理 | 16GB 以上可尝试 |
| vLLM baseline | 24GB 以上更稳 |
| 大 batch 推理 | 40GB / 80GB 更好 |
| SFT / LoRA | 24GB 起步，多卡更好 |
| GRPO | 多卡高显存，论文用 H200 |

---

### 5.3 环境管理方式

需要问：

```text
服务器上是否已经有 conda？
是否可以用 Docker？
是否有 NVIDIA Docker？
是否已经装好 CUDA / PyTorch？
是否有公共数据盘？
```

SceneGraphVLM 官方更推荐 Docker / Linux 环境，因为官方仓库里明确提到训练和推理环境依赖：

- PyTorch；
- MSwift；
- vLLM；
- FlashAttention；
- DeepSpeed；
- Docker image；
- checkpoints。

---

### 5.4 是否有任务调度系统

需要问：

```text
是直接登录机器跑，还是需要 Slurm 提交任务？
如果是 Slurm，有没有示例 sbatch 脚本？
```

如果是直接登录，通常可以：

```text
python xxx.py
```

如果是 Slurm，通常要写：

```text
sbatch run_scenegraphvlm.sh
```

---

## 6. 第二步：登录服务器后先做什么？

登录后不要马上跑代码，先检查机器。

### 6.1 看 GPU

运行：

```bash
nvidia-smi
```

你会看到：

- GPU 型号；
- 显存总量；
- 当前谁在用；
- 显存占用；
- CUDA driver 版本。

如果发现 GPU 已经满了，不要硬跑。  
要么换卡，要么等别人任务结束，要么向服务器管理员或项目协作者确认排队规则。

---

### 6.2 看系统

运行：

```bash
uname -a
```

看是不是 Linux。

再看 CUDA：

```bash
nvcc --version
```

或者：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## 7. 第三步：怎么把 SceneGraphVLM 放到服务器？

有两种方式。

---

### 7.1 方式 A：服务器直接 git clone

如果服务器能联网：

```bash
git clone https://github.com/markus0440/SceneGraphVLM.git
cd SceneGraphVLM
```

优点：

- 最干净；
- 不用从本地传整个项目；
- 后续方便更新。

---

### 7.2 方式 B：从本地上传

如果服务器不能访问 GitHub，可以从本地传。

常用工具：

- scp；
- rsync；
- WinSCP；
- VSCode Remote SSH。

例如：

```bash
scp -r D:/scene username@server_ip:/home/username/projects/SceneGraphVLM
```

Windows 上更推荐新手用 WinSCP 或 VSCode Remote SSH，界面更直观。

---

## 8. 第四步：怎么处理 checkpoint？

SceneGraphVLM 官方 released checkpoints 包括：

```text
AG
PSG
PVSG
```

本地已经有：

```text
D:\scene\checkpoints\AG
D:\scene\checkpoints\PSG
D:\scene\checkpoints\PVSG
```

每个 checkpoint 约 2.08GB。

如果服务器能联网，建议在服务器重新下载官方 checkpoint。  
如果不能联网，可以从本地上传：

```text
D:\scene\checkpoints.zip
```

或只上传需要的：

```text
D:\scene\checkpoints\PVSG
```

因为 baseline 优先用 PVSG。

服务器上建议目录：

```text
/home/username/projects/SceneGraphVLM/checkpoints/PVSG
```

---

## 9. 第五步：怎么准备环境？

SceneGraphVLM 有两类环境。

### 9.1 简化推理环境

适合先跑 baseline：

```text
Python
PyTorch
transformers
ms-swift
qwen-vl-utils
Pillow
opencv-python
```

这种环境和我们本地 Windows 跑通的方式类似，只是换成 Linux。

优点：

- 容易先跑通；
- 不依赖完整 Docker；
- 出问题少。

缺点：

- 慢；
- 不能复现论文的 vLLM 速度。

---

### 9.2 官方训练 / 高速推理环境

适合接近论文：

```text
Docker
NVIDIA Docker
PyTorch
vLLM
FlashAttention
DeepSpeed
MSwift
```

官方仓库说明中，训练和推理容器大致形式是：

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm \
  --shm-size=16g \
  -v /path/to/SceneGraphVLM:/workspace \
  qwen-grpo:cu130
```

新手不用一上来就完全理解 Docker。  
你可以先把它理解成：

> Docker 是一个打包好的实验环境，里面已经有项目需要的软件。你进去以后，就像进入了一台专门配置好的小系统。

---

## 10. 第六步：在远程 GPU 上怎么跑 SceneGraphVLM baseline？

建议 baseline 先分两步。

---

### 10.1 先跑 smoke test

目的：

> 确认服务器环境能加载 checkpoint，并能生成一个结果。

输入只用 1 张图。

如果 smoke test 成功，再跑更多帧。

---

### 10.2 再跑 PVSG released checkpoint baseline

推荐先用：

```text
checkpoint: checkpoints/PVSG
backend: transformers 或 vLLM
batch_size: 1
```

如果用 vLLM，官方示例接近：

```bash
CUDA_VISIBLE_DEVICES=0 python metrics/qwen-bench/infer/GEN-prompt/infer_swift_gen_prompt.py \
  --model checkpoints/PVSG \
  --test-jsonl datasets/data_playground/PVSG_json/pvsg_psfr_gt_prompt/test.jsonl \
  --output-dir metrics/results/checkpoints-inference/released/PVSG-GEN-prompt \
  --run-name SceneGraphVLM-PVSG-GEN \
  --infer-backend vllm \
  --batch-size 64 \
  --max-new-tokens 2048 \
  --temperature 0.0 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --response-prefix $'<answer>\n' \
  --prev-source model
```

但这个参数适合高显存服务器，不适合本地 16GB。

如果导师给的是 24GB GPU，可以先保守改成：

```bash
--batch-size 1
--max-new-tokens 1024
--gpu-memory-utilization 0.8
```

如果导师给的是 A100 80GB，可以再尝试接近官方：

```bash
--batch-size 32 或 64
--max-new-tokens 2048
```

---

## 11. 第七步：怎么让任务不断线？

远程服务器上最常见的问题是：

> 你电脑 SSH 断了，任务也跟着停了。

所以要用 `tmux` 或 `screen`。

推荐用 tmux。

基本流程：

```bash
tmux new -s scenegraphvlm
```

进入后运行实验。

如果你要退出但不停止任务：

```text
按 Ctrl+B，然后按 D
```

下次回来：

```bash
tmux attach -t scenegraphvlm
```

这对跑长实验非常重要。

---

## 12. 第八步：怎么查看实验是否在跑？

看 GPU：

```bash
nvidia-smi
```

看日志：

```bash
tail -f logs/run.log
```

看输出文件：

```bash
ls metrics/results/...
```

如果你看到显存被占用、GPU utilization 在动、日志持续更新，说明任务在跑。

---

## 13. 第九步：怎么把结果拿回本地？

实验完成后，把结果目录下载回来即可。

常见结果包括：

```text
*.jsonl
metrics.json
rendered frames
videos
logs
```

可以用 scp：

```bash
scp -r username@server_ip:/home/username/projects/SceneGraphVLM/metrics/results ./results
```

或者用 WinSCP 直接拖拽。

对你这个项目，建议拿回：

```text
推理 jsonl
评估 metrics json
可视化图片 / 视频
运行日志
```

这样本地继续写报告和做 PPT。

---

## 14. SceneGraphVLM 项目中，本地和远程 GPU 怎么分工？

我建议这样分工。

### 14.1 本地电脑负责

```text
读论文
整理报告
准备小样本
跑 smoke test
跑少量真实视频帧
做可视化
写 PPT
调试脚本逻辑
```

本地已经能完成这些。

---

### 14.2 远程 GPU 负责

```text
vLLM baseline
大规模 PVSG / PSG / AG 推理
完整 test set 评估
更接近论文速度的实验
LoRA / SFT / GRPO 后训练
多视频批处理
```

这部分更吃 GPU 和 Linux 环境。

---

## 15. 针对本项目，远程 GPU 最小可行计划

如果导师明天就给你一个 GPU 账号，我建议按这个顺序做。

---

### 阶段 1：确认服务器能用

目标：

```text
能登录
能看到 GPU
能运行 Python
```

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 阶段 2：跑 SceneGraphVLM smoke test

目标：

```text
1 张图能出 TOON scene graph
```

使用 PVSG checkpoint。

---

### 阶段 3：跑真实视频小样本

目标：

```text
复现我们本地 real_video_probe，但在远程 GPU 上测试速度
```

记录：

- GPU 型号；
- 显存占用；
- 后端；
- batch size；
- 每帧推理时间；
- 输出是否可解析；
- 平均物体数；
- 平均关系数。

---

### 阶段 4：尝试 vLLM

目标：

```text
看远程 GPU 能否接近论文的 1 秒级推理
```

如果 vLLM 成功，重点比较：

| 项目 | 本地 Windows | 远程 GPU |
|---|---:|---:|
| 后端 | Transformers | vLLM |
| GPU | 3080 Ti Laptop 16GB | 导师 GPU |
| 每帧时间 | 约 8～10 秒 | 待测 |
| batch size | 1 | 可尝试更大 |

---

### 阶段 5：跑标准数据集子集

目标：

```text
不是只跑真实视频，而是跑一小部分 PVSG test jsonl
```

这样更接近论文 baseline。

先跑：

```text
10 条 / 50 条 / 100 条
```

不要一上来跑全量。

---

## 16. 你需要避免的坑

### 16.1 不要在服务器上直接跑超大任务

新手容易一上来：

```text
batch size 64
max_new_tokens 2048
全量 test set
```

然后显存爆掉或者任务卡死。

正确做法：

```text
1 张图 → 10 张图 → 100 张图 → 全量
```

---

### 16.2 不要忘记保存日志

科研复现不是只看结果，还要知道怎么跑出来的。

建议每次保存：

```text
run command
GPU 型号
显存
代码版本
checkpoint
输入数据
输出目录
日志
```

---

### 16.3 不要把本地 Windows 经验完全照搬到服务器

服务器通常是 Linux。

路径不同：

```text
Windows: D:\scene\checkpoints\PVSG
Linux: /home/username/SceneGraphVLM/checkpoints/PVSG
```

命令也会不同。

---

### 16.4 不要抢别人 GPU

如果是多人共用服务器，先看：

```bash
nvidia-smi
```

如果 GPU 满了，先问，不要硬跑。

---

### 16.5 不要把 API key / 密码写进公开代码

如果用 Gemini / OpenAI / Comet ML，不要把 key 写进 GitHub。

用环境变量：

```bash
export COMET_API_KEY=...
```

---

## 17. 向服务器管理员或项目协作者确认的信息

正式开始远程 GPU 实验前，建议确认以下信息：

```text
1. 服务器 IP、用户名、登录方式是什么？是否需要 VPN？
2. GPU 型号和显存是多少？当前账号可以使用几张卡？
3. 是直接 SSH 登录运行，还是需要 Slurm / 平台提交任务？
4. 服务器上是否支持 Docker / NVIDIA Docker？
5. 是否已有 PyTorch、CUDA、conda、vLLM、DeepSpeed 环境？
6. 是否有推荐的数据和模型存放目录？
7. 多人共用 GPU 时有什么使用规范？
8. 是否允许长时间运行全量 PVSG / PSG / AG baseline？
```

建议执行顺序是先跑 1 张图 smoke test，再跑小规模 PVSG / 真实视频 baseline，最后再扩大到全量任务。

---

## 18. 和 SceneGraphVLM 复现的关系总结

本项目中，远程 GPU 的意义不是“替代本地工作”，而是：

> 本地负责快速验证和报告整理，远程 GPU 负责接近论文设置的大规模、高速、标准化实验。

对于 SceneGraphVLM：

- 本地已经证明：released checkpoint 能跑；
- 本地瓶颈：Windows + Transformers，速度慢；
- 远程 GPU 目标：Linux + vLLM，接近论文 baseline；
- 如果要训练：必须远程 GPU，多半还需要多卡。

---

## 19. 最终建议

下一步最合理的路线：

```text
本地：
  保留当前 SceneGraphVLM baseline 结果
  整理真实视频和报告

远程 GPU：
  先跑 smoke test
  再跑 PVSG checkpoint 小样本
  再尝试 vLLM
  最后考虑标准数据集子集评估
```

不要一上来就追求完整论文复现。  
先把 baseline 跑稳，再逐步扩大实验规模。

一句话总结：

> 调导师的 GPU，本质是通过 SSH / 平台登录一台远程 Linux GPU 服务器，在服务器上运行 SceneGraphVLM。你的本地电脑负责操作和整理结果，服务器负责真正计算。对 SceneGraphVLM 来说，远程 GPU 最重要的价值是支持 Linux + vLLM + 更大显存，从而实现比本地 Windows 更接近论文设置的 baseline。

