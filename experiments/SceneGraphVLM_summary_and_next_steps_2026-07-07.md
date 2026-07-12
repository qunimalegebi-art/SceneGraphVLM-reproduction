# SceneGraphVLM 本地实验总结与后续推进方向

日期：2026-07-07  
主题：SceneGraphVLM 是否适合作为视频场景图生成中的“关系生成后端”

---

## 1. 这条线我们想验证什么？

本次 SceneGraphVLM 相关实验，核心不是单纯复现论文结果，而是想回答一个更实际的问题：

> SceneGraphVLM 能不能接在 Grounded-SAM-2 / SAMJAM 后面，作为一个专门生成关系的后端？

也就是说，我们希望前端模型负责“看见并框住物体”，例如输出：

- object id
- object class
- bbox
- mask
- track id

然后 SceneGraphVLM 不再自由发现物体，而是在这些已知物体之间生成关系，例如：

```text
hand holding box
box on table
person touching object
```

如果这条路线可行，那么 pipeline 可以变成：

```text
视频帧
  ↓
Grounded-SAM-2 / SAMJAM 负责物体检测、分割、跟踪
  ↓
SceneGraphVLM 负责关系生成
  ↓
视频场景图
```

这也是它相比 Gemini / Qwen2-VL 这类通用 VLM 最值得期待的地方：  
SceneGraphVLM 是专门为 scene graph / relation generation 训练的模型，理论上应该更适合输出结构化关系。

---

## 2. 当前电脑能做到什么？

目前这台电脑已经可以完成 SceneGraphVLM 的本地初步实验。

硬件与环境大致情况：

| 项目 | 情况 |
|---|---|
| GPU | RTX 3080 Ti Laptop |
| 显存 | 16GB |
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| 运行系统 | Windows |
| 使用模型 | SceneGraphVLM PVSG checkpoint |
| checkpoint 大小 | 约 2.21GB |
| 推理方式 | Transformers，本地单卡推理 |

目前已经完成：

- SceneGraphVLM 官方代码本地部署；
- 官方 checkpoint 下载与校验；
- Windows 环境下推理脚本适配；
- 在真实视频帧上跑通 SceneGraphVLM；
- 做了“直接作为关系后端”的约束实验；
- 和 SAMJAM 在同一段真实视频上的结果进行了对比。

所以结论是：

> 这台电脑可以做 SceneGraphVLM 的 smoke test、单视频推理、小规模对比实验和后续轻量 LoRA 尝试；但不适合做大规模全量训练。

---

## 3. 已经完成的 SceneGraphVLM 实验

### 3.1 真实视频帧推理实验

我们使用了和 SAMJAM 测试相同的一段真实视频帧，抽取 4 帧进行 SceneGraphVLM 推理。

实验结果：

| 指标 | 结果 |
|---|---:|
| 成功解析帧数 | 4 / 4 |
| 平均每帧物体数 | 10.00 |
| 平均每帧关系数 | 1.25 |
| 总关系数 | 5 |
| 平均推理时间 | 约 9.01 秒 / 帧 |

生成的关系主要包括：

```text
on × 3
walking-on × 1
holding × 1
```

说明：

- 模型确实可以本地运行；
- 输出格式比较结构化；
- 能生成 TOON / scene graph 风格结果；
- 但真实视频上的关系数量偏少；
- 关系质量还不稳定，存在误判。

典型问题包括：

- 关系过少，平均每帧只有 1 个多关系；
- 大物体框容易主导结果，例如 floor、table；
- 人体、手、物体之间的细粒度关系识别不稳定；
- 出现过类似 `adult holding mat` 这样的不太合理关系；
- 对真实厨房类视频的动作关系捕捉较弱。

这一点和 SAMJAM 刚好相反：

| 方法 | 特点 |
|---|---|
| SAMJAM | 关系数量多，但有噪声，跨帧 ID / 名称不稳定 |
| SceneGraphVLM | 输出更规整，但关系非常稀疏，真实视频泛化不足 |

---

### 3.2 约束式“关系后端”实验

为了验证 SceneGraphVLM 能不能作为 SAMJAM / Grounded-SAM-2 后面的关系生成器，我们做了一个更关键的实验。

实验设定：

不让 SceneGraphVLM 自己自由发现物体，而是给它一个已有物体表，例如：

```text
object_1: hand
object_2: box
object_3: table
...
```

然后要求它只输出这些物体之间的关系：

```text
object_1 holding object_2
object_2 on object_3
```

理想情况是：

- 不新增物体；
- 不修改 object id；
- 不重新命名 object；
- 只生成关系三元组；
- subject 和 object 必须来自给定 object table。

实际结果：

| 指标 | 结果 |
|---|---:|
| 完全遵守“只输出关系”的帧数 | 0 / 4 |
| 使用给定 object id 的有效关系比例 | 0% |
| 出现格式错误的关系 | 有 |
| 平均推理时间 | 约 7.99 秒 / 帧 |

这个结果非常关键。

它说明：

> 当前官方 SceneGraphVLM checkpoint 不能仅通过 prompt 直接变成一个稳定的“关系生成后端”。

它倾向于按照自己训练时的方式重新生成 object table 和 relation table，而不是严格接受外部输入的 object id / bbox / class。

也就是说，它现在更像是一个“端到端场景图生成模型”，而不是一个“可插拔的关系预测模块”。

---

## 4. 当前结论：SceneGraphVLM 能不能直接当后端？

短答案：

> 现在还不能直接当 Grounded-SAM-2 / SAMJAM 的关系后端。

原因不是它完全跑不通，而是它不满足后端模块最重要的要求：  
**必须服从前端给定的 object id 和 object context。**

作为后端，它应该做的是：

```text
输入：已有 object id、class、bbox、mask、track
输出：这些已知 object 之间的 relation
```

但当前模型实际更像：

```text
输入：图片 / 视频帧
输出：自己理解到的 objects + relations
```

这会造成几个问题：

1. **ID 对不上**

   前端说 object_3 是 box，但 SceneGraphVLM 可能自己重新编号或重新生成物体。

2. **物体表不一致**

   Grounded-SAM-2 / SAMJAM 已经检测到的物体，SceneGraphVLM 不一定按这个列表来。

3. **关系无法稳定接回原 pipeline**

   如果 subject / object id 对不上，生成的关系就很难接回 mask、bbox 和 track。

4. **prompt 约束不够**

   即使明确要求“只输出给定对象之间的关系”，当前 checkpoint 也没有很好遵守。

所以当前判断是：

> SceneGraphVLM 适合作为专用关系模型的研究方向，但现阶段不能直接替换 Gemini 或直接接入 SAMJAM。

---

## 5. 它和 Gemini 相比怎么样？

目前不能简单说 SceneGraphVLM 效果比 Gemini 好。

更准确的说法是：

| 维度 | Gemini / Qwen2-VL | SceneGraphVLM |
|---|---|---|
| 通用视觉理解 | 更强 | 较弱 |
| 自然语言描述 | 更强 | 不是重点 |
| prompt 服从能力 | 通常更好 | 当前较弱 |
| 结构化 scene graph 输出 | 需要强 prompt 约束 | 原生更接近 |
| 真实视频泛化 | 更好一些 | 当前不足 |
| 本地可控性 | 依赖 API / 大模型 | 本地可运行 |
| 作为关系专用模型的潜力 | 中等 | 高 |
| 直接作为后端 | 可以尝试 prompt 工程 | 当前不合格 |

所以当前结论是：

> Gemini 更适合做短期 baseline 和通用关系生成；SceneGraphVLM 更适合做“后续训练成专用关系后端”的候选模型。

换句话说：

- 如果目标是这周汇报能展示效果，SAMJAM + Gemini 更稳；
- 如果目标是后续做一个更专业、更可控的视频场景图系统，SceneGraphVLM 值得继续研究；
- 但 SceneGraphVLM 要真正变成后端，需要训练或微调，不是改 prompt 就够。

---

## 6. SceneGraphVLM 后续可以怎么推进？

后续可以分成三个层次推进。

---

### 方向一：继续作为对照模型

这是最稳、成本最低的方向。

把 SceneGraphVLM 作为一个专用模型 baseline，与 SAMJAM / Gemini 进行对比：

```text
同一段视频
  ↓
SAMJAM 输出 objects + relations
  ↓
SceneGraphVLM 输出 objects + relations
  ↓
比较两者的物体数量、关系数量、错误类型、结构化程度
```

可以重点观察：

- 谁的关系更多？
- 谁的关系更准确？
- 谁更容易出现幻觉？
- 谁的格式更稳定？
- 谁对真实视频更泛化？

这个方向适合短期汇报，因为它不需要重新训练模型。

可以形成的结论：

> SceneGraphVLM 在格式上更接近场景图，但真实视频上的关系召回不足；SAMJAM / Gemini 关系更丰富，但噪声和跨帧一致性问题更明显。

---

### 方向二：做“关系后端”适配实验

这是更接近我们最终目标的方向。

目标是让 SceneGraphVLM 从：

```text
image → objects + relations
```

变成：

```text
image + known objects → relations only
```

这一步可以继续尝试 prompt 和输入格式设计，例如：

```text
已知物体：
object_1 = hand, bbox = [...]
object_2 = box, bbox = [...]
object_3 = table, bbox = [...]

要求：
只输出 object_1、object_2、object_3 之间的关系；
不能新增 object；
不能修改 object id；
predicate 从候选词表中选择；
输出 JSON / TOON 格式。
```

但是根据目前实验结果，这个方向只靠 prompt 成功概率不高。

它更适合作为“训练数据构造”的前置工作：

- 设计标准输入格式；
- 设计标准输出格式；
- 确定关系词表；
- 确定 object id 如何传入；
- 确定 bbox / mask / track 信息如何表达。

---

### 方向三：做 relation-only 微调 / LoRA

这是最有可能让 SceneGraphVLM 真正变成后端的方向。

核心思路：

不要让模型再学习完整的：

```text
image → objects + relations
```

而是专门训练它做：

```text
image + object table → relation table
```

训练样本可以长这样：

输入：

```text
图像 / 视频帧

已知 objects:
object_1: hand, bbox=[...]
object_2: cup, bbox=[...]
object_3: table, bbox=[...]

请只生成这些对象之间的关系。
```

输出：

```text
rel_1: object_1 holding object_2
rel_2: object_2 on object_3
```

这样训练后，模型才可能学会：

- 服从外部 object id；
- 不重新生成 object；
- 只预测关系；
- 把关系接回前端检测结果；
- 适合作为 Grounded-SAM-2 / SAMJAM 后端。

这个方向的难度比 prompt 工程高，但科研价值也更高。

---

## 7. 后训练需要什么？

如果后续真的要做 SceneGraphVLM 后训练，大致需要准备以下内容。

### 7.1 数据

需要 relation-only 格式的数据。

可以来自：

- PVSG；
- PSG；
- Action Genome；
- 自己从 SAMJAM / Grounded-SAM-2 生成伪标签；
- Gemini 辅助生成的关系标签；
- 少量人工修正的真实视频样本。

关键不是数据量一开始就很大，而是格式要对。

目标格式应该是：

```text
输入：image/frame + object table
输出：relations only
```

而不是：

```text
输入：image/frame
输出：objects + relations
```

因为我们想训练的是“后端关系生成器”，不是重新训练一个完整的场景图生成模型。

---

### 7.2 关系词表

需要限制 predicate，不然模型会自由发挥。

可以先设计一个小词表，例如：

高价值动作关系：

```text
holding
touching
using
cutting
opening
placing
carrying
```

空间关系：

```text
on
in
inside
under
next to
in front of
behind
```

低价值关系：

```text
near
beside
around
```

训练和评估时可以优先关注高价值关系，尤其是手-物体交互关系。

---

### 7.3 评价指标

后续不能只看“生成得像不像”，最好要有一些量化指标。

可以评估：

| 指标 | 含义 |
|---|---|
| relation precision | 生成的关系有多少是对的 |
| relation recall | 应该生成的关系有多少被找到了 |
| relation F1 | precision 和 recall 的综合 |
| ID compliance | 关系是否只使用给定 object id |
| format compliance | 输出格式是否完全合规 |
| hallucination rate | 是否生成不存在的 object / relation |
| temporal consistency | 跨帧关系是否稳定 |

对于“关系后端”这个任务，最重要的是：

```text
ID compliance
format compliance
relation precision / recall
```

如果 ID 对不上，即使关系看起来合理，也很难接入实际 pipeline。

---

## 8. 推荐的后续路线

我建议不要一上来就做大规模训练，而是分阶段推进。

### 第一阶段：短期展示阶段

目标：把已有结果讲清楚。

可以做：

1. 展示 SceneGraphVLM 已经本地跑通；
2. 展示真实视频上的输出结果；
3. 对比 SAMJAM 和 SceneGraphVLM；
4. 说明 SceneGraphVLM 直接作为后端失败；
5. 提出 relation-only 微调路线。

这一阶段适合本周汇报。

---

### 第二阶段：后端适配阶段

目标：验证输入输出格式。

可以做：

1. 固定 object table 输入格式；
2. 固定 relation table 输出格式；
3. 继续测试不同 prompt；
4. 统计 ID compliance 和 format compliance；
5. 对比 Gemini / Qwen2-VL / SceneGraphVLM 在同一任务上的表现。

这一阶段可以回答：

> 通用 VLM 和专用 SceneGraphVLM，谁更适合作为关系后端？

---

### 第三阶段：轻量微调阶段

目标：让 SceneGraphVLM 真正学习“只生成关系”。

可以做：

1. 构造少量 relation-only 数据；
2. 使用 LoRA 微调；
3. 只训练小 checkpoint；
4. 评估 ID compliance、format compliance、relation precision / recall；
5. 和原始 SceneGraphVLM、Gemini、SAMJAM 进行对比。

这一阶段如果做成，会比较有研究价值。

因为它不是单纯复现，而是把 SceneGraphVLM 改造成一个更适合真实 pipeline 的关系模块。

---

## 9. 目前可以在汇报中怎么说？

可以简洁地说：

> 我们已经在本地跑通了 SceneGraphVLM，并在真实视频帧上完成了初步测试。结果显示，SceneGraphVLM 的输出格式更接近标准场景图，但在真实视频上的关系生成比较稀疏，平均每帧约 1.25 个关系。进一步的约束实验表明，当前官方 checkpoint 不能仅通过 prompt 直接作为 Grounded-SAM-2 / SAMJAM 的关系后端，因为它不能稳定遵守外部给定的 object id 和 object table。因此，SceneGraphVLM 目前更适合作为专用关系模型的候选方向，而不是直接替换 Gemini。后续如果要推进，需要构造 image + object table → relations only 的数据，并进行 LoRA / SFT 微调。

---

## 10. 一句话总结

SceneGraphVLM 这条线已经完成了“本地可运行”和“后端可行性初测”。  
结论是：

> 它有成为专用关系后端的潜力，但当前官方模型不能直接接入 SAMJAM；真正可行的路线是把它改造成 relation-only 模型，再通过 LoRA / SFT 让它学习服从外部 object id 和 bbox 约束。

