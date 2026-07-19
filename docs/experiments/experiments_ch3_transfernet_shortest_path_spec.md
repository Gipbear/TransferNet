# 第三章 TransferNet 最短路径后处理基线规范

> 状态：实现与 WebQSP 全量路径生成已完成。SP 是当前第三章路径主对照的一组；完整实验矩阵见
> [第三章 TransferNet 路径检索论文与实验总计划](experiments_ch3_transfernet_current_plan.md)。

## 1. 目的与定位

本规范定义第三章在 **TransferNet / WebQSP** 上采用的“最短路径后处理”基线。该基线
借鉴 GNN-RAG 的路径构造思路：先由节点打分模型给出候选答案实体，再从问题主题实体到
候选答案实体抽取知识图谱最短路径，最后将路径作为结构化证据输出。

本实验只复现上述**路径构造环节**，不复现 GNN-RAG 的 ReaRev 骨干、稠密子图构造或
语言模型生成流水线。因此论文、图表和产物中统一称为：

> **基于 TransferNet 候选答案的最短路径后处理基线**

可在首次出现时标注“最短路径（SP）”，但不得称为“完整 GNN-RAG 复现”。

## 2. 科学问题与非目标

### 2.1 科学问题

在相同骨干模型、相同得分缓存、相同最大跳数和相同最终路径预算下，模型逐跳得分引导的
路径检索是否比“只连接主题实体与骨干候选答案”的最短路径后处理，得到更高的路径答案
覆盖、更稳定的首条路径质量和更少的证据冗余？

### 2.2 非目标

- 不比较 TransferNet 的原始答案指标；该指标在不同路径构造方法之间固定，应仅记为
  `backbone` 参照。
- 不声称该基线等价于完整 GNN-RAG，也不在本实验中比较 ReaRev。
- 本阶段不执行路径忠实性干预、语言模型问答或路径监督微调。SP 已作为独立输入接入第三章
  下游 QA 对照；该对照的规则由 `experiments_ch3_downstream_qa_spec.md` 定义，不能反向证明
  路径忠实性。
- 不使用关系分数、逐跳实体分数、`eta` 或 MMR 对最短路径进行排序，以保证其是纯粹的
  答案节点后处理基线。

## 3. 输入与公平性约束

| 项目 | 规范 |
|---|---|
| 数据集与骨干 | 第一轮固定为 WebQSP / TransferNet。 |
| 得分缓存 | 复用已确认配置的 `topk500_test/test.pt`，不得重新前向或生成额外候选子图。 |
| 候选答案 | 从每条样本的 `e_score_indices`、`e_score_values` 按最终实体分数降序取前 `candidate_topk=20` 个实体。`prediction` 仅含最高分并列实体，不能作为候选答案来源。 |
| 图结构 | 复用该数据集适配器的 `KGEdgeSource`；WebQSP 使用含反向边的训练图邻接表。 |
| 最大跳数 | 固定为 `len(sample.rel_probs)`，即当前得分缓存可用的最大推理跳数；不得搜索更长路径。 |
| 最终路径预算 | `path_budget=20`，与已确认 TARRS 配置的 `beam_size=20` 一致。 |
| 路径方向 | 仅沿 `KGEdgeSource.neighbors()` 给出的有向边搜索；反向关系由图邻接表显式提供。 |
| 自环 | 默认剔除“路径终点等于起始主题实体”的结果，与现役检索器的 `drop_loopback` 口径一致。 |

## 4. 算法规范

### 4.1 候选答案排序

设最终实体缓存为 \(\{(e_i, q_i)\}\)。先按 \((-q_i, e_i)\) 排序，再保留前
\(N=20\) 个实体。缓存候选少于 20 个时，保留全部；分数相同的实体按数值实体 ID 升序
打破并列。

### 4.2 有界最短路径枚举

对每个主题实体 \(s\) 与候选答案 \(a\)，在有向 KG 中执行宽度优先搜索：

1. 只保留长度不大于 `max_hop` 的路径；
2. 首次到达 \(a\) 的深度 \(d\) 即该实体对的最短长度；仅保留长度为 \(d\) 的路径；
3. 单条路径中不允许重复实体，避免循环路径；
4. 每个 `(topic, candidate)` 最多保留 `max_paths_per_pair=20` 条最短路径；
5. 邻接边按 `(relation_id, tail_id)` 升序访问，保证多条等长最短路径可复现。

当样本有多个主题实体时，分别枚举 `(topic, candidate)` 对；相同的三元组序列只保留一次。

### 4.3 路径排序与截断

最短路径基线不使用逐跳路径评分。全部候选路径按以下键确定性排序：

1. 对应候选答案的最终实体分数降序；
2. 路径长度升序；
3. 候选答案实体 ID 升序；
4. 起始主题实体 ID 升序；
5. 关系 ID 序列、实体 ID 序列的字典序升序。

然后截断为前 `path_budget=20` 条。为复用现役 JSONL 格式，路径的 `log_score` 写为
该路径终点的 \(\log(\max(q_i, 10^{-9}))\)；它仅是可视化和确定性排序的兼容字段，
不得与 TARRS 的路径分数作数值比较。

## 5. 配置、命令与产物契约

### 5.1 配置

在 `experiments/configs/ch3/webqsp_transfernet_v1.json` 中增加：

```json
"shortest_path_baseline": {
  "id": "top20_hop_available",
  "label": "TransferNet 候选答案最短路径后处理",
  "candidate_topk": 20,
  "max_paths_per_pair": 20,
  "path_budget": 20,
  "max_hop_source": "available_steps",
  "drop_loopback": true
}
```

所有字段均为必填；`max_hop_source` 当前只接受 `available_steps`，防止配置写出超过
score cache 可支持范围的跳数。

### 5.2 编排入口

新增第三章阶段：

```bash
python -m experiments.run_ch3 --dataset webqsp --phase shortest_path
```

`--phase all` 在配置存在 `shortest_path_baseline` 时包含该阶段。该阶段只读取既有
`topk500_test` score cache 和 KG 邻接表，不加载 TransferNet checkpoint。

### 5.3 目录与文件

```text
data/output/kgqa/ch3_retrieval/webqsp/transfernet/
└── shortest_path_baselines/transfernet_v1/top20_hop_available/
    ├── test.jsonl
    ├── test_summary.json
    └── test/
        ├── run_manifest.json
        ├── progress.json
        └── logs/{run.log,events.jsonl,console.log}
```

运行清单必须记录：输入 cache 指纹与路径、`candidate_topk`、`max_paths_per_pair`、
`path_budget`、`max_hop_source`、`drop_loopback`、图边来源和排序规则版本。

`test_summary.json` 沿用现役汇总结构：

- `backbone`：TransferNet 原始实体预测，仅作固定参照；
- `path`：最短路径尾实体构成的路径答案指标和多样性指标；
- `n`：样本数。

## 6. 对照与报告规范

WebQSP 全量测试集在相同 `path_budget=20` 下报告四组：

| 组别 | 路径构造 | 目的 |
|---|---|---|
| SP | TransferNet Top-20 候选答案的有界最短路径 | 答案节点后处理基线 |
| 普通 Score-Beam | 现役联合评分，`lambda_val=0`、`eta=0` | 验证逐跳模型分数引导 |
| 终点感知 Score-Beam | 现役联合评分，`lambda_val=0`、`eta=1.0` | 验证终点实体质量建模 |
| TARRS | 现役联合评分，`lambda_val=0.2`、`eta=1.0` | 验证终点感知质量建模与冗余抑制 |

主指标为 `path.answer_hit`；约束指标为 `path.top1_hit`；`relation_jaccard_diversity`、
`tail_diversity`、`relation_coverage` 用于解释覆盖与冗余的变化。路径 F1、Precision、
Recall 为诊断指标，不单独作为方法优劣结论。

论文中只能写“相对基于骨干候选答案的最短路径后处理”，不得泛化为对所有最短路径方法或
完整 GNN-RAG 的结论。

## 7. 验收标准

1. 同一输入连续运行的 JSONL 三元组序列、顺序和 `log_score` 完全一致；
2. 每条输出路径均由 KG 邻接表中的有向边组成，长度不超过可用跳数；
3. 每条路径终点属于该样本最终实体分数 Top-20 候选之一；
4. 所有产物仅落在 `data/output/kgqa/ch3_retrieval/.../shortest_path_baselines/`；
5. 演练不读取 checkpoint、不写入实际评测 JSONL，也不得把已有完成运行的 `progress.json`
   覆盖为 `running`；
6. 汇总中的 `backbone` 在 SP、普通 Score-Beam、终点感知 Score-Beam、TARRS 四组完全相同，
   比较结论只使用 `path`。
