# 第三章 TransferNet 路径检索论文与实验总计划

> 状态：进行中。本文档是当前第三章 TransferNet 实验与论文取证的唯一执行计划；不修改论文正文
> `docs/chapter3_new.md`。ReaRev 相关设想仅保留为后续扩展，见
> [第三章重写讨论与 ReaRev 后续扩展备忘](../chapter3_revision_discussion.md)。

## 1. 本轮范围与结论边界

本轮只研究 **WebQSP / TransferNet**。目标不是重新训练 TransferNet，也不是复现完整
GNN-RAG，而是在同一骨干、同一 score cache、同一最大跳数和相同 `path_budget=20` 下，检验
模型推理得分引导的路径检索能否相对候选答案最短路径后处理提供更高覆盖、稳定首条质量和更少
关系冗余的证据集合。

本轮可支持的表述是“模型推理得分引导的显式证据路径检索”。不能使用“完整 GNN-RAG 复现”、
“跨骨干通用”或“已证明模型忠实性”。路径忠实性需要充分性/必要性等独立干预实验；当前未实施。

检索参数 `beam=20, lambda=0.2, eta=1.0` 由 WebQSP 测试集上的固定基座模型端到端 QA 对照后人工确认。因此所有正式表格
都应标注为“固定已确认配置下的诊断性对照”，不得将它表述为无偏的最终测试集选参结论。

## 2. 论文中的方法定位与创新点

第三章的主线为：

```text
TransferNet 隐式图传播
  → 逐跳关系/实体得分与终点实体得分导出
  → 跨跳候选路径重建与终点感知质量排序
  → 首项保持质量的关系冗余抑制（TARRS）
  → 面向下游推理的高覆盖、低冗余证据集合
```

论文只保留两个技术创新点：

1. **终点感知的跨跳显式路径重建与质量建模。** 利用 TransferNet 的关系分数、逐跳实体分数和
   最终实体分数，在统一候选池中构造、比较并排序不同跳数的显式路径。
2. **自适应关系冗余抑制。** 在不改变首条路径基础质量的前提下，对已选路径的关系重复施加
   抑制，在固定预算内选择互补证据。

路径级指标体系是验证手段，不作为第三个技术创新。`path.answer_hit` 是主指标，
`path.top1_hit` 是首条质量约束；Precision、Recall、F1 与多样性指标用于诊断覆盖、噪声和冗余。

## 3. 实验矩阵

### 3.1 路径检索主对照

所有组固定 TransferNet、最终实体候选、可用跳数和 20 条路径预算。`backbone` 指标来自
TransferNet 原始实体预测，在四组中相同，只作参照，不能写成路径方法的答案指标。

| 组别 | 路径构造与参数 | 回答的问题 |
|---|---|---|
| SP | TransferNet Top-20 候选答案的有界最短路径后处理 | 图结构连接能提供怎样的路径证据下界？ |
| 普通 Score-Beam | `beam=20, lambda=0, eta=0` | 逐跳模型分数引导相对 SP 的作用。 |
| 终点感知 Score-Beam | `beam=20, lambda=0, eta=1.0` | 终点实体质量建模的作用。 |
| TARRS | `beam=20, lambda=0.2, eta=1.0` | 关系冗余抑制在终点感知排序之上的作用。 |

SP 与 Score-Beam 是不同的候选路径构造策略，作公平并列比较；后三组构成对终点感知与冗余抑制的
逐步模块对照。

### 3.2 逐跳分数组成消融

固定已确认的 beam、阈值和路径预算，比较下列四组：

| 实验 ID | 逐跳排序分数 | `eta` | 目的 |
|---|---|---:|---|
| `joint_eta1` | 关系与实体联合 | 1.0 | 正式基线 |
| `joint_eta0` | 关系与实体联合 | 0 | 移除终点实体融合 |
| `relation_only` | 仅关系分数 | 0 | 检验关系信号 |
| `entity_only` | 仅实体分数 | 0 | 检验实体信号 |

该消融与 3.1 的 `lambda=0` / `lambda=0.2` 对照共同解释方法组件，不能混用为同一张表。

### 3.3 参数敏感性与效率

报告 `beam ∈ {3,5,10,20,50,100}`、`lambda ∈ {0,0.1,0.2,0.3,0.5,0.7,1.0}`、
`eta ∈ {0,0.5,1.0,1.5,2.0}` 对路径覆盖、首条质量、关系多样性和运行时间的影响。由于扫描发生在
测试集，该部分只解释权衡趋势与确认理由，不得作为独立泛化性能证据。

### 3.4 固定大模型的下游 QA 对照

固定基座模型 `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`、`format=v2`、
`path_format=chain`、`entity_repr=name`、`max_new_tokens=256`、`batch_size=4` 和确定性解码，
比较无路径、SP、普通 Score-Beam、终点感知 Score-Beam、TARRS 五组上下文。

零样本层是本轮正式下游实验；固定 LoRA 上下文反事实和训练源消融均依赖第四章正式主实验，
本轮不作为完成条件。下游 QA 用来检验不同证据对固定模型的可利用性，不等价于路径忠实性实验。

## 4. 当前状态与剩余步骤

| 项目 | 状态 | 论文可用性 |
|---|---|---|
| 已确认 QA 优先配置与 train/test 发布 | 已完成 | `beam=20, lambda=0.2, eta=1.0` 已作为第四、五章的固定上游输入 |
| Top-k 饱和性路径评测 | 待按新基线重评 | score cache 可复用；若作为正式路径表，必须以 `lambda=0.2, eta=1.0` 重算路径指标 |
| SP 全量路径基线 | 已完成 | 可进入 3.1 路径主对照 |
| 普通 Score-Beam 全量路径结果 | 已完成 | 可作为 `lambda=0, eta=0` 的对照 |
| 终点感知 Score-Beam 与 TARRS 全量路径结果 | 已完成 | 已统一为 `lambda=0, eta=1` 与 `lambda=0.2, eta=1`，可进入 3.1 主对照 |
| 分数组成消融 | 已完成 | 四组为 `joint_eta1`、`joint_eta0`、`relation_only` 与 `entity_only`；旧 `joint_eta15` 产物已删除 |
| 参数扫描 | 已完成 | 仅用于参数敏感性与诊断，不作为现役比较配置 |
| 五组下游 QA 冒烟 100 条 | 已完成 | 仅验证编排、输入对齐和报告，不进入论文数值表 |
| 五组下游 QA 全量 1,581 条 | 已完成 | 五组均已按正式输入完成，可进入第三章正式 QA 表 |
| ReaRev 适配、跨骨干比较、路径忠实性干预 | 未实施 | 后续扩展，不阻塞本轮论文闭环 |

全量下游 QA 的前置产物为已发布 profile、两份 Score-Beam 候选 JSONL 和 SP 测试 JSONL。云端必须先
同步下列 gitignore 文件，再执行 `--phase validate`：

```text
confirmed_profiles/transfernet_v1/{confirmed_config.json,test.jsonl}
confirmed_profiles/transfernet_v1/candidates/beam20_lambda0_eta0/test.jsonl
confirmed_profiles/transfernet_v1/candidates/beam20_lambda0_eta1/test.jsonl
shortest_path_baselines/transfernet_v1/top20_hop_available/test.jsonl
```

全量 QA 的五组结果均已按当前正式配置核验完成。下一步将路径主对照、分数组成消融、参数趋势和
全量 QA 分别制表或作图，并更新 `docs/chapter3_new.md`。

## 5. 论文表格与结论规则

论文至少分为四张表或图，避免把不同层次的指标混为一体：

1. **路径主对照表**：SP、普通 Score-Beam、终点感知 Score-Beam、TARRS 的
   `answer_hit`、`top1_hit`、关系多样性、尾实体多样性和诊断性 F1；
2. **分数组成消融表**：四种逐跳分数组合，单独报告路径指标；
3. **参数趋势图**：beam、lambda、eta 的覆盖—首条质量—多样性权衡与效率；
4. **下游 QA 表**：五种上下文的 `hit1`、`hit_any`、`macro_f1`、`micro_f1`、EM，并与同批
   输入重算的路径指标并列展示。

若 TARRS 提升路径覆盖或多样性而未提升 QA，必须如实报告为“固定提示与基座模型未充分利用该证据”；
不能以路径指标替代 QA 结论，也不能以 QA 指标反推模型内部推理忠实性。

## 6. 关联规范

- [最短路径后处理基线规范](experiments_ch3_transfernet_shortest_path_spec.md)：SP 的输入、算法和
  公平性契约；
- [多检索路径下游大模型问答规范](experiments_ch3_downstream_qa_spec.md)：五组 QA 上下文对照的
  输入、运行和报告契约；
- `experiments/README.md`：实际运行命令、产物目录、恢复语义和云端前置产物说明。
