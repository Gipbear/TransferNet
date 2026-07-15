# 第三章多检索路径下游大模型问答规范

## 1. 目的与定位

本规范定义第三章在 **WebQSP / TransferNet** 上的下游大模型问答（QA）对照。目标是回答：
在题目、基础模型、提示词、路径表示、路径预算和确定性解码均相同的条件下，不同路径检索
方法提供的证据，能否提升大模型最终答案质量。

这是一项**检索上下文反事实评测**：保持大模型不变，只替换测试时输入的路径 JSONL。它服务于
第三章“检索证据质量对下游推理的影响”的结论，不替代第四章的路径监督微调主实验。

术语统一如下：

- **无路径**：保留同一题目和 golden，只通过 `--no_paths` 不向模型展示任何检索路径；
- **最短路径（SP）**：基于 TransferNet Top-20 候选答案的最短路径后处理基线；
- **普通 Score-Beam**：`beam=20，lambda=0，eta=0`。不含终点感知，也不含冗余抑制；
- **终点感知 Score-Beam**：`beam=20，lambda=0，eta=1.5`。只加入终点感知质量建模；
- **TARRS**：`beam=20，lambda=0.5，eta=1.5`。在终点感知 Score-Beam 上加入自适应关系冗余抑制。

因此，`lambda=0，eta=1.5` 不得称为“普通 Score-Beam”。

## 2. 科学问题、比较链路与非目标

### 2.1 科学问题

在相同的路径预算 20 和相同的大模型推理设置下，以下逐级改变是否改善最终 QA：

```text
无路径
  → SP（候选答案驱动的结构连接）
  → 普通 Score-Beam（逐跳模型分数引导）
  → 终点感知 Score-Beam（加入终点质量）
  → TARRS（再加入关系冗余抑制）
```

SP 与普通 Score-Beam 不是严格嵌套的同一算法，故二者只作并列基线比较；后两步则构成
“终点感知”“冗余抑制”两个模块的逐级贡献对照。

### 2.2 非目标

- 不将 TransferNet 原始实体预测指标称为本实验的 QA 指标；它是固定的 `backbone` 参照。
- 不在本规范中重新训练 TransferNet、生成新的 score cache 或改动第三章检索算法。
- 不以已有测试集参数扫描结果宣称无偏的最终泛化性能；当前检索参数由 WebQSP 测试集诊断后
  人工确认，下游结果应标为“固定已确认配置下的诊断性对照”。
- 不把固定 LoRA 的测试上下文替换误写成“训练源消融”。后者需要重新建集并训练新的 LoRA，
  是独立的后续实验。
- 不把本实验混入第四章多种训练策略、输出格式或训练数据规模的主实验矩阵。

## 3. 实验层次

### 3.1 第一层：基座模型零样本检索上下文对照（正式优先）

使用同一个未加载 LoRA 的基座模型，运行全部五个条件。由于当前 `kgqa.pfit.eval` 固定关闭
采样（`do_sample=False`），相同输入与环境下无需为生成随机性重复运行；仍应记录软件版本和
输入指纹。

| 条件 ID | 路径来源 | 检索参数或规则 | 是否向模型展示路径 | 作用 |
|---|---|---|---|---|
| `no_path` | 已确认 TARRS 测试 JSONL，仅复用题目与 golden | `--no_paths` | 否 | 衡量无显式证据时的基座 QA |
| `shortest_path` | `shortest_path_baselines/.../top20_hop_available/test.jsonl` | Top-20 候选、可用跳数、预算 20 | 是 | 候选答案最短路径基线 |
| `score_beam` | `candidates/beam20_lambda0_eta0/test.jsonl` | `beam=20，lambda=0，eta=0` | 是 | 逐跳得分引导基线 |
| `terminal_score_beam` | `candidates/beam20_lambda0_eta15/test.jsonl` | `beam=20，lambda=0，eta=1.5` | 是 | 终点感知贡献 |
| `tarrs` | 已发布 `confirmed_profiles/.../test.jsonl` | `beam=20，lambda=0.5，eta=1.5` | 是 | 完整方法 |

所有路径输入必须有相同测试样本集合、顺序、题目和 `golden`；运行前必须进行逐条
`question`（兼容历史字段 `question_raw`）/ `golden` 对齐检查。若任何一个条件缺失或错位，整组不进入正式比较。

### 3.2 第二层：固定路径监督适配器的检索上下文反事实对照（次级）

在第一层完成后，可选用**一个**训练输入、训练超参数和权重均固定的第四章主实验 LoRA，重复
同一五个测试条件。该层的问题是“已学会阅读路径的模型面对不同检索证据时如何变化”，不是
“哪种路径更适合训练”。

适配器只能来自 `data/output/kgqa/ch4_pfit/webqsp/<config_id>/<experiment_id>/seed_<n>/adapter/`，
且其 `manifest.json` 必须证明训练输入来自第三章已确认的 `train.jsonl`。当前旧
`data/output/kgqa/webqsp/pfit/` 下的适配器可做本地冒烟核对，但不进入正式表格，因为其目录和
训练输入不满足本规范的统一产物与来源要求。

### 3.3 第三层：训练源消融（后续独立实验）

训练源消融是指：分别将 SP、普通 Score-Beam、终点感知 Score-Beam、TARRS 的**训练集**路径
转换为 SFT 数据，各自训练独立 LoRA，然后在一个固定、预先指定的测试检索条件上评测。

它确实需要额外训练：至少要生成 SP 的训练路径，并进行 2--4 份 QLoRA 训练。该层会同时改变
训练证据分布和权重，不能与前两层的“只换测试上下文、不重训”结论混写；本轮不实施。

## 4. 固定输入、模型与公平性约束

| 项目 | 第一层固定值 | 第二层固定值 |
|---|---|---|
| 数据集 / 骨干 | WebQSP / TransferNet | 同左 |
| 测试集 | 1,581 条 `qa_test_webqsp_fixed_1581.txt` 对应 JSONL | 同左 |
| 模型 | `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`，无 adapter | 同一基座 + 一个固定 adapter |
| 提示与路径格式 | `format=v2`，`path_format=chain`，`entity_repr=name` | 同左 |
| 路径预算 | 各 JSONL 中既有的前 20 条路径，不再额外截断或补充 | 同左 |
| 解码 | `do_sample=False`，`max_new_tokens=256`，`batch_size=4` | 同左 |
| 其他开关 | `show_score=false`，`noise_paths=0`，不去尾去重、不乱序、不拒答提示 | 同左 |
| 执行方式 | `kgqa.pfit.eval_batch` 本地离线加载一次模型，顺序评测五组输入 | 同左 |

`no_path` 条件仍读取 TARRS JSONL 以复用题目与 golden，但必须传入 `--no_paths`；它不应被称为
“TARRS 无路径版本”。

## 5. 指标与结论边界

下游 QA 的主结果取 `kgqa.pfit.eval` 的答案指标：`hit1`、`hit_any`、`macro_f1`、`micro_f1`、
`exact_match`，并按 WebQSP hop 分组报告。路径指标包括 `answer_hit`、`top1_hit`、Precision、
Recall、F1、边/关系 Jaccard 多样性、尾节点多样性、关系覆盖率与边覆盖率。报告从本次实际送入
模型的 JSONL 重算路径指标，确保冒烟样本与全量实验均和下游 QA 使用同一批样本；无路径条件的
路径指标标为“不适用”。路径与 QA 指标并列展示但不混算。

建议论文表包括：

| 条件 | 上游路径 answer_hit | 上游路径 F1 | 关系多样性 | QA Hit@1 | QA Hit_any | QA Macro-F1 | QA Micro-F1 | EM |
|---|---:|---:|---:|---:|---:|---:|---:|

解释时先陈述路径质量变化，再陈述最终 QA 的绝对值与相对 `score_beam` 的差值。若路径指标提升
但 QA 不升，不得掩盖；应报告为“当前提示词/模型无法充分利用该证据”的限制。

## 6. 目录、配置与运行清单

新增专用配置：`experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json`。它必须显式记录：

- 已确认检索配置文件与其 SHA-256；
- 五个条件的 JSONL 路径、方法标签、检索参数、输入 SHA-256；
- 模型、adapter（可空）、提示格式、解码和路径展示开关；
- 比较层次（`base_zeroshot` 或 `fixed_pfit_adapter`）及 adapter 训练清单指纹；
- 固定测试集样本数和对齐规则版本。

产物统一写为：

```text
data/output/kgqa/ch3_retrieval/webqsp/transfernet/
└── downstream_qa/transfernet_v1/
    ├── base_zeroshot/{smoke_<n>,full}/
    │   ├── no_path/
    │   ├── shortest_path/
    │   ├── score_beam/
    │   ├── terminal_score_beam/
    │   └── tarrs/
    ├── fixed_pfit_adapter/<adapter_id>/       # 第二层可选
    │   └── <同上五个条件>/
    ├── smoke_inputs/smoke_<n>/               # 按 hop 共同抽样的 JSONL
    └── reports/<层次>/{smoke_<n>,full}/
        ├── condition_matrix.json
        └── summary.md
```

每个条件目录均含 `eval/`（`predictions.jsonl`、`summary.json`、pfit 阶段清单）以及统一运行时
`run_manifest.json`、`progress.json`、`logs/run.log`、`logs/events.jsonl`、`logs/console.log`。
`condition_matrix.json` 汇总每个条件的输入与结果指纹，不复制逐条预测。

## 7. 验收标准

1. 五个条件的输入行数均为 1,581，且每一行的题目和 golden 与锚定条件一致；
2. 同一层五个条件的模型、adapter、格式、路径预算、解码、实体映射完全一致，唯一可变项是
   `--no_paths` 与检索 JSONL；
3. 每个完成运行均有完整运行清单、进度、日志、预测和汇总；
4. 零样本层没有加载 adapter；固定 adapter 层五组引用同一个 adapter 指纹；
5. 演练不加载模型、不创建预测、不覆盖已有完成状态；
6. 汇总报告能够追溯到第三章已确认配置和每份 JSONL 的 SHA-256；
7. 训练源消融另建配置、训练目录和表格，不与测试上下文对照合并。
