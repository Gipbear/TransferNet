# 第四章消融实验说明

## 背景

当前主方法（v2 微调模型）在 WebQSP 测试集上的表现：

| 指标 | 全量 (n=1541) | 路径命中子集 (n=1388) |
|------|-------------|---------------------|
| Hit@1 | 76.44% | 84.87% |
| Macro F1 | 70.72% | 78.52% |
| Exact Match | 59.18% | 65.71% |
| Citation Accuracy | 76.41% | 84.83% |
| Citation Recall | 78.27% | 86.90% |
| Hallucination Rate | 0.10% | 0.04% |
| Format Compliance | 100% | 100% |

训练配置：beam20_lam0.2 检索路径，v2 格式（路径引用 + 答案），shuffle 打乱，含分数，5 epochs。

消融实验的目的是验证：哪些设计选择对性能提升起关键作用。

---

## 实验设计

消融实验分三组，共 7 次训练、15 次评估。

---

### Group A：输出格式消融

**目的**：验证 v2 路径引用（citation）机制相对于其他输出格式的优势。

**固定变量**：训练集 beam20_lam0.2，shuffle，含分数，测试集 beam20_lam0.2

| 配置 | 输出格式 | 训练目标（Assistant 输出示例） |
|------|---------|-------------------------------|
| v1（answer-only） | 仅答案 | `Answer: entity1 \| entity2` |
| **v2（基线）** | 路径引用 + 答案 | `Supporting Paths: 2, 5`<br>`Answer: entity1` |
| v3（JSON） | JSON 结构 | `{"reasoning": ["Path 2"], "answer": ["entity1"]}` |
| v4（CoT） | 自然语言推理链 + 答案 | `[Reasoning] Path 2 supports ...\n[Answer]\nSupporting Paths: 2\nAnswer: entity1` |

v2 是主方法，v1/v3/v4 是对比项。v1 没有 citation 机制，理论上 citation 相关指标（Citation Accuracy/Recall）无法计算。

---

### Group B：训练数据消融

**目的**：验证各训练数据配置选项对性能的影响。

**固定变量**：v2 格式，训练集 beam20_lam0.2，测试集 beam20_lam0.2

| 配置 | 变化点 | 说明 |
|------|-------|------|
| **基线（v2）** | shuffle + 含分数 + 无 distractor 限制 | 复用 Group A 的 v2 结果 |
| no_shuffle | 关闭路径顺序随机打乱 | 验证 shuffle 防止 positional bias 的作用 |
| no_score | 路径字符串不含 `[score=S]` | 验证分数信息对模型的影响（build + eval 均需对齐） |
| dist_0.3 | distractor_ratio=0.3 | 干扰路径占比≤30%，验证降低噪音的效果 |
| dist_0.5 | distractor_ratio=0.5 | 干扰路径占比≤50% |

> **注意**：no_score 配置下，`build_kgcot_dataset.py` 和 `eval_faithfulness.py` 必须同时使用 `--no_score`，确保训练和推理时路径格式一致。

---

### Group C：检索参数消融

**目的**：验证微调后的 v2 模型在不同检索配置下的泛化能力，同时找到与微调模型最匹配的检索参数。

**固定变量**：复用最佳 v2 adapter（`models/webqsp_v2_best/`），仅做推理，无需重新训练。

**扫描 beam_size（固定 lambda=0.2）**：

| 测试集 | beam_size | 路径数 |
|--------|----------|-------|
| beam5_lam0.2 | 5 | 少量路径，精度高 |
| beam10_lam0.2 | 10 | — |
| beam15_lam0.2 | 15 | — |
| beam20_lam0.2 | 20 | **基线** |
| beam30_lam0.2 | 30 | 更多路径，召回高 |

**扫描 lambda（固定 beam=20）**：

| 测试集 | lambda | 多样性 |
|--------|--------|-------|
| beam20_lam0.0 | 0.0 | 无多样性惩罚（贪心） |
| beam20_lam0.2 | 0.2 | **基线** |
| beam20_lam0.5 | 0.5 | 中等多样性 |
| beam20_lam0.7 | 0.7 | 高多样性 |

---

## 文件说明

```
scripts/
  run_ablation.sh              # 消融实验自动化编排脚本
  collect_ablation_results.py  # 结果汇总脚本，输出对比表格和 CSV

llm_infer/
  build_kgcot_dataset.py       # 训练数据构建（支持 --format --no_shuffle --no_score --distractor_ratio）
  train_sft.py                 # QLoRA 微调（LLaMA 3.1 8B, LoRA rank=16）
  eval_faithfulness.py         # 评估（答案准确率 + 忠实度指标）

data/output/WebQSP/
  predict_train.jsonl          # 训练集（TransferNet + MMR 检索结果）
  ablation/                    # 消融实验输出（run_ablation.sh 创建）
    groupA_v1/
      kgcot_train.jsonl        # 构建的训练数据
      beam20_lam0.2_v1_ft_eval.log  # 评估日志
    groupA_v3/ ...
    groupA_v4/ ...
    groupB_noshuffle/ ...
    groupB_noscore/ ...
    groupB_dist0.3/ ...
    groupB_dist0.5/ ...
    groupC/                    # 各检索参数的评估结果
      beam5_lam0.2_v2_ft_eval.log
      beam10_lam0.2_v2_ft_eval.log
      ...
    results.csv                # 汇总表格（collect_ablation_results.py 生成）

models/
  webqsp_v2_best/              # 主方法 v2 adapter（Group A/C 的基线）
  ablation/
    groupA_v1/                 # 各消融配置训练的 adapter
    groupA_v3/
    groupA_v4/
    groupB_noshuffle/
    groupB_noscore/
    groupB_dist0.3/
    groupB_dist0.5/
```

---

## 运行步骤

### 前提条件

1. 确认训练集存在：`data/output/WebQSP/predict_train.jsonl`
2. 确认测试集存在：`data/output/WebQSP/grid_search/paths/beam20_lam0.2.jsonl`
3. 确认基线 adapter 存在：`models/webqsp_v2_best/`（若路径不同，见下方说明）
4. GPU 环境已就绪（建议 24GB+ 显存）

---

### 步骤一：验证解析脚本（可选，但推荐）

用已有的基线评估日志验证结果汇总脚本是否能正确解析：

```bash
python scripts/collect_ablation_results.py \
    --baseline_log data/output/WebQSP/grid_search/paths/beam20_lam0.2_v2_ft_eval.log
```

预期：终端打印包含基线指标的对比表格，CSV 写出到 `data/output/WebQSP/ablation/results.csv`。

---

### 步骤二：验证单实验流程（推荐）

先只跑 Group C（纯 eval，无需训练，约 3~4 小时），确认整个流程正常：

```bash
bash scripts/run_ablation.sh --group C
```

> 若基线 adapter 路径不是 `models/webqsp_v2_best/`，用环境变量覆盖：
> ```bash
> BEST_ADAPTER=models/webqsp_v2 bash scripts/run_ablation.sh --group C
> ```

---

### 步骤三：运行 Group A（输出格式消融，约 4 小时）

```bash
bash scripts/run_ablation.sh --group A
```

此步骤会：
1. 构建 v1/v3/v4 训练数据（各约 1 分钟）
2. 训练 v1/v3/v4 模型（各约 17 分钟）
3. 评估 v1/v2/v3/v4（各约 30 分钟，v2 复用基线 adapter）

---

### 步骤四：运行 Group B（训练数据消融，约 4 小时）

```bash
bash scripts/run_ablation.sh --group B
```

此步骤会：
1. 构建 noshuffle / noscore / dist0.3 / dist0.5 四份训练数据
2. 分别训练四个模型
3. 分别评估

---

### 步骤五：汇总所有结果

```bash
python scripts/collect_ablation_results.py
```

输出：
- 终端打印 Group A/B/C 三张对比表格
- 写出 `data/output/WebQSP/ablation/results.csv`

---

### 一键全量运行（约 10 小时）

```bash
bash scripts/run_ablation.sh
```

---

### 断点续跑

脚本每步均检查输出是否已存在，中断后直接重新执行即可：

- 训练数据跳过条件：`ablation/{config}/kgcot_train.jsonl` 已存在
- 训练跳过条件：`models/ablation/{config}/adapter_config.json` 已存在
- 评估跳过条件：`ablation/{config}/*_eval.log` 已存在

---

## 指标说明

| 指标 | 含义 |
|------|------|
| Hit@1 | 首个预测答案命中率 |
| Macro F1 | 样本级平均 F1（答案集合对比） |
| EM (Exact Match) | 预测答案集合与真实答案完全一致的比例 |
| Citation Accuracy | 被引用路径中确实包含答案的比例 |
| Citation Recall | 含答案路径被模型引用的比例 |
| Hallucination Rate | 预测答案不在任何输入路径中的比例 |
| Format Compliance | 输出格式合规率（能被正确解析） |

> v1（answer-only）不输出路径引用，Citation Accuracy/Recall 无意义，表格中显示 `--`。
