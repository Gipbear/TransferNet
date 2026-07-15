# 第三章 TransferNet 最短路径后处理基线实施计划

> 前置规范：[第三章 TransferNet 最短路径后处理基线规范](../../experiments/experiments_ch3_transfernet_shortest_path_spec.md)

## 目标

为 WebQSP / TransferNet 增加“候选答案最短路径后处理”实验基线，并在统一第三章编排下
生成与 Score-Beam、TARRS 完全同口径的路径指标。该实现只复现 GNN-RAG 的最短路径构造
思想，不复现完整 GNN-RAG。

## 完成定义

- 可通过 `python -m experiments.run_ch3 --dataset webqsp --phase shortest_path` 运行；
- 产物只写入 `data/output/kgqa/ch3_retrieval/webqsp/transfernet/shortest_path_baselines/`；
- 输出路径、汇总结构和运行时文件符合规范；
- 单元测试锁定最短路径、确定性、候选边界和编排演练；
- WebQSP 全量运行后可与既有 `Score-Beam(λ=0)`、TARRS 逐项比较。

## 实施顺序

### 任务 1：先写无模型回归测试

**文件**

- 新建：`tests/kgqa/test_shortest_path_baseline.py`
- 修改：`tests/kgqa/test_experiment_runners.py`

**测试覆盖**

1. 从最终实体分数稳定取 Top-N；分数并列时按实体 ID 排序；
2. 有向 BFS 只返回最大跳数内的最短路径，且单条路径无重复实体；
3. 同一 `(topic, candidate)` 的等长多路径按 `(relation_id, tail_id)` 顺序稳定返回；
4. 多主题、多候选时三元组序列去重，`max_paths_per_pair` 与 `path_budget` 生效；
5. 路径终点必须属于 Top-N 最终实体候选，不读取关系分数、逐跳实体分数、`eta` 或 MMR；
6. 编排演练只产生计划性运行清单，且不会将既有 `completed` 进度覆写为 `running`。

**验证**

```bash
python -m unittest tests.kgqa.test_shortest_path_baseline tests.kgqa.test_experiment_runners -v
```

### 任务 2：实现最短路径基线内核

**文件**

- 新建：`kgqa/retrieve/shortest_path.py`

**公共接口**

```python
@dataclass(frozen=True)
class ShortestPathParams:
    candidate_topk: int = 20
    max_paths_per_pair: int = 20
    path_budget: int = 20
    drop_loopback: bool = True

def retrieve_shortest_paths_one(
    sample: SampleScore,
    edge_source: KGEdgeSource,
    id2ent: dict[int, str],
    id2rel: dict[int, str],
    *,
    params: ShortestPathParams,
) -> RetrieveResult: ...
```

**实现约束**

- 最大跳数从 `len(sample.rel_probs)` 推导，不接受外部数值覆盖；
- 只消费 `topic_ids`、`e_score_indices`、`e_score_values` 和 `KGEdgeSource`；
- 候选排序、BFS、去重、截断、`log_score` 序列化均严格遵循规范；
- 复用 `engine.build_prediction()`、`engine.path_to_triples()`，避免复制骨干预测与路径 JSON
  序列化口径；
- 不修改现役 TARRS 引擎公式、`RetrieveParams` 或 score cache 格式。

**验证**

```bash
python -m unittest tests.kgqa.test_shortest_path_baseline -v
```

### 任务 3：增加独立 CLI 与统一汇总复用

**文件**

- 新建：`kgqa/retrieve/cli/shortest_path.py`
- 修改：`kgqa/retrieve/cli/eval.py`
- 修改：`tests/kgqa/test_cli.py`

**设计**

- 新 CLI 接受 `--dataset`、`--backbone`、`--cache`、`--input_dir`、`--candidate_topk`、
  `--max_paths_per_pair`、`--path_budget`、`--output`、`--summary` 与统一运行时参数；
- 将现有私有 `_evaluate_results()` 提升为语义明确的公共汇总函数，供普通检索评测和
  最短路径 CLI 共用，确保 `backbone` / `path` 汇总口径完全相同；
- CLI 不接受 `beam_size`、`lambda_val`、`eta`、`step_score_mode`，避免错误暗示最短路径
  使用了 TARRS 的评分机制；
- CLI 的运行清单写明 `method=shortest_path_postprocess` 与规范中的全部参数。

**验证**

```bash
python -m unittest tests.kgqa.test_cli tests.kgqa.test_shortest_path_baseline -v
python -m kgqa.retrieve.cli.shortest_path --help
```

### 任务 4：接入第三章配置、路径和编排

**文件**

- 修改：`kgqa/experiments/config.py`
- 修改：`experiments/configs/ch3/webqsp_transfernet_v1.json`
- 修改：`experiments/run_ch3.py`
- 修改：`tests/kgqa/test_experiment_runners.py`

**设计**

- 增加 `ExperimentPaths.ch3_shortest_path_dir()`；
- 配置新增 `shortest_path_baseline`，并严格校验字段和合法取值；
- `run_ch3` 新增 `--phase shortest_path`；配置存在该节时，`--phase all` 一并执行；
- 只允许离线 score cache，默认读取 `topk{config.topk}_{selection_split}`；
- 单独阶段和 `all` 均支持 `--dry_run`，且演练不得污染真实运行目录或覆盖完成状态。

**验证**

```bash
python -m experiments.run_ch3 --dataset webqsp --phase shortest_path --dry_run --no_progress
python -m unittest tests.kgqa.test_experiment_runners -v
```

### 任务 5：补充中文实验说明并做全量验证

**文件**

- 修改：`experiments/README.md`
- 修改：`docs/experiments/experiments_kgqa_reproducible_layout.md`

**说明内容**

- 基线的准确名称、输入、输出、候选答案口径和公平性约束；
- 与 Score-Beam、TARRS 的比较表及 `backbone` / `path` / 后续第三章 QA 的指标边界；
- 运行命令、预计耗时、恢复语义和产物路径；
- 明确该实验不是完整 GNN-RAG 和不支撑忠实性结论。

**验证**

```bash
python -m unittest \
  tests.kgqa.test_shortest_path_baseline \
  tests.kgqa.test_cli \
  tests.kgqa.test_engine \
  tests.kgqa.test_experiment_runners -v
python -m compileall -q kgqa/retrieve experiments tests/kgqa
git diff --check
```

### 任务 6：执行正式实验与比较（代码合入后）

```bash
python -m experiments.run_ch3 --dataset webqsp --phase shortest_path
```

核对：

1. `test.jsonl` 行数为 1,581；
2. `progress.json` 为 `completed`；
3. `test_summary.json` 的 `backbone` 与已发布 TARRS 结果一致；
4. 将 SP、`beam20_lambda0_eta15`、`beam20_lambda05_eta15` 的 `path` 指标汇总进同一张论文表；
5. 只有完成该对照后，才在论文中讨论“逐跳得分引导相对最短路径后处理”的效果。

## 提交边界

实施时按职责拆分，且不纳入当前工作区的 ReaRev、骨干模型默认值、历史计划和论文文件改动：

1. `feat(kgqa): 增加最短路径后处理基线`：内核、CLI、编排、配置与无模型测试；
2. `docs(experiments): 说明第三章最短路径基线`：实验说明与产物约定；
3. 正式实验结果为 gitignore 产物，不提交；论文表格或分析结论另行审核后单独提交。
