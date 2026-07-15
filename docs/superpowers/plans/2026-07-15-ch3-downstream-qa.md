# 第三章多检索路径下游大模型问答实施计划

> 前置规范：[第三章多检索路径下游大模型问答规范](../../experiments/experiments_ch3_downstream_qa_spec.md)

## 目标与完成定义

为 WebQSP / TransferNet 的 SP、普通 Score-Beam、终点感知 Score-Beam、TARRS 提供可复现的
下游大模型 QA 对照，并以无路径为下界。第一阶段先在未训练基座模型上完成五组确定性评测；
随后在一个来自正式第四章主实验的固定 LoRA 上重放相同对照。实现不得改变第四章主矩阵的
训练输入选择规则。

完成后应满足：

- 一个专用编排入口可按层次、条件、演练和断点续跑运行；
- 输出只写入 `data/output/kgqa/ch3_retrieval/.../downstream_qa/`；
- 每一组可追溯到已确认的第三章检索配置、输入 JSONL、模型/adapter 和解码配置；
- WebQSP 全量五组 QA 的汇总和机制解释表可直接用于论文草稿；
- 训练源消融保留为后续独立计划，不在本次实施范围内。

## 实施顺序

### 任务 1：锁定配置契约与无模型测试

**文件**

- 新建：`experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json`
- 新建：`tests/kgqa/test_ch3_downstream_qa.py`

**配置内容**

1. 读取 `confirmed_profiles/transfernet_v1/confirmed_config.json`，并要求 `status=confirmed`；
2. 显式登记五个条件，不从参数扫描目录自动推断；
3. `score_beam` 固定 `beam20_lambda0_eta0`，`terminal_score_beam` 固定
   `beam20_lambda0_eta15`，`tarrs` 固定已发布 `test.jsonl`；
4. 固定 `format=v2`、`path_format=chain`、`entity_repr=name`、模型、解码和路径预算；
5. 第二层 adapter 设为可选配置节；为空时只允许运行零样本层。

**测试覆盖**

1. 拒绝未确认的 profile、未知条件、重复条件、缺失五组或错误的 `lambda/eta` 标签；
2. 拒绝将 `lambda=0, eta>0` 标记为普通 Score-Beam；
3. 检查五份 JSONL 的行数、`question`（兼容 `question_raw`）和 golden 完全对齐；
4. 检查 `no_path` 仅能以明确的 `--no_paths` 运行；
5. 检查同层模型、格式、解码及 adapter 指纹一致；
6. 演练不加载模型、不写预测，也不把已有 `completed` 进度改回运行中。

**验证**

```bash
python -m unittest tests.kgqa.test_ch3_downstream_qa -v
```

### 任务 2：实现输入校验、配置解析与目录构造

**文件**

- 新建：`experiments/ch3_downstream_qa.py`
- 修改：`kgqa/experiments/config.py`
- 修改：`tests/kgqa/test_ch3_downstream_qa.py`

**实现约束**

- 复用 `load_confirmed_config()`、运行时清单和原子进度更新，不复制其实现；
- 为 `ExperimentPaths` 增加 `ch3_downstream_qa_dir(dataset, backbone, config_id)`；
- 对每份输入流式计算 SHA-256，并同时逐行检查 `question`（兼容 `question_raw`）、`golden`、样本数；
- 对路径条件检查每条输入有 `mmr_reason_paths`（兼容 `paths`）字段；不检查或修改其检索内容；
- 仅用配置显式列出的条件，不扫描 `candidates/`，防止把诊断网格误作正式方法组。

**验证**

```bash
python -m unittest tests.kgqa.test_ch3_downstream_qa -v
```

### 任务 3：实现专用编排入口与汇总报告

**文件**

- 新建：`experiments/run_ch3_downstream_qa.py`
- 新建：`experiments/ch3_downstream_report.py`
- 修改：`tests/kgqa/test_experiment_runners.py`

**CLI 设计**

```bash
python -m experiments.run_ch3_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json \
  --layer base_zeroshot \
  --condition all \
  --phase eval \
  --smoke_size 100 \
  --dry_run
```

- `--layer {base_zeroshot,fixed_pfit_adapter}`；
- `--condition {all,no_path,shortest_path,score_beam,terminal_score_beam,tarrs}`；
- `--phase {validate,eval,report,all}`；
- `--smoke_size <n>`：从五组已对齐输入中按 WebQSP hop 共同、确定性抽样；不使用 pfit
  的文件前缀 `--limit` 作为冒烟集；
- `--dry_run`、`--no_progress`、`--progress_interval`、`--log_level` 与现役运行时一致；
- `eval` 通过 `kgqa.pfit.eval_batch` 执行：同一模型和 adapter 只加载一次，但每个条件仍有独立
  `exp_dir`、`run_dir`、清单、进度、预测与汇总；
- `report` 只读取已完成 `summary.json` 和清单，写出 `condition_matrix.json`、中文
  `summary.md`，不加载模型。

固定 adapter 层启动前，编排器必须验证 adapter 位于 `ch4_pfit` 正式目录，读取其训练清单，
并要求其训练输入为当前已确认 profile 的 `train.jsonl`。不满足时给出中文错误，而不是悄悄
复用旧 `data/output/kgqa/webqsp/pfit/` 产物。

**验证**

```bash
python -m experiments.run_ch3_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json \
  --layer base_zeroshot --phase all --dry_run --no_progress
python -m unittest tests.kgqa.test_ch3_downstream_qa tests.kgqa.test_experiment_runners -v
```

### 任务 4：补充中文实验说明和可重复命令

**文件**

- 修改：`experiments/README.md`
- 修改：`docs/experiments/experiments_kgqa_reproducible_layout.md`

**说明内容**

- 五组名称、准确的 `lambda/eta` 含义和比较链路；
- 零样本、固定 adapter 反事实评测、训练源消融三者的区别；
- 前置产物、演练命令、正式运行命令、恢复语义和结果目录；
- 估时记录格式：单组耗时、模型首次加载耗时、五组总耗时，不预填未经实测的数值；
- 当前测试集参数确认带来的诊断性结论限制。

**验证**

```bash
python -m compileall -q experiments kgqa/experiments tests/kgqa
git diff --check
```

### 任务 5：执行第一层零样本全量实验并核对结果

先运行全部输入校验和 100 条冒烟评测。冒烟样本必须由锚定 TARRS 测试集确定性抽取，保留
WebQSP 的 1-hop / 2-hop 分层比例，不能用文件前 100 条替代。确认五组输出、格式和汇总正常后，
执行全量 1,581 条：

```bash
python -m experiments.run_ch3_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json \
  --layer base_zeroshot --phase all --no_progress
```

验收：每组 `progress.json` 完成、预测 1,581 条、输入指纹一致、`no_path` 的运行清单明确记录
`no_paths=true`，报告同时列出 QA 与上游路径指标。

### 任务 6：执行第二层固定 adapter 反事实评测（依赖第四章主实验）

待第四章主实验已用第三章发布 `train.jsonl` 完成一份 adapter 后，更新专用配置中的 adapter
引用，再对五组测试输入运行：

```bash
python -m experiments.run_ch3_downstream_qa \
  --dataset webqsp \
  --config experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json \
  --layer fixed_pfit_adapter --phase all --no_progress
```

本任务不触发任何训练；若 adapter 不存在或来源不合规，保持明确阻塞状态。第二层仅作为
“路径阅读模型的上下文敏感性”结果，不与第四章训练策略消融混排。

## 后续：训练源消融（不在本计划实施）

另立规范与计划后再做：

1. 为 SP 生成 train JSONL，并为四个路径条件固定各自 train JSONL；
2. 采用相同样本数、格式、训练轮数、种子和 QLoRA 超参数建集、训练；
3. 所有 adapter 在一个预注册的固定测试路径条件（建议 TARRS）上评测；
4. 另设“训练源 × 测试路径”小矩阵时，明确总训练次数并报告交互，而不是只比较对角线；
5. 输出进入 `ch4_pfit` 的独立实验 ID，论文定位为第四章训练数据来源消融。

## 提交边界

建议按职责拆分，且不提交 gitignore 的模型、预测与日志产物：

1. `feat(kgqa): 增加第三章下游 QA 对照编排`：配置解析、校验、路径、编排、汇总与测试；
2. `docs(experiments): 说明第三章下游 QA 对照`：本规范、README 与目录说明；
3. 实际结果和论文图表在完整核对后另行提交；不纳入未提交的 ReaRev、历史文档或无关改动。
