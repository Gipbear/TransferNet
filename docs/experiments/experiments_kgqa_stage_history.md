# KGQA 阶段迁移与验证历史档案（2026-07-09 至 2026-07-12）

> 文档性质：历史工程迁移与验证记录，不是现役实验规范，也不能直接作为论文正式结果表的来源。
> 现役 Ch3、Ch4、Ch5 的目录与产物规则见
> [KGQA 三章可复现实验与产物约定](experiments_kgqa_reproducible_layout.md)、`experiments/README.md`
> 及各章节专用规范。这里保留旧产物路径、验证数字、迁移问题与提交证据，供回溯与论文数字核对。

## 目录

1. [Stage1：统一检索框架全量验证](#stage1)
2. [Stage2：PFIT 迁移与验证](#stage2)
3. [Stage3：checked-batch agent 迁移与验证](#stage3)

<a id="stage1"></a>
## Stage1：统一检索框架全量验证

**目的。** 在不重训骨干模型的前提下，完成 WebQSP、MetaQA 和 CWQ 的 score cache、离线检索与
评测链路验证。三数据集使用统一检索内核；下表数字是当时的工程保真记录，输出目录均为历史路径且
已被现役 `data/output/kgqa/` 约定取代。

| 数据集 | checkpoint 与测试集 | 结果摘要 | 路径结果摘要 | 验证结论 |
|---|---|---|---|---|
| WebQSP（2026-07-09） | `WebQSP_run_20260518_2241/model-49-0.7154.pt`；1,581 条 | Hit@1 0.7116，HitAny 0.7938，Macro-F1 0.5799，Micro-F1 0.2587，EM 0.3858 | answer_hit 0.9576，top1_hit 0.7312，P/R/F1=0.2392/0.9223/0.3145 | 与训练报告 acc 0.7154 的差为 0.004；离线回归与 online/offline parity 通过。 |
| MetaQA（2026-07-10） | `MetaQA_KB/model_epoch-6_acc-0.9937.pt`；39,093 条 | overall Hit@1 0.9938，HitAny 0.9991，Macro-F1 0.7947，Micro-F1 0.5681，EM 0.6171 | answer_hit 0.9999，top1_hit 0.9940，P/R/F1=0.5512/0.9812/0.6480 | 与训练 acc 0.9937 对齐；1/2/3-hop 切片均通过 sanity。 |
| CWQ（2026-07-11） | `CWQ/model-29-0.4206.pt`；3,531 条 | Hit@1 0.4084，HitAny 0.4656，Macro-F1 0.3835，Micro-F1 0.3433，EM 0.3121 | answer_hit 0.6907，top1_hit 0.4123，P/R/F1=0.0959/0.6746/0.1423 | 与训练 acc 0.4206 的差为 0.012；逐样本子图的 online/offline parity 通过。 |

### 共用口径与历史产物

- 流程均为 `dump_scores` 生成缓存后，以 offline backend 评测；检索内核是当时的统一
  `tail_blend` / `LogNorm` / MMR 实现。
- WebQSP 与 CWQ 使用 MID 口径；MetaQA 使用实体名口径并按 hop 分组。
- 历史缓存与汇总位于 `data/output/WebQSP/`、`data/output/MetaQA_KB/`、`data/output/CWQ/`，
  均为 gitignore 产物；不能与现役第三章的已确认 profile 混用。

### 数据集差异与已修复问题

- **WebQSP：** 原内核回归由 `tests/kgqa/test_webqsp_regression.py` 锁定；本次验证是统一框架的
  接口基准，合并记录为 PR #2（`245ead8`）。
- **MetaQA：** 首次全量结果为零，根因是 `gold_key="name"` 未将整数 gold id 经 `id2ent`
  还原实体名；修复并增加端到端 `hit1>0` 断言（`6c548d9`）。
- **CWQ：** 每个样本使用自身 `triples` 构造邻接；后端从初始化期全局边源改为按样本调用
  `adapter.kg_edge_source(sample)`。全量 dump 约 20 分钟，缓存约 554 MB；`cls.*` checkpoint
  unexpected key 可由 `strict=False` 正常忽略。

<a id="stage2"></a>
## Stage2：PFIT 迁移与验证

**目的。** 将 Ch4 从只读 legacy `llm_infer/` 迁移至数据集无关的 `kgqa/pfit/`：
build → QLoRA train → eval。该阶段保留旧产物路径用于 parity 与历史数字核对；2026-07-13 后的
新第四章实验由 `experiments/` 编排，历史目录不再作为新实验输出目标。

### 历史数据准备与 smoke

| 数据集 | 历史训练输入 | 检索配置与产物 | smoke 结论 |
|---|---|---|---|
| WebQSP | train 2,996 条；test 1,581 条 | `WebQSP_run_20260518_2241/model-49-0.7154.pt`，beam=20、lambda=0.2；`scores/` 与 `retrieve/` | `webqsp_main_smoke100`：loss 0.136，Hit@1 0.74，HitAny 0.87，Macro-F1 0.62，EM 0.38。 |
| MetaQA | 分层 20K：5,837/7,227/6,936；test 39,093 条 | `MetaQA_KB/model_epoch-6_acc-0.9937.pt`，beam=20、lambda=0.2；`subsets/`、`scores/` 与 `retrieve/` | `metaqa_main_smoke100`：loss 0.089，Hit@1/HitAny 0.96，Macro-F1 0.91，EM 0.82。 |

历史全量 `webqsp_main`（单 run）结果为：Hit@1 85.83、Hits 89.44、Macro-F1 77.91、EM 63.63、
Cit-P 83.35、Cit-R 86.50、Hallucination Rate 0.14。它用于当时与论文表 4-9 的 parity 门槛核对，
不是当前第四章多种子正式结果。

### 当时注册实验、问题与回归

- 历史注册项为 WebQSP/MetaQA 各四类：main、`spot_nl`、base zero-shot、no-path；当时仅
  `webqsp_main` 全量完成，MetaQA main 与其余变体仍待运行。
- 修复：`subset_qa` 支持 MetaQA 预处理 `.pt` 分层采样（`996d7d3`）；`e_s` 占位符改为大小写
  不敏感回填（`2177bfc`）；补回 transformers FutureWarning 过滤并将 WebQSP 默认 BERT 对齐
  bge checkpoint（`cdb7d88`、`27fad78`）。
- 当时回归为 261 tests OK，`tests/run_pfit_lib_test.sh` 与 `tests/run_ablation_lib_test.sh` 通过。

<a id="stage3"></a>
## Stage3：checked-batch agent 迁移与验证

**目的。** 将 Ch5 checked-batch（PV-GAC）由只读 legacy `oh_my_agent/` 迁入 `kgqa/agent/`，
并迁移 LLM 服务与路径服务。该阶段是行为保持验证，不产出新的正式论文数字。

### 保真与 smoke 记录

- WebQSP gatev2 的 1,581 条离线回放通过；53 条等距抽样与
  `score2_hopoff_top3_max2_gatev2` 逐位一致。
- 新旧 WebQSP 路径服务在真实缓存上对 5 条样本 parity 通过：三元组、prediction、group_tails
  一致，`log_score` 容差由测试锁定。
- WebQSP 在线 smoke（50 条；K=50、batch=20、lambda=0.2、`hybrid-reject-list`）结果：
  Hit@1 0.9200、HitAny 0.9600、Macro-F1 0.9200、EM 0.8400、Cit-P 0.9152、Cit-R 0.9031、
  Hallucination 0、Format 1.0。重跑同一目录会 resume，不重新调用检索或 LLM。
- MetaQA base zero-shot smoke 为每 hop 10 条：overall Hit@1/HitAny 0.9667、Macro-F1 0.8686、
  Cit-P 0.9667；仅验证通路，不替代适配器正式评测。

### 现役入口与 legacy 边界

| 用途 | 现役入口 | legacy 保留 |
|---|---|---|
| 批量评测 | `python -m kgqa.agent.cli.eval_checked_batch --dataset <ds>` | `oh_my_agent/cli/` |
| 单问调试 | `python -m kgqa.agent.cli.run_checked_batch` | `oh_my_agent/cli/` |
| 路径服务 | `kgqa.retrieve.api.path_retrieve_server` | `oh_my_agent/path_retrieve_server/` |
| LLM 服务 | `kgqa.serving.llm` | `oh_my_agent/llm_server/` |

`oh_my_agent/` 仅保留 WebQSP parity 与历史论文数字凭证。新 Ch5 实验写入
`data/output/kgqa/<dataset>/agent/<run_id>/`，不得覆盖旧
`data/output/WebQSP/checked_batch_agent/`。

## 当前文档路由

| 需求 | 应查阅的现役文档 |
|---|---|
| Ch3 路径检索与下游 QA | `experiments/README.md`、第三章专用配置与对应运行产物 |
| Ch4 目录、配置与可复现编排 | `docs/experiments/experiments_kgqa_reproducible_layout.md`、`experiments/README.md`、`kgqa/pfit/specs.py` |
| Ch5 现役 checked-batch | `AGENTS.md` 的服务与评测章节、`kgqa/agent/` CLI 与 `experiments/README.md` |
