# 第三章实验待办专项执行计划

> 用途：跟踪 `experiment_todo.md` 中尚未完成的实验与论文证据。
>
> 分支：`experiment/ch3-todo-execution`
>
> 基线：本地 `main`，创建时 `HEAD=3fc023f2f36d754ce51700af90388f7c98a2f50d`
>
> 原则：所有正式对照固定 `beam=20, lambda=0.2, eta=1, K=20`；参数扫描只用于趋势分析，不重新选参。

## 状态约定

- `[ ]` 未开始
- `[-]` 进行中
- `[x]` 已完成，并已记录产物与结论
- `[!]` 受阻，需要补充输入、配置或人工决策

## 阶段计划

### P0. 分支与实验基线

- [x] 从本地 `main` 创建 `experiment/ch3-todo-execution`。
- [x] 核验 `git merge-base HEAD main` 与 `git rev-parse main` 一致。
- [x] 记录当前工作树中已有的用户改动，不覆盖、不混入本专项变更。
- [x] 统一 WebQSP、MetaQA、CWQ 的配置、输入、checkpoint、score cache 和运行环境指纹。
- [x] 明确正式结果目录、样本数和数据划分，避免历史 cache 与当前实验口径混用。

**阶段结论：** 分支已建立在本地 `main` 当前提交上；工作树原有未提交改动仍保留，后续专项文件只在本分支继续。三套配置均已通过 `experiments.ch3.run --phase scores --dry_run`；MetaQA 现有 test cache 含 1/2/3-hop，正式 P4 仍需显式筛选 3-hop，不能直接把 39,093 条全量缓存当作 P4 结果。

#### P0 审计记录

| 数据集 | 正式输入目录 | checkpoint | train / test QA | 已验证 test cache | test 样本数 | hop 口径 |
|---|---|---|---|---|---:|---|
| WebQSP | `data/input/WebQSP` | `data/ckpt/WebQSP/model-49-0.7154.pt` | `qa_train_webqsp_fixed.txt` / `qa_test_webqsp_fixed_1581.txt` | `data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt` | 1,581 | 骨干可用步数，最高 2-hop |
| MetaQA | `data/input/MetaQA_KB` | `data/ckpt/MetaQA_KB/model_epoch-7_acc-0.9936.pt` | `train.pt` / `test.pt` | `data/output/MetaQA_KB/score_cache/metaqa_test_full.pt` | 39,093 | 1-hop 9,947；2-hop 14,872；3-hop 14,274；P4 只取 3-hop |
| CWQ | `data/input/CWQ`（解析后指向 `data/resources/CWQ`） | `data/ckpt/CWQ/model-29-0.4206.pt` | `train_simple.json` / `test_simple.json` | `data/output/CWQ/score_cache/cwq_test_full.pt` | 3,531 | 逐样本子图，缓存含 3,531 份 `triples` |

正式运行目录统一使用：

- score cache：`data/output/kgqa/shared/<dataset>/backbones/transfernet/scores/`
- 第三章结果：`data/output/kgqa/ch3_retrieval/<dataset>/transfernet/`
- WebQSP 已有正式结果：`data/output/kgqa/ch3_retrieval/webqsp/transfernet/`

上表中的三个已验证 cache 是历史或服务目录中的只读输入；若正式运行重新生成 cache，必须写入上述 canonical 目录，并在 manifest 中记录输入指纹，不能把旧服务产物和新实验结果混写。

#### P0 文件指纹与环境

| 项目 | SHA-256 / 版本 |
|---|---|
| WebQSP 配置 | `095f3f629d5843ba6f22b86c2c8a4f266cdfffb5a4fc6a88d455da231f4d3519` |
| MetaQA 配置 | `f4214e2b09ae303eb1bd32d320bcb8e45167c2937b9759cd67e3de14f44e6d7c` |
| CWQ 配置 | `2e3c59f13e81c3b340cd8fc3c3705385a1bf41654ddfe648350a868028c7e114` |
| WebQSP checkpoint | `ffbd8a2edb8a65fc474c74d59b956b398adda1cc52ca9374a5afc0f50c3e4c1e` |
| MetaQA checkpoint | `24ac064027977e3f67c61162438c652806644ea0baef38c39a13b6cc45f465b2` |
| CWQ checkpoint | `9c858a38db90d60b26af8bfff0c9181a96ee0a9db34a135b66a733ae9d40c8fb` |
| Python / PyTorch / NumPy / Transformers | `3.12.12 / 2.10.0+cu128 / 2.3.3 / 5.2.0` |
| 平台 / GPU | `WSL2 Linux 6.18.33.2 / NVIDIA GeForce RTX 4060 Ti 16,380 MiB / driver 560.94` |
| Git 基线 | `3fc023f2f36d754ce51700af90388f7c98a2f50d` |

### P1. 固定加性惩罚基线（T1）

- [ ] 在路径选择器中增加三种显式策略：无惩罚、固定加性惩罚、自适应惩罚。
- [ ] 将策略字段贯通 offline backend、CLI、实验编排器、manifest 和结果摘要。
- [ ] 增加回归测试：`lambda=0` 退化、首条路径保持、固定/自适应公式、确定性并列排序和非法策略校验。
- [ ] WebQSP 小样本冒烟，确认输出字段、样本对齐和路径预算。
- [ ] WebQSP 全量运行三组：无惩罚、固定、自适应。
- [ ] 回填 `experiment_results.md` 的表4-1、差值表和表5-1。

**阶段结论：** 待运行。只有完成本阶段后，才能判断“自适应惩罚优于固定加性惩罚”是否成立。

### P2. WebQSP 统计证据（T3、T5）

- [ ] 预先固定配对比较和统计指标，避免结果导向的比较选择。
- [ ] 对路径 Answer Hit@20、Top1 Hit 和下游 QA Hit@1 计算配对 bootstrap 95% 置信区间。
- [ ] 在同一环境、同一 cache、同一题目顺序下，对 SP、Score-Beam、终点感知、固定、自适应进行预热和重复计时。
- [ ] 采集平均时间、P50、P95、峰值内存、扩展状态数、候选路径数和最终路径数。
- [ ] 回填结果文档第8、9节，并保存机器可读统计产物。

**阶段结论：** 待运行。没有统一计时和配对区间前，不使用“显著提升”或“低额外成本”表述。

### P3. WebQSP 成功/失败案例（T4）

- [ ] 筛选终点融合收益案例。
- [ ] 筛选 TARRS 答案覆盖或关系互补收益案例。
- [ ] 筛选多样性提高但引入噪声、首条路径或下游 QA 下降的失败案例。
- [ ] 保存问题、主题实体、golden、各方法路径、路径分数、关系相似度和下游答案。
- [ ] 将案例与适用边界写入 `data/analysis/YYYYMMDD_HHMM__ch3_case_analysis/README.md`。

**阶段结论：** 待运行。案例只用于解释机制和边界，不替代全量统计。

### P4. MetaQA 3-hop 路径主对照（O1）

- [ ] 补齐 MetaQA checkpoint、QA 输入、score cache 和 3-hop 数据划分核验。
- [ ] 只运行 3-hop 路径级主对照：SP、得分引导、固定、自适应。
- [ ] 检查各条件的题目顺序、golden、样本数和路径格式完全一致。
- [ ] 按 3-hop 报告 Answer Hit@K、Top1 Hit、F1、关系多样性和平均路径数。

**阶段结论：** 待运行。完成前不写跨数据集稳定性结论。

### P5. MetaQA 3-hop 端到端 QA（T6）

- [ ] 将第三章下游 QA 编排扩展为支持 MetaQA。
- [ ] 固定同一个 base model、提示、解码设置和路径预算。
- [ ] 运行无路径、SP、得分路径、固定、自适应五组条件。
- [ ] 报告 Hit@1、Hit_any、Macro-F1、Micro-F1、EM，并补充 QA Hit@1 配对区间。

**阶段结论：** 待运行。必须建立在 MetaQA 3-hop 路径产物稳定之后。

### P6. CWQ 路径主对照与端到端 QA（O2、T7）

- [ ] 补齐 CWQ checkpoint、QA 输入、逐样本子图和 score cache 配置。
- [ ] 核对逐样本 triples 与缓存、golden、样本索引的一致性。
- [ ] 先完成 SP、得分引导、固定、自适应路径级对照。
- [ ] 分析子图覆盖、候选裁剪、路径排序和 LLM 利用失败来源。
- [ ] 路径结果稳定后，按固定模型、提示、解码和路径预算完成端到端 QA。

**阶段结论：** 待运行。CWQ 路径级结果完成前不启动端到端 QA。

### P7. 外部方法可比性核对（T2）

- [ ] 为 TransferNet、ReaRev、UniKGQA、RoG、ToG、GNN-RAG 等方法记录论文、年份、表号/页码。
- [ ] 记录数据划分、KG 版本、子图/检索方式、实体链接、骨干/LLM 和是否额外训练。
- [ ] 逐行给出可比性说明，只在条件足够接近时写总体竞争力结论。
- [ ] 回填 `experiment_results.md` 第3节。

**阶段结论：** 待核对。外部结果不能只记录一个数字，必须保留来源和比较边界。

### P8. 论文证据收口

- [ ] 将已完成实验的数值、配置、summary、JSONL 和图表逐项互相追溯。
- [ ] 更新 `experiment_results.md` 的状态、结论和待补标记。
- [ ] 更新 `experiment_todo.md`，删除已完成事项或标记其结果位置。
- [ ] 核对 `thesis_outline.md` 的每个论文主张都有对应证据。
- [ ] 明确保留的限制：不重跑既有四组分数组成消融、五组 WebQSP QA、210 组扫描和 top-k 饱和性。

**阶段结论：** 待收口。完成后再形成第三章最终实验结论。

## 当前总状态

- 已完成：P0 的分支、配置、输入、缓存和环境审计。
- 当前进行：专项计划建立。
- 下一步：实现 P1 固定加性惩罚基线。
- 当前未完成：P1–P8 的实验、统计、案例和论文收口。

## 最终结论记录区

> 只有对应阶段的代码、测试和正式产物都完成后，才将阶段状态改为 `[x]`，并在这里写入结论。不得用冒烟结果替代全量结果。

- 固定加性惩罚结论：待补
- WebQSP 统计与效率结论：待补
- WebQSP 案例与适用边界：待补
- MetaQA 3-hop 结论：待补
- CWQ 结论：待补
- 外部方法可比性结论：待补
- 第三章最终证据边界：待补
