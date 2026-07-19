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
| WebQSP 配置 | `c066ecada0e1cc9cf0e6b70275cde9d589719bd960298975ae71ccf6d2513c9e` |
| MetaQA 配置 | `c169312a4775fe1c321fc771280ffb9c0edc86f40e3094de351e5f7175a74f77` |
| CWQ 配置 | `9ec3d64af8d54c55854e803e85d3de45799c26b36f788f7039e7a2b704352146` |
| WebQSP checkpoint | `ffbd8a2edb8a65fc474c74d59b956b398adda1cc52ca9374a5afc0f50c3e4c1e` |
| MetaQA checkpoint | `24ac064027977e3f67c61162438c652806644ea0baef38c39a13b6cc45f465b2` |
| CWQ checkpoint | `9c858a38db90d60b26af8bfff0c9181a96ee0a9db34a135b66a733ae9d40c8fb` |
| Python / PyTorch / NumPy / Transformers | `3.12.12 / 2.10.0+cu128 / 2.3.3 / 5.2.0` |
| 平台 / GPU | `WSL2 Linux 6.18.33.2 / NVIDIA GeForce RTX 4060 Ti 16,380 MiB / driver 560.94` |
| Git 基线 | `3fc023f2f36d754ce51700af90388f7c98a2f50d` |

### P1. 固定加性惩罚基线（T1）

- [x] 在路径选择器中增加三种显式策略：无惩罚、固定加性惩罚、自适应惩罚。
- [x] 将策略字段贯通 offline backend、CLI、实验编排器、manifest 和结果摘要。
- [x] 增加回归测试：`lambda=0` 退化、首条路径保持、固定/自适应公式、确定性并列排序和非法策略校验。
- [x] WebQSP 小样本冒烟，确认输出字段、样本对齐和路径预算。
- [x] WebQSP 全量运行三组：无惩罚、固定、自适应。
- [x] 回填 `experiment_results.md` 的表4-1、差值表和表5-1。

**阶段结论：** 已完成。WebQSP 全量 1,581 条、同一 `topk=500` cache、`K=20, lambda=0.2, eta=1` 下，自适应相对固定加性惩罚的 Answer Hit@20 从 92.66 提高到 93.67，关系多样性从 65.26 提高到 71.10，Top1 均为 73.06；集合 F1 从 41.94 降至 40.93。结论限定为“自适应在保持首条命中的同时改善答案覆盖和关系互补性”，不能表述为所有质量指标全面更优。正式结果位于 `data/output/kgqa/ch3_retrieval/webqsp/transfernet/penalty_ablations/transfernet_v1/`。

**验证记录：** P1 定向回归 56 项通过（跳过 1 项产物门控测试），`compileall` 与 `git diff --check` 通过。默认全量套件仍有与 P1 无关的历史失败：只读 legacy 调用已移除的 `alpha_final` 参数，以及 `test_retrieve_golden` 的 fake backend 与现役逐样本接口不一致；本阶段未扩大范围修复。

### P1-QA. WebQSP 固定与自适应惩罚的下游 QA 补充（T8，延后执行）

- [x] 在 `webqsp_transfernet_v1_downstream_qa.json` 中增加 `fixed` 条件，输入固定为 `penalty_ablations/transfernet_v1/fixed/test.jsonl`。
- [x] 实现 `--condition fixed` 独立评测与 `batch_fixed/` 独立批处理记录，复用已有 `terminal_score_beam`（无惩罚）和 `tarrs`（自适应）QA 结果，不重跑或覆盖其他五组已完成条件。
- [x] 固定同一 base model、prompt、路径格式、解码设置和 20 条路径预算，已完成 100 条分层 `fixed` 冒烟并通过对齐校验。
- [x] 在相同设置下运行 WebQSP 全量 1,581 条 `fixed` QA。
- [x] 将无惩罚 / 固定 / 自适应三组的 QA Hit@1、Hit_any、Macro-F1、Micro-F1 和 EM 回填第7节，并在 P2 中补充配对区间。
- [x] 论文中将 **QA Macro-F1** 作为最终回答集合质量的主 F1，Micro-F1 和 EM 作补充；将当前路径 F1 明确标为“路径尾实体集合 F1”，只用于解释检索覆盖—噪声权衡。

**阶段结论：** 已完成。六组输入均通过 1,581 条样本对齐与统一 QA 签名校验，正式汇总位于 `data/output/kgqa/ch3_retrieval/webqsp/transfernet/downstream_qa/transfernet_v1/reports/base_zeroshot/full/condition_matrix.json`。固定惩罚条件的 Hit@1 / Hit_any / QA Macro-F1 / QA Micro-F1 / EM 分别为 82.16% / 89.31% / 64.71% / 34.20% / 34.47%。自适应相对固定的五项差值依次为 +0.57 / +0.70 / +0.16 / -0.89 / -0.63 个百分点；其中 QA Macro-F1 与 Hit@1 的配对 95% CI 分别为 [-0.76, 1.08] 和 [-0.51, 1.64] 个百分点，均跨 0。因此只能表述为“自适应的路径覆盖及下游命中点估计略高”，不能宣称其相对固定惩罚稳定改善最终回答集合质量。

### P2. WebQSP 统计证据（T3、T5）

- [x] 预先固定配对比较和统计指标，避免结果导向的比较选择。
- [x] 对已有全量产物的路径 Answer Hit@20、Top1 Hit 和下游 QA Hit@1 计算配对 bootstrap 95% 置信区间。
- [x] P1-QA 全量完成后，对固定 / 自适应的 QA Macro-F1 与 Hit@1 补充配对区间。
- [x] 在同一环境、同一 cache、同一题目顺序下，对 SP、Score-Beam、终点感知、固定、自适应进行预热和重复计时。
- [x] 采集平均时间、P50、P95、峰值内存、扩展状态数、候选路径数和最终路径数。
- [x] 回填结果文档第8、9节，并保存机器可读统计产物。

**预注册口径：** 配置冻结在 `experiments/configs/ch3/webqsp_transfernet_v1_p2.json`。路径主比较为 Score-Beam−SP、终点感知−Score-Beam、自适应−固定、自适应−无惩罚，指标固定为 Answer Hit@20 和 Top1 Hit；下游 QA 主比较为终点感知−Score-Beam、TARRS−终点感知、TARRS−无路径，指标固定为 Hit@1；P1-QA 完成后按同一预定口径补入 TARRS−固定惩罚的 Macro-F1 与 Hit@1。估计量均为同一样本上的均值差（左方法减右方法），使用固定随机种子和 20,000 次配对 bootstrap 百分位区间；以区间是否跨 0 判断证据方向，不追加事后筛选的比较或未经预注册的 p 值。

**阶段结论：** 已完成，机器可读产物与口径说明归档于 `data/analysis/20260719_1719__ch3_p2_evidence/`。20,000 次配对 bootstrap 显示：终点感知相对 Score-Beam 的路径 Answer Hit@20、Top1 Hit 和 QA Hit@1 区间均不跨 0；自适应相对固定的路径 Answer Hit@20 为 +1.01 个百分点，95% CI [0.38, 1.64]，Top1 差值为 0 且区间跨 0；TARRS 相对终点感知的 QA Hit@1 为 +1.58 个百分点，95% CI [0.44, 2.72]。新增的 TARRS−固定惩罚比较中，QA Macro-F1 为 +0.16 个百分点，95% CI [-0.76, 1.08]；QA Hit@1 为 +0.57 个百分点，95% CI [-0.51, 1.64]，两者均跨 0。同一 cache、前 100 条预热、全量 1,581 条重复 3 次的效率测量中，五组平均时间均约 66–69ms/题，自适应相对 Score-Beam 未呈现可分辨的额外开销；SP 平均时间接近但 P95 和增量 Python 内存峰值更高。

### P3. WebQSP 成功/失败案例（T4）

- [ ] 筛选终点融合收益案例。
- [ ] 筛选 TARRS 答案覆盖或关系互补收益案例。
- [ ] 筛选多样性提高但引入噪声、首条路径或下游 QA 下降的失败案例。
- [ ] 保存问题、主题实体、golden、各方法路径、路径分数、关系相似度和下游答案。
- [ ] 将案例与适用边界写入 `data/analysis/YYYYMMDD_HHMM__ch3_case_analysis/README.md`。

**阶段结论：** 待运行。案例只用于解释机制和边界，不替代全量统计。

### P4. MetaQA 3-hop 路径主对照（O1）

- [x] 补齐 MetaQA checkpoint、QA 输入、score cache 和 3-hop 数据划分核验。
- [x] 只运行 3-hop 路径级主对照：SP、得分引导、固定、自适应。
- [x] 检查各条件的题目顺序、golden、样本数和路径格式完全一致。
- [x] 按 3-hop 报告 Answer Hit@K、Top1 Hit、F1、关系多样性和平均路径数。

**执行入口：**

```bash
python -m experiments.ch3.metaqa_p4 --phase prepare
python -m experiments.ch3.run --dataset metaqa --config experiments/configs/ch3/metaqa_transfernet_v1_p4.json --phase penalty_ablation --no_progress
python -m experiments.ch3.run --dataset metaqa --config experiments/configs/ch3/metaqa_transfernet_v1_p4.json --phase shortest_path --no_progress
python -m experiments.ch3.metaqa_p4 --phase report
```

**阶段结论：** 已完成。先从 39,093 条 MetaQA test cache 中按数据集标签显式筛出 14,274 条 3-hop 样本，源顺序保持不变；四组结果的 `sample_index`、问题、golden、样本数及路径三元组格式均通过严格对齐检查。SP、得分引导、固定和自适应四组的 Answer Hit@20 / Top1 Hit 均为 100.00%；得分引导、固定、自适应的 F1 均为 57.47%，高于 SP 的 50.97%，但关系多样性约 18.6%，低于 SP 的 55.00%。自适应相对固定只将关系多样性从 18.58% 提高到 18.65%，其余主指标在四位小数口径相同，因此 P4 不能支持“自适应在 MetaQA 上稳定显著优于固定惩罚”或“三个数据集均稳定提升”的结论。机器汇总位于 `data/output/kgqa/ch3_retrieval/metaqa/transfernet/p4_3hop/transfernet_v1_3hop_summary.json`。

### P5. MetaQA 3-hop 端到端 QA（T6）

- [x] 将第三章下游 QA 编排扩展为支持 MetaQA。
- [x] 固定同一个 base model、提示、解码设置和路径预算。
- [x] 严格校验五组 14,274 条 3-hop 输入，并完成共同前 30 条的五条件真实冒烟与汇总。
- [x] 使用实际 tokenizer 预计算 TARRS 全量输入长度：P99=842、最大=1,025，0 条超过 2,048。
- [ ] 全量运行 TARRS 完整方法单一条件（14,274 条 3-hop 样本）。
- [ ] 基于 TARRS 全量结果报告 Hit@1、Hit_any、Macro-F1、Micro-F1、EM；不再补充 MetaQA 条件对比或配对区间。

**全量执行入口：**

```bash
python -m experiments.ch3.run_downstream_qa \
  --dataset metaqa --condition tarrs --layer base_zeroshot \
  --phase eval --progress_interval 100

python -m experiments.ch3.run_downstream_qa \
  --dataset metaqa --condition tarrs --layer base_zeroshot \
  --phase report
```

同机 `smoke_30` 的 30 个样本单条件生成约耗时 2–3 分钟，线性外推 TARRS 单条件全量约需 19 GPU 小时，实际以运行时为准。重跑会跳过已有兼容 `summary.json` 的条件；被中断的单个条件需要从头执行。

**阶段结论：** 执行代码与冒烟已完成，全量 TARRS QA 尚未运行。五组输入均为同一批 14,274 条 MetaQA 3-hop 样本，题目与 golden 签名一致；实际 tokenizer 扫描显示 TARRS 输入 P99=842、最大=1,025，未触发 2,048 上下文限制；`smoke_30` 的五组预测数均为 30，逐条件状态均为 `completed`，统一报告位于 `data/output/kgqa/ch3_retrieval/metaqa/transfernet/downstream_qa/transfernet_v1_3hop/reports/base_zeroshot/smoke_30/`。冒烟指标只用于确认执行链路健康，不写入正式全量结果表，也不据此比较方法优劣。

### P6. CWQ 路径主对照与端到端 QA（O2、T7，整体延期）

**阶段状态：** 延期。本阶段不执行 CWQ 路径级实验、下游 QA 或任何消融，当前论文正文不提 CWQ。

- [ ] （延期）补齐 CWQ checkpoint、QA 输入、逐样本子图和 score cache 配置。
- [ ] （延期）核对逐样本 triples 与缓存、golden、样本索引的一致性。
- [ ] （延期）先完成 SP、得分引导、固定、自适应路径级对照。
- [ ] （延期）分析子图覆盖、候选裁剪、路径排序和 LLM 利用失败来源。
- [ ] （延期）路径结果稳定后，按固定模型、提示、解码和路径预算完成端到端 QA。

**阶段结论：** 已决定延期，不纳入当前实验闭环和论文正文。

### P7. 外部方法可比性核对（T2）

- [x] 为 TransferNet、ReaRev、UniKGQA、RoG、ToG、GNN-RAG 等方法记录论文、年份、表号/页码。
- [x] 记录 KG 版本、子图/检索方式、实体链接、骨干/LLM 和是否额外训练。
- [x] 逐行给出可比性说明，只在条件足够接近时写总体竞争力结论。
- [x] 回填 `experiment_results.md` 第3节。

**阶段结论：** 已完成。WebQSP 外部结果已按论文表号/页码、KG/子图、实体链接、骨干/训练差异逐行记录。由于子图、骨干、训练方式和指标定义仍不相同，本阶段只形成有边界的背景比较，不宣称优于现有 SOTA。CWQ 继续延期。

### P8. 论文证据收口

- [ ] 将已完成实验的数值、配置、summary、JSONL 和图表逐项互相追溯。
- [ ] 更新 `experiment_results.md` 的状态、结论和待补标记。
- [ ] 更新 `experiment_todo.md`，删除已完成事项或标记其结果位置。
- [ ] 核对 `thesis_outline.md` 的每个论文主张都有对应证据。
- [x] 明确保留的限制：不重跑既有四组分数组成消融、六组 WebQSP QA、210 组扫描和 top-k 饱和性。

**阶段结论：** 待收口。完成后再形成第三章最终实验结论。

## 当前总状态

- 已完成：P0 基线审计；P1 固定加性惩罚；P1-QA 六组 WebQSP 全量评测；P2 配对统计与同环境效率；P4 MetaQA 3-hop 路径主对照。
- 当前进行：P5 MetaQA 3-hop 端到端 QA；执行代码和 `smoke_30` 已完成，等待用户运行 TARRS 单条件 14,274 条全量评测。
- 下一步：P5 全量完成后生成单条件报告；P3 按此前安排继续跳过，CWQ 整体延期。
- 当前未完成：P3、P5、P8 的案例和最终论文证据收口；P6/CWQ 已延期。

## 最终结论记录区

> 只有对应阶段的代码、测试和正式产物都完成后，才将阶段状态改为 `[x]`，并在这里写入结论。不得用冒烟结果替代全量结果。

- 固定加性惩罚结论：固定惩罚相对无惩罚带来小幅覆盖和关系多样性收益；自适应相对固定进一步提高 Answer Hit 1.01、关系多样性 5.84 个百分点并保持 Top1，但路径尾实体集合 F1 下降 1.01 个百分点。
- 固定 / 自适应下游 QA 结论：固定惩罚与自适应的 QA Macro-F1 分别为 64.71% 和 64.87%；自适应差值 +0.16 个百分点，95% CI [-0.76, 1.08]，不能支持最终回答集合质量的稳定优势。Hit@1 点估计提高 0.57 个百分点，但 95% CI [-0.51, 1.64] 同样跨 0。
- WebQSP 统计与效率结论：终点融合的核心路径与 QA 命中收益具有不跨 0 的配对区间；自适应相对固定的路径覆盖提升稳定，但 Top1 和下游 QA Macro-F1 / Hit@1 未形成不跨 0 的优势。五组平均检索时间约 66–69ms/题，TARRS 相对普通 Score-Beam 未呈现可分辨的额外开销。
- WebQSP 案例与适用边界：待补
- MetaQA 3-hop 结论：四组 Answer Hit@20 与 Top1 均为 100.00%；三种得分方法 F1 为 57.47%，高于 SP 的 50.97%，但关系多样性更低；自适应相对固定仅提高 0.07 个百分点的关系多样性，不能支持稳定优势。
- CWQ 结论：延期，不纳入当前版本
- 外部方法可比性结论：WebQSP 外部结果已完成协议审计。监督式 KGQA、LLM 路径规划和 GNN+LLM 结果不能与本章固定 Llama 3.1 8B 零样本 QA 或路径 Answer Hit 直接合并排名，论文正文仅明确子图、模型、训练方式和指标定义差异。
- 第三章最终证据边界：待补
