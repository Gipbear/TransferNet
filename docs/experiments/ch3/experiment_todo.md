# 第三章待补实验与待决事项

> 本文件只记录当前结果文档无法覆盖、且需要新增实验、外部文献核对或用户决策的事项。
> 已完成结果见 [experiment_results.md](experiment_results.md)。
> 第三章正式实验范围现收敛为 WebQSP，P3 案例与 P8 证据收口均已完成。MetaQA 3-hop 的既有产物转为归档，后续仅在第四章路径监督微调后重新评估；CWQ 继续延期。

## 0. 已完成并转入结果文档

| 编号 | 完成事项 | 结论与产物 |
|---|---|---|
| T1 | 固定加性惩罚基线 | 已完成无惩罚 / 固定 / 自适应三组 WebQSP 全量对照；自适应相对固定提高 Answer Hit 与关系多样性、保持 Top1，但 F1 下降。结果见 `experiment_results.md` 表4-1、表5-1及 `penalty_ablations/transfernet_v1/`。 |
| T3 | 同一环境检索效率基准 | 已完成 SP、普通 Score-Beam、终点感知、固定和自适应五组全量 3 次重复计时。五组平均时间约 66–69ms/题，TARRS 相对普通 Score-Beam 未呈现可分辨的额外开销；产物见 `data/analysis/20260719_1719__ch3_p2_evidence/efficiency.json`。 |
| T5 | 配对 bootstrap 置信区间 | 已完成路径与下游 QA 的预定比较。自适应相对固定的路径 Answer Hit@20 区间不跨 0；QA Macro-F1 / Hit@1 的差值为 +0.16 / +0.57 个百分点，但 95% CI 均跨 0。结果见 `experiment_results.md` 表9-3及 `data/analysis/20260719_1719__ch3_p2_evidence/paired_bootstrap.json`。 |
| T8 | WebQSP 固定惩罚下游 QA | 已完成 1,581 条 fixed 全量评测和六组统一报告。fixed 的 Hit@1 / Hit_any / Macro-F1 / Micro-F1 / EM 为 82.16% / 89.31% / 64.71% / 34.20% / 34.47%；结果见 `experiment_results.md` 表7-1及 `downstream_qa/transfernet_v1/reports/base_zeroshot/full/condition_matrix.json`。 |
| T4 | WebQSP 成功与失败案例分析 | 已完成终点融合收益（S1）、TARRS 关系互补收益（S2）和多样性噪声边界（F1）三个正式对齐样本；归档见 `data/analysis/20260722_2040__ch3_case_analysis/README.md`，图见 `docs/experiments/ch3/images/fig3-6_case_analysis.svg`。 |
| T2 | 外部先进方法结果与可比性核对 | 已完成 WebQSP 的 TransferNet、ReaRev、UniKGQA、RoG、ToG、GNN-RAG 论文/年份、表号/页码、KG/子图、实体链接、骨干/LLM、训练差异和逐行可比性说明。结果见 `experiment_results.md` 第3.1、3.2节；CWQ 按决定延期。 |

## 1. 待决事项

| 编号 | 事项 | 为什么需要决定 | 可选方向 |
|---|---|---|---|
| D2 | 扩展至 MetaQA | 暂缓，MetaQA 不属于第三章正式范围。已有 3-hop 路径与未微调 LLM 零样本 QA 产物仅归档。 | 第四章完成路径监督微调后，如仍有需要再单独立项，使用微调模型和新基线重新评估；CWQ 不执行、不进入当前论文；ReaRev 暂不纳入。 |
| D3 | 参数报告口径 | 已确认不重跑验证集。 | 全部正式对比固定 `beam=20, lambda=0.2, eta=1`；参数扫描只分析趋势，不使用“独立验证集选出最优参数”的表述。 |

## 2. 若论文保留相应主张，则必须补充的证据

| 编号 | 需要补充的工作 | 触发的论文主张 | 当前缺口 | 产物要求 |
|---|---|---|---|---|
| T6 | MetaQA 3-hop 微调后评估 | 第四章可能的补充评估，不属于第三章 | 当前无路径监督微调后的 MetaQA 比较。 | 仅在第四章路径监督微调完成后重新立项；不得使用现有零样本 QA 结果代替。 |
| T7 | CWQ 端到端 QA | 暂延期，不进入当前论文 | CWQ 整体延期，当前无路径级或 QA 结果。 | 本阶段不执行，后续重新立项时再补充。 |

## 3. 可选扩展，不阻塞当前 WebQSP / TransferNet 闭环

| 编号 | 扩展 | 预期回答的问题 |
|---|---|---|
| O2 | CWQ 路径主对照 | 暂延期，不进入当前论文。 |
| O3 | ReaRev 适配 | 暂缓：不纳入当前第三章实验范围。 |
| O4 | 路径充分性 / 必要性干预 | 路径是否对模型答案具有因果贡献，而非仅相关。 |
| O5 | 解耦搜索宽度与路径数量的扩展 | 当前 `beam_size=K` 已足以报告路径规模趋势。仅当后续需要单独研究内部搜索宽度时，再新增独立控制量；不作为本章当前必做实验。 |

## 4. 当前不应补做的事项

- 不重新运行已完成的四组分数组成消融、六组全量 QA、210 组参数扫描或 top-k 饱和性；
- 不将 100 条冒烟结果写入论文数值表；
- 不把骨干 TransferNet 原始答案指标当作路径方法指标；
- 不将“自适应在覆盖—关系多样性上优于固定”扩大为“所有路径质量指标都更优”；当前全量结果中自适应的集合 F1 低于固定惩罚。

