# kgqa Stage1 Plan3 — CWQ 全量评测记录（2026-07-11）

## 配置

- ckpt: `data/ckpt/CWQ/model-29-0.4206.pt`（不重训，仅前向；`num_ways=1, num_steps=2, bert-base-cased, rev=False`）
- 数据: CompWebQ(CWQ) test 全量 3531 条（`data/input/CWQ/test_simple.json`，无空子图样本，扣除数 0）
- 流程: `kgqa.cli.dump_scores --dataset cwq`（全量缓存 554MB，逐样本子图 `triples` 随缓存自包含）→ `kgqa.cli.eval --backend offline`
- 内核: 复用 WebQSP 单一检索内核（`tail_blend`/`LogNorm`），engine 公式未改；CWQ 为逐样本子图，边来源由 `CWQAdapter.kg_edge_source(sample)` 用 `GlobalKG.from_triples(sample.triples)` 现建邻接（不再 init 期取全局图）
- 口径: MID 口径（gold 为整数 id 经 `id2ent` 还原为 MID，与 prediction 的 MID 键同口径）；`metric_spec` `gold_key="mid"`、`group_by=None`，故无分跳切片
- 缓存/summary: `data/output/CWQ/score_cache/cwq_test_full.pt`、`data/output/CWQ/eval/cwq_test_summary.json`（均 gitignored）

## 答案级指标（answer）

| 切片 | n | hit1 | hit_any | macro_f1 | micro_f1 | EM |
|---|---|---|---|---|---|---|
| overall | 3531 | 0.4084 | 0.4656 | 0.3835 | 0.3433 | 0.3121 |

- overall hit1 0.4084 与 ckpt 训练报告 acc 0.4206 量级吻合（差 0.012，在 ±0.02 内），验证「dump → offline 检索 → 评测」链路保真。
- 差异主要源于评测内核（WebQSP `tail_blend`/`LogNorm` top 路径口径）与训练时 `predict.py` 的 `e_score` argmax 口径不完全一致；量级一致即视为通过。

## 路径级指标（path）

| 切片 | n | answer_hit | top1_hit | precision | recall | f1 |
|---|---|---|---|---|---|---|
| overall | 3531 | 0.6907 | 0.4123 | 0.0959 | 0.6746 | 0.1423 |

- 多样性/覆盖：`jaccard_diversity` 0.9012、`relation_jaccard_diversity` 0.6096、`tail_diversity` 0.8229、`relation_coverage` 0.2892、`edge_coverage` 0.7374。

## 与旧 CompWebQ predict 内核的差异说明

单一 WebQSP 内核收敛：未逐条复现 `CompWebQ/predict.py` 的多路（`num_ways`）内核数值，路径集合与答案 top1 存在差异属预期。保真依据为 online/offline parity 测试（`tests/kgqa/test_cwq_end_to_end.py::test_online_offline_parity_first3`）+ 答案 acc 量级对齐（0.4084 vs 0.4206）+ 缓存含逐样本 triples 的集成断言。

## 过程记录

- 有效样本 N = 3531，空子图扣除 0（test_simple.json 全部样本 `subgraph.tuples` 非空）。注意 CompWebQ `DataLoader` 内部报的 "bad number"（如小子集 20 条中 3 条）是训练期排除口径，不减少产出样本数。
- 逐样本子图分发：两个 backend 从「init 取一次全局 edge_source」改为「每次检索调 `adapter.kg_edge_source(sample)`」，WebQSP/MetaQA 全局图数据集忽略 sample 参数、行为不变（62 项 kgqa 测试零回归）。
- BERT 加载：`bert-base-cased` 有本地缓存，设 `HF_HUB_OFFLINE=1` 免代理；加载 ckpt 时 `cls.*`（MLM/NSP 头）报 UNEXPECTED 属正常（不同任务架构），`strict=False` 忽略。
- 耗时：全量 dump（3531 条前向 + 缓存 554MB）约 20 分钟；全量 offline 评测数分钟。
- 未踩 Plan2 的 name 口径全 0 坑：CWQ 走 MID 口径（`gold_key="mid"`），gold/pred 同为 MID，端到端 `hit1>0` 断言一次通过。
