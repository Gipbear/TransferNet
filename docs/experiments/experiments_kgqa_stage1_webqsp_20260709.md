# kgqa Stage1 Plan1 — WebQSP 全量评测记录（2026-07-09）

## 配置

- ckpt: `data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt`（不重训，仅前向；BERT + 内联稀疏矩阵，2-hop）
- 数据: WebQSP test 全量 1581 条（`data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt`）
- 流程: `kgqa.cli.dump_scores --dataset webqsp`（缓存 `webqsp_test_1581.pt`）→ `kgqa.cli.eval --backend offline`
- 内核: kgqa 统一检索内核（`tail_blend`/`LogNorm` + MMR 选择），engine 公式为数值红线未改
- 口径: MID 口径（`gold_key="mid"`，`group_by=None`），无分跳切片
- 缓存/summary: `data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt`、`data/output/WebQSP/kgqa_eval/webqsp_test_full_summary.json`（均 gitignored）

## 答案级指标（answer）

| 切片 | n | hit1 | hit_any | macro_f1 | micro_f1 | EM |
|---|---|---|---|---|---|---|
| overall | 1581 | 0.7116 | 0.7938 | 0.5799 | 0.2587 | 0.3858 |

- overall hit1 0.7116 与 ckpt 训练报告 acc 0.7154 量级吻合（差 0.004），验证「dump → offline 检索 → 评测」链路保真。
- 该链路是 Plan2(MetaQA)/Plan3(CWQ) 的接口基准：三数据集共享同一 engine 与评测口径。

## 路径级指标（path）

| 切片 | n | answer_hit | top1_hit | precision | recall | f1 |
|---|---|---|---|---|---|---|
| overall | 1581 | 0.9576 | 0.7312 | 0.2392 | 0.9223 | 0.3145 |

- 多样性/覆盖：`jaccard_diversity` 0.9388、`relation_jaccard_diversity` 0.7158、`tail_diversity` 0.4943、`relation_coverage` 0.2268、`edge_coverage` 0.6131。

## 与 Ch3 的差异说明

单一 kgqa 内核收敛：未逐条复现 Ch3 的 `mmr_diversity_beam_search` 数值，路径集合与多样性指标与 Ch3 存在差异属预期。保真依据为离线回归锁（`tests/kgqa/test_webqsp_regression.py`，检索路径须与旧 `scripts/offline_path_search.run_experiment` 逐条一致，免 ckpt）+ 答案 acc 对齐（0.7116 vs 0.7154）+ online/offline parity（`test_backend_parity.py`）。

## 过程记录

- Plan1 是统一框架的核心骨架：先落地 WebQSP 端到端以锁定 `types.py`/`scores`/`retrieve`/`eval` 接口，MetaQA/CWQ 再按同接口接入。
- 全量 1581 条评测秒级完成（离线缓存检索，无前向）。
- 合并记录：PR #2（merge commit `245ead8`）。
