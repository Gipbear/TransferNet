# kgqa Stage3 — checked-batch agent 迁移与 smoke 验证（2026-07-12）

> 这是历史验证记录，保留原始产物路径以便论文数字核对。2026-07-13 起的新第五章实验
> 使用 `experiments/` 编排，并遵循
> [KGQA 三章可复现实验与产物约定](experiments_kgqa_reproducible_layout.md)。

## 范围与结论

Ch5 checked-batch（PV-GAC）已由只读 legacy `oh_my_agent/` 迁入
`kgqa/agent/`；LLM 服务迁入 `kgqa/llm_server/`，路径检索服务统一为
`kgqa/server/`。本期是行为保持的工程迁移，不产出新的正式论文数字。

- WebQSP gatev2：Task 6 一次性全量 1581 条离线回放已通过；本次复核的
  53 条等距抽样与官方 `score2_hopoff_top3_max2_gatev2` 记录逐位一致。
- WebQSP 新旧路径服务：真实 `webqsp_test_1581.pt` 缓存下 5 条服务 parity
  通过（三元组、prediction、group_tails 一致；log_score 容差由测试锁定）。
- 全套回归：`python -m unittest discover -s tests -t . -p 'test*.py' -q`
  为 285 tests OK、17 skipped（外部产物测试默认关闭）。

## WebQSP 在线 smoke（Task 9）

新入口使用 `kgqa.server` 与 `kgqa.llm_server`，输出：
`data/output/kgqa/webqsp/agent/smoke_50/`。

| n | Hit@1 | HitAny | Macro-F1 | EM | Cit-P | Cit-R | Halluc. | Format |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.9200 | 0.9600 | 0.9200 | 0.8400 | 0.9152 | 0.9031 | 0.0000 | 1.0000 |

- 参数：K=50、batch=20、lambda=0.2、`hybrid-reject-list`、score margin=4、
  hop filter、large-answer expansion、constrained checking。
- stop reasons：`mixed=39`、`all_wrong_after_answer=7`、`path_exhausted=4`；
  全部逐样本记录、初始检索与首批答题记录均已写入。
- 该结果与 legacy 20 条 canonical smoke 的量级一致；在线生成不要求逐位相同。
- 对同一输出目录执行相同命令会显示 `[resume] reusing 50 completed samples`，
  不重新调用检索或 LLM，指标保持不变。

## MetaQA base 零样本 smoke（Task 10）

MetaQA 检索服务使用 `data/output/MetaQA_KB/score_cache/metaqa_test_full.pt`；
LLM 请求显式带 `--no_adapter`。输入
`data/output/kgqa/metaqa/agent/smoke_base/input_retrieve_30.jsonl` 从同一缓存
按 hop 各取 10 条（源缓存索引 0–9、9947–9956、24819–24828），避免 test
按 hop 分块导致 `--limit` 只覆盖 1-hop。

| hop | n | Hit@1 | HitAny | Macro-F1 | Cit-P | used_adapter |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 10 | 1.0000 | 1.0000 | 0.9800 | 1.0000 | false |
| 2 | 10 | 1.0000 | 1.0000 | 0.8789 | 1.0000 | false |
| 3 | 10 | 0.9000 | 0.9000 | 0.7470 | 0.9000 | false |
| overall | 30 | 0.9667 | 0.9667 | 0.8686 | 0.9667 | false |

- JSONL 中 hop 分布为 10/10/10；group_tails、实体恒等映射、brackets/e_s 问题
  清洗、hop filter 与 large-answer expansion 均成功走通，未暴露 2-hop 假设。
- hallucination rate=0、format compliance=1。数字仅为通路 smoke，不能替代
  等 `metaqa_main` adapter 完成后的正式评测。

## 新旧入口与后续

| 用途 | 现役入口 | legacy 保留 |
|---|---|---|
| 批量评测 | `python -m kgqa.agent.cli.eval_checked_batch --dataset <ds>` | `oh_my_agent/cli/` |
| 单问调试 | `python -m kgqa.agent.cli.run_checked_batch` | `oh_my_agent/cli/` |
| 路径服务 | `kgqa.server.path_retrieve_server` | `oh_my_agent/path_retrieve_server/` |
| LLM 服务 | `kgqa.llm_server.server` | `oh_my_agent/llm_server/` |
| 演示页面 | `kgqa.agent.demo_page/` | `oh_my_agent/demo_page/` |

`oh_my_agent/` 仅作 WebQSP parity 与历史论文数字凭证，待 MetaQA 新数字落地后再考虑物理删除。后续正式实验应在 `data/output/kgqa/<dataset>/agent/<run_id>/` 下运行；不覆盖旧 `data/output/WebQSP/checked_batch_agent/`。
