# kgqa Stage1 Plan2 — MetaQA 全量分跳评测记录（2026-07-10）

## 配置

- ckpt: `data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt`（不重训，仅前向）
- 数据: MetaQA_KB test 全量 39093 条（hop1: 9947 / hop2: 14872 / hop3: 14274）
- 流程: `kgqa.cli.dump_scores --dataset metaqa`（全量缓存 718MB，`per_hop_limit=0`）→ `kgqa.cli.eval --backend offline`
- 内核: 复用 WebQSP 单一检索内核（`tail_blend`/`LogNorm`），MetaQA 用 gold hop 合成 one-hot `hop_attn`（与 Ch3 用 gold hop 一致）；engine 在 3-hop 输入下无需修改边界
- 缓存/summary: `data/output/MetaQA_KB/score_cache/metaqa_test_full.pt`、`data/output/MetaQA_KB/eval/metaqa_test_summary.json`（均 gitignored）

## 答案级指标（answer）

| 切片 | n | hit1 | hit_any | macro_f1 | micro_f1 | EM |
|---|---|---|---|---|---|---|
| overall | 39093 | 0.9938 | 0.9991 | 0.7947 | 0.5681 | 0.6171 |
| 1-hop | 9947 | 0.9758 | 0.9964 | 0.9608 | 0.9428 | 0.9056 |
| 2-hop | 14872 | 1.0000 | 1.0000 | 0.8526 | 0.7035 | 0.7263 |
| 3-hop | 14274 | 1.0000 | 1.0000 | 0.6185 | 0.4304 | 0.3023 |

- overall hit1 0.9938 与 ckpt 训练报告 acc 0.9937 吻合，验证「dump → offline 检索 → 评测」链路保真。
- 2/3-hop hit1=1.0、1-hop 0.9758：1-hop 存在并列最高分导致 top1 偶有旁落；多答案样本随跳数增多，recall/EM 随跳数下降属预期（检索 top 路径数固定）。

## 路径级指标（path）

| 切片 | n | answer_hit | top1_hit | precision | recall | f1 |
|---|---|---|---|---|---|---|
| overall | 39093 | 0.9999 | 0.9940 | 0.5512 | 0.9812 | 0.6480 |
| 1-hop | 9947 | 0.9998 | 0.9764 | 0.7600 | 0.9997 | 0.7970 |
| 2-hop | 14872 | 1.0000 | 1.0000 | 0.4822 | 0.9944 | 0.6204 |
| 3-hop | 14274 | 1.0000 | 1.0000 | 0.4774 | 0.9545 | 0.5728 |

## 与 Ch3 的差异说明

单一 WebQSP 内核收敛：未逐条复现 Ch3 的 `mmr_diversity_beam_search` 数值，路径集合与多样性指标与 Ch3 存在差异属预期。保真依据为 online/offline parity 测试（`tests/kgqa/test_metaqa_end_to_end.py`）+ 答案 acc 对齐（0.9938 vs 0.9937）+ 分跳 sanity。

## 过程记录

- 首次全量评测全 0：根因是 `kgqa/cli/eval.py:_gold_strings` 的 `gold_key=="name"` 分支未把整数 gold id 经 `id2ent` 还原实体名（Plan1 预留分支，MetaQA 首次走到）。已修复并补单元测试与端到端 `hit1>0` 断言（commit `6c548d9`）。
