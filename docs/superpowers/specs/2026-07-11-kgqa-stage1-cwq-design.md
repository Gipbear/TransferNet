# kgqa Stage 1 · Plan3 — CWQ 端到端 Design

> 父设计：`docs/superpowers/specs/2026-07-08-unified-kgqa-retrieval-eval-design.md`（统一 KGQA 框架整体）。
> 前置：Plan1（WebQSP，PR #2 merge `245ead8`）、Plan2（MetaQA，PR #3 merge `5e1c1ed`）均已合并，接口已锁定。本 spec 只覆盖 Plan3 的 CWQ 接入。

## 目标

把 **CompWebQ(CWQ)** 接入已锁定的 kgqa 框架，跑通「dump → 检索 → 评测」端到端。核心差异点是**逐样本子图**：每条样本自带 `subgraph.tuples`，KG 边来源从进程级全局图变为逐样本构建。engine 公式、`types.py`、`eval/` 不动。

## 关键决策（brainstorm 已确认）

1. **子图内嵌 dump 缓存**：producer 把每条样本的 subgraph tuples 写入 `SampleScore.triples`（Plan1 预留字段，dump/loader 侧分支已存在），缓存自包含，评测不再读 358MB 的 `test_simple.json`。缓存体积估算 300-400MB（与 MetaQA 718MB 同量级）。
2. **方案 A——backend 逐样本传参**：`OfflineBackend`/`OnlineBackend` 从「`__init__` 取一次 `kg_edge_source()`」改为「每次检索调 `adapter.kg_edge_source(sample)`」；WebQSP/MetaQA 适配器忽略参数返回进程级缓存的全局 KG，**零行为变化**。engine 不动（只读 `edge_source.valid_edges_dict`，per-sample GlobalKG 天然满足）。
3. **沿用单一检索内核**：复用 `tail_blend`/`LogNorm`，不逐条复现 Ch3（CWQ 旧路径由 `CompWebQ/predict.py` 的 `mmr_diversity_beam_search` 另一内核生成，仅作 schema 参照）。检索指标可能偏离旧数值，属预期。
4. **先小后全量**：主体用小子集打通端到端 + 锁接口 + 全套测试；全量 3531 条正式数字列为 plan 末尾执行步。

## 现状与数据形态（已核实）

- **ckpt**：`data/ckpt/CWQ/model-29-0.4206.pt`（acc 0.4206），训练参数 `num_ways=1, num_steps=2, bert-base-cased, rev=False`（`data/ckpt/CWQ/log.txt` 确认）。
- **模型**：`CompWebQ/model.py` forward 返回 `{e_score, word_attns, rel_probs, ent_probs, hop_attn}`。`num_ways=1` 时 `torch.prod` 是恒等 → 与 WebQSP **同构**（2-hop、原生 hop_attn、sigmoid rel_probs），**无需合成 hop_attn，无 multi-way 特殊处理**。
- **QA 数据**：`data/input/CWQ/test_simple.json` — JSONL 3531 条，每条 `{id, question, answers[{kb_id, text}], entities[int], subgraph{tuples[[s,r,o]], entities}}`。子图均值约 3400 三元组/条（前 200 条抽样，max 约 1 万）；answers 平均 1.9 个。
- **词表**：`entities.txt`（MID，约 250 万）、`relations.txt`（6432）；tuples 中的 id 即两文件的行号（全局 id 空间）。答案 `kb_id` 是 MID → `gold_key="mid"`，与 WebQSP 同口径。
- **`rev=False`** → 子图边不补反向，`build_valid_edges_dict(tuples)` 直接用（与 `CompWebQ/predict.py:64` 现行做法一致）。
- **无 hop 标签**：simple.json 不带分跳信息 → `group_by=None`，hop 由 `hop_attn.argmax` 推断（engine 现行为）。
- **坑 1**：`CompWebQ/data.py:load_data` 会 tokenize train(2.6GB)+dev+test 三个 split 并整体 pickle 缓存 → producer 不走它。
- **坑 2**：`CompWebQ/data.py` DataLoader 跳过 `len(triples)==0` 的样本（所有 split 生效）→ 有效条数以 producer 实际产出为准；`sample_index` 在 producer 内连续编号，gold 直接从 batch 读（同 WebQSP），QA/score 天然对齐。`CWQAdapter.load_qa` 应用同一跳过规则保持对齐。
- **坑 3（顺手修）**：`kgqa/cli/retrieve.py` online 分支硬编码 `WebQSPScoreProducer`（dump_scores 已修过同类问题）。

## 架构

沿用方案 C。engine 消费 `SampleScore` + `KGEdgeSource` + `id2ent/id2rel`，不认识数据集；CWQ 差异全部收进策略口 + 适配器 + backend 的逐样本分发。

### 新增/改动组件

1. **`kgqa/retrieve/backends/offline.py` + `online.py`（改）** — 去掉 `__init__` 里的 `self.edge_source`，每次检索改调 `self.adapter.kg_edge_source(sample)`。`kgqa/datasets/base.py` 的 sample 参数注解放宽为鸭子类型（CWQ 用 `.triples`，其余忽略）。

2. **`kgqa/models/cwq.py`（新）** — `CWQScoreProducer(ScoreProducer)`：
   - **不走 `load_data`**：直接读 `entities.txt`/`relations.txt` 建 `ent2id/rel2id`，仅对 `qa_file` 构造 `CompWebQ.data.DataLoader`。
   - 前向 mirror `WebQSPScoreProducer`（`topk` 截断 `ent_probs`/`e_score`、`> 0` mask）。
   - **附 triples**：`batch[3][i].tolist()` 写入 `SampleScore.triples`。
   - `CacheMeta.id2ent/id2rel` = invert 词表（MID / 关系字符串）。

3. **`kgqa/scores/cwq.py`（新）** — `CWQScoreLoader(ScoreLoader)`：`torch.load` 缓存 → `ScoreBundle`，恢复每条 `triples`（dump 侧 `s.get("triples")` 分支已存在）。

4. **`kgqa/datasets/cwq.py`（新）** — `CWQAdapter(DatasetAdapter)`：
   - `name="cwq"`、`max_hop=2`。
   - `load_qa(path, limit)`：读 JSONL（`question`、`answers[].kb_id`、`entities`），跳过空子图样本保持与 producer 对齐。
   - `entity_name` 恒等（MID 口径，同 WebQSP 不在 eval 链路做名字映射）。
   - `kg_edge_source(sample)` → `GlobalKG.from_triples(sample.triples)`；`sample is None` 时抛错（CWQ 无全局图）。
   - `score_loader()` → `CWQScoreLoader()`。
   - `metric_spec()` → `MetricSpec(gold_key="mid", group_by=None, answer_metrics=True, path_metrics=True)`。
   - registry 注册 `"cwq"`。

5. **`kgqa/cli/dump_scores.py`（改）** — 分发表加 `cwq→CWQScoreProducer`。

6. **`kgqa/cli/retrieve.py`（改）** — online 分支 producer 硬编码改为按 dataset 分发（webqsp/metaqa/cwq），与 dump_scores 同款。

### 数据流

```
test_simple.json ──CWQScoreProducer(仅 test 词表+loader)──▶ SampleScore(int ids, triples) ──dump──▶ .pt 缓存
  .pt ──CWQScoreLoader──▶ ScoreBundle(triples 在样本内)
    每样本: GlobalKG.from_triples(s.triples) + SampleScore ──engine.retrieve_one(tail_blend)──▶ RetrieveResult
      eval.py: gold(int→MID via id2ent), group_by=None ──▶ summary{overall}
```

## 保真 / 验证（红线）

单一内核收敛下，不做逐条 Ch3 复现，改为：

1. **online/offline parity**：小子集上双后端检索路径逐条一致（有 ckpt 才跑）。
2. **答案 sanity**：小子集/全量 hit1 与 ckpt acc 0.4206 量级吻合（不要求相等）。
3. **WebQSP/MetaQA 零回归**：backend 逐样本分发改动后 `tests/kgqa` 全量测试通过（离线回归锁免 ckpt）。
4. 端到端评测补 `hit1 > 0` 断言（Plan2 全 0 坑的防回归惯例）。

## 测试（TDD，mirror Plan2）

- `tests/kgqa/test_dataset_cwq.py` — `load_qa` 解析 MID gold/topic、空子图跳过；`metric_spec` 为 mid + 无分组；`max_hop==2`；registry 返回 cwq；`kg_edge_source(None)` 抛错。
- `tests/kgqa/test_scores_cwq.py` — loader 恢复 triples（缓存存在才跑）。
- `tests/kgqa/test_backends_per_sample.py` — backend 逐样本 edge_source 分发；WebQSP/MetaQA 路径仍复用缓存全局 KG（mock adapter，免数据）。
- `tests/kgqa/test_dump_cwq.py` — dump 小子集产缓存含 triples（有 ckpt/数据才跑）。
- `tests/kgqa/test_cwq_end_to_end.py` — parity + `hit1>0` + 条数 sanity（有 ckpt/数据才跑）。

## 范围外（不做）

- 不改 engine 公式、不动 `types.py`/`eval/`、不改 WebQSP/MetaQA 适配器行为。
- 不碰 MetaQA-Text、SFT、agent；不训练。
- 旧 `CompWebQ/predict.py` 保留不删。
- `torch.load weights_only` 加固留作独立 chore，不混入本 plan。

## 文件结构（Plan3 落地范围）

```
kgqa/
├── retrieve/backends/offline.py   # 改：逐样本 kg_edge_source(sample)
├── retrieve/backends/online.py    # 改：同上
├── datasets/base.py               # 改：sample 参数注解放宽
├── models/cwq.py                  # 新：CWQScoreProducer
├── scores/cwq.py                  # 新：CWQScoreLoader
├── datasets/cwq.py                # 新：CWQAdapter
├── datasets/registry.py           # 改：注册 cwq
├── cli/dump_scores.py             # 改：cwq 分发
└── cli/retrieve.py                # 改：online producer 按 dataset 分发
tests/kgqa/
├── test_dataset_cwq.py            # 新
├── test_scores_cwq.py             # 新（缓存存在才跑）
├── test_backends_per_sample.py    # 新（mock，免数据）
├── test_dump_cwq.py               # 新（ckpt/数据存在才跑）
└── test_cwq_end_to_end.py         # 新（ckpt/数据存在才跑）
```
