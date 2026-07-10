# kgqa Stage 1 · Plan2 — MetaQA 端到端 Design

> 父设计：`docs/superpowers/specs/2026-07-08-unified-kgqa-retrieval-eval-design.md`（统一 KGQA 框架整体）。
> 前置：Plan1（WebQSP 端到端）已实现+合并（PR #2, merge `245ead8`），接口已锁定。本 spec 只覆盖 Plan2 的 MetaQA 接入。

## 目标

把 **MetaQA_KB** 接入已锁定的 kgqa 框架，跑通「dump → 检索 → 分跳评测」端到端。仅在方案 C 的三个策略口（KG 边来源、得分格式、数据适配器）上填 MetaQA 实现，`retrieve/engine.py`、`types.py`、`eval/` 不改动。

## 关键决策（brainstorm 已确认）

1. **收敛到单一 WebQSP 检索内核**：MetaQA 直接复用已迁移的 `tail_blend` / `LogNorm` 内核，**不给引擎新增第二条检索 method**。因此**不逐条复现 Ch3**（Ch3 的 MetaQA 路径由 `MetaQA_KB/predict.py` 的 `mmr_diversity_beam_search` 另一套内核生成，与本框架不同源）。首要目标是统一工程框架，MetaQA 检索指标可能偏离 Ch3 数值，属预期。
2. **先小后全量**：Plan2 主体用「每跳约几百条」的固定小子集打通端到端 + 锁接口 + 全套测试；**全量 39093 test 的 dump 与正式 overall/by_hop 数字**列为 plan 末尾的执行步（或单独跑），不阻塞接口验收。

## 现状与数据形态（已核实）

- **模型**：`MetaQA_KB/model.py` forward 返回 `{e_score, hop_attn, rel_probs, ent_probs}`，与 WebQSP **同形** → engine 的 `SampleScore` 消费方式可直接复用。3 跳（`num_hop=3`），含跨跳 cycle prevention。
- **ckpt**：`data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt`（acc 0.9937）。
- **QA 数据**：`data/input/MetaQA_KB/test.json` — JSON 列表，每条 `{question, topic_entity, answers[], hop∈{1,2,3}}`，实体是**名字**（如 `"Grégoire Colin"` / `"Before the Rain"`），test 39093 条。
- **KG**：`data/input/MetaQA_KB/{Msubj,Mobj,Mrel}.npy`（稀疏 COO，三列即 triple 索引）+ `vocab.json`（`entity2id/relation2id/word2id/topic_entity`）。**邻接已含反向边**（路径中出现 `starred_actors_inv`）→ 构建 `valid_edges_dict` 时**不再手动补 `_reverse`**。
- **无 MetaQA 检索基线可逐条对齐**：`scripts/offline_path_search.py` 为 WebQSP 专用；`data/output/MetaQA_KB/grid_search/paths/*.jsonl` 由另一套内核生成，仅作 schema 参照（`mmr_reason_paths`/`log_score`/`hop` 与 kgqa 输出一致）。

## 架构

沿用 Plan1 的方案 C。engine 消费 `SampleScore` + `KGEdgeSource` + `id2ent/id2rel`，不认识数据集；MetaQA 差异全部收进策略口 + 适配器。

### 新增/改动组件

1. **`kgqa/kg/global_kg.py`（改）** — 新增 `GlobalKG.from_metaqa_npy(input_dir)`：读 `Msubj/Mobj/Mrel.npy` 三列 + `vocab.json`，zip 成 `(subj_id, rel_id, obj_id)` triples → `build_valid_edges_dict`。**不补反向边**（npy 已含双向）。既有 `from_input_dir`（WebQSP，补 `_reverse`）不动。

2. **`kgqa/models/metaqa.py`（新）** — `MetaQAScoreProducer(ScoreProducer)`：加载 `MetaQA_KB` 模型前向产 `SampleScore`，mirror `WebQSPScoreProducer`。要点：
   - `topk` 截断 `ent_probs` / `e_score`（`> 0` mask），与 WebQSP 一致。
   - **附 hop**：从 test.json 逐条读 `hop` 写入 `SampleScore.hop`（分跳评测依赖）。
   - `CacheMeta.id2ent/id2rel` = vocab 的 `id2entity/id2relation`（名字）。

3. **`kgqa/scores/metaqa.py`（新）** — `MetaQAScoreLoader(ScoreLoader)`：`torch.load` 缓存 `.pt` → `ScoreBundle`，恢复每条 `hop`。

4. **`kgqa/datasets/metaqa.py`（新）** — `MetaQAAdapter(DatasetAdapter)`：
   - `name="metaqa"`、`max_hop=3`。
   - `load_qa(path, limit)`：读 test.json（JSON 列表），`topic_ids=[topic_entity]`、`gold_ids=answers`（名字）、`hop=条目 hop`、`sample_index`。
   - `entity_name` 恒等（实体本身即名字）。
   - `kg_edge_source` → `GlobalKG.from_metaqa_npy(input_dir)`（进程内缓存一次）。
   - `score_loader()` → `MetaQAScoreLoader()`。
   - `metric_spec()` → `MetricSpec(gold_key="name", group_by="hop", answer_metrics=True, path_metrics=True)`。
   - registry 注册 `"metaqa"`。

5. **`kgqa/cli/dump_scores.py`（改）** — 数据集分发（producer registry：`webqsp→WebQSPScoreProducer`、`metaqa→MetaQAScoreProducer`），去掉「仅支持 webqsp」硬拒绝。`_bundle_to_cache` 的 samples 增加 `hop` 字段（`s.hop is not None` 才写，向后兼容 WebQSP 缓存）。

### 数据流

```
test.json ──load_qa──▶ QASample(name topic/gold, hop)
  MetaQA_KB.model ──produce──▶ SampleScore(int ids, hop) ──dump──▶ .pt 缓存
    .pt ──MetaQAScoreLoader──▶ ScoreBundle
      SampleScore + GlobalKG(npy) ──engine.retrieve_one(tail_blend)──▶ RetrieveResult
        eval.py: gold(int→name via id2ent) + group_by="hop" ──▶ {overall, by_hop{1,2,3}}
```

## 保真 / 验证（红线）

单一内核收敛下，不做逐条 Ch3 复现，改为三重 sanity：

1. **online/offline parity**：小子集上，online 后端（实时前向）与 offline 后端（读缓存）检索路径逐条一致（Plan1 同款，仅在有 ckpt 时跑）。
2. **答案 sanity**：小子集 hit1 与 MetaQA_KB 模型 acc（~0.9937）量级吻合（不要求相等，检索/预测口径差异允许）。
3. **分跳 sanity**：`summary.json` 出 `{overall, by_hop{"1","2","3"}}`，各 hop 条数与子集抽样一致。
4. **引擎 3-hop 适配点**：显式验证 `_method_hop_numbers("tail_blend", hop, 3)` 在 3 跳输入下行为正确（WebQSP 只验过 2 跳）。若发现 3 跳下内核有隐含 2-hop 假设，作为 Plan2 内的 bug 记录并最小修正（不改公式，仅修边界）。

## 测试（TDD，mirror Plan1）

- `tests/kgqa/test_global_kg_metaqa.py` — `from_metaqa_npy` 邻接正确、不含多余反向边。
- `tests/kgqa/test_scores_metaqa.py` — loader 恢复 hop（缓存存在才跑）。
- `tests/kgqa/test_dataset_metaqa.py` — `load_qa` 解析名字 topic/gold/hop；`metric_spec` 为 name + hop；`max_hop==3`；registry 返回 metaqa。
- `tests/kgqa/test_dump_metaqa.py` — dump 小子集产缓存含 hop（有 ckpt 才跑）。
- `tests/kgqa/test_backend_parity_metaqa.py` — online/offline parity（有 ckpt 才跑）。
- 复用现有 `test_answer_eval` 的 group_by 分跳覆盖（已存在，无需重写）。

## 范围外（不做）

- 不给 engine 新增第二套检索 method、不改 engine 公式、不动 WebQSP 代码。
- 不碰 MetaQA-Text、CWQ、SFT、agent。
- 不训练、不改 4 个 TransferNet 模块（只读它们的 model/data 做前向）。
- 迁移期旧代码（`MetaQA_KB/predict.py`、`offline_path_search.py`）保留不删。

## 文件结构（Plan2 落地范围）

```
kgqa/
├── kg/global_kg.py            # 改：+ from_metaqa_npy
├── models/metaqa.py           # 新：MetaQAScoreProducer
├── scores/metaqa.py           # 新：MetaQAScoreLoader
├── datasets/metaqa.py         # 新：MetaQAAdapter
├── datasets/registry.py       # 改：注册 metaqa
└── cli/dump_scores.py         # 改：数据集分发 + hop 字段
tests/kgqa/
├── test_global_kg_metaqa.py   # 新
├── test_scores_metaqa.py      # 新
├── test_dataset_metaqa.py     # 新
├── test_dump_metaqa.py        # 新（ckpt 存在才跑）
└── test_backend_parity_metaqa.py  # 新（ckpt 存在才跑）
```
