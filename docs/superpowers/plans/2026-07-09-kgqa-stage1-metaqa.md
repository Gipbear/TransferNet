# kgqa Stage 1 · Plan2 — MetaQA 端到端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 MetaQA_KB 接入已锁定的 kgqa 框架，复用单一 WebQSP 检索内核跑通「dump → 检索 → 分跳评测」端到端。

**Architecture:** 方案 C——只在三个策略口（KG 边来源 `kg/`、得分格式 `scores/`、数据适配器 `datasets/`）+ 模型封装 `models/` 上填 MetaQA 实现；`retrieve/engine.py`、`types.py`、`eval/` 不改动。MetaQA 模型 eval 返回不含 `hop_attn`，故 producer 用 **gold hop 合成 one-hot `hop_attn`** 喂给未改动的 engine（`argmax()+1==gold_hop`），与 Ch3「用 gold hop」一致。

**Tech Stack:** Python 3.12（conda `py312_t271_cuda`）、PyTorch 2.7、numpy、unittest。

## Global Constraints

- Python 3.12，本地环境 `py312_t271_cuda`；测试用 `python -m unittest`。
- **收敛单一内核**：MetaQA 复用已迁移的 `tail_blend`/`LogNorm` 内核，不给 engine 新增 method、不改 engine 公式。不逐条复现 Ch3；保真靠 online/offline parity + 答案 acc + 分跳 sanity。
- **先小后全量**：主体用「每跳 N 条」小子集打通 + 锁接口 + 测试；全量 39093 的 dump 与正式数字为末尾执行步（Task 7）。
- 不训练、不改 4 个 TransferNet 模块（仅只读其 model/data 做前向）、不动 WebQSP、不碰 MetaQA-Text/CWQ/SFT/agent。迁移期旧代码保留不删。
- 提交遵循 CLAUDE.md：Conventional Commits + 中文正文；`Co-Authored-By` 用 `git config user.name`/`user.email` 读取（禁止硬编码）；按文件名暂存。**仅在步骤明确到「Commit」时提交。**
- 分支 `kgqa-stage1-metaqa`（已建，spec commit `ef6e22d`）。
- 现有资产：ckpt `data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt`；输入 `data/input/MetaQA_KB/`（含 `vocab.json`、`test.pt`、`Msubj/Mobj/Mrel.npy`、`test.json`）。
- 模型超参（须匹配 ckpt）：`num_steps=3`、`dim_word=300`、`dim_hidden=1024`、`aux_hop=1`。

## 已核实的关键事实

- MetaQA_KB `model.py` forward：`model(questions, e_s_onehot)`（2 参）→ dict `{e_score, word_attns, rel_probs, ent_probs}`，**无 hop_attn**。`rel_probs`/`ent_probs` 为长度 3 的 list。
- `MetaQA_KB.data.DataLoader(vocab_json, question_pt, batch_size)`：读预处理 `.pt`（pickle 4 数组），batch = `(question, topic_entity, answer, hop)`；`loader.vocab` 有 `id2word/id2entity/id2relation/entity2id/relation2id`。question 是 token id，需 `id2word` 还原。
- KG 三元组：`Msubj/Mobj/Mrel.npy` 形状 `(Tsize, 2)`，第 1 列是 id；`triples = stack([Msubj[:,1], Mrel[:,1], Mobj[:,1]])`。**邻接已含反向边，不补 `_reverse`**。
- test.pt / test.json **按 hop 分块排序**（hop1: 9947 → hop2: 14872 → hop3: 14274，共 39093）→ 小子集必须**每跳取前 N 条**分层抽样才能覆盖 3-hop。
- 现有 kgqa 接口（复用，不改）：
  - `kgqa.scores.base.SampleScore(question, topic_ids, gold_ids, hop_attn, rel_probs, ent_indices, ent_scores, e_score_indices, e_score_values, sample_index=-1, hop=None, triples=None)`
  - `kgqa.scores.base.CacheMeta(dataset, split, id2ent, id2rel, num_samples, topk_entities=500, input_dir=None, qa_file=None, extra={})`、`ScoreBundle(meta, samples)`、`ScoreLoader`(ABC: `load(cache_path)->ScoreBundle`)
  - `kgqa.kg.global_kg.GlobalKG(valid_edges_dict)`、`.from_triples(triples)`、`.neighbors(node)`、`.valid_edges_dict`
  - `kgqa.types.QASample(question, topic_ids, gold_ids, sample_index=-1, hop=None, extra={})`、`MetricSpec(gold_key, group_by, answer_metrics, path_metrics)`
  - `kgqa.datasets.base.DatasetAdapter`(ABC: `load_qa/entity_name/kg_edge_source/score_loader/metric_spec`)
  - `kgqa.models.base.ScoreProducer`(ABC: `load_checkpoint(ckpt)`、`produce(input_dir, qa_file, *, split, batch_size, topk)->ScoreBundle`)
  - `utils.path_utils.build_valid_edges_dict(triples)->dict[int,list[(rel,tail)]]`、`utils.path_utils.filter_tensor(t, thr)->list[(idx,val)]`、`utils.misc.idx_to_one_hot(idx_tensor, size)`

---

## 文件结构（本 plan 落地范围）

```
kgqa/
├── kg/global_kg.py            # 改：+ GlobalKG.from_metaqa_npy
├── scores/metaqa.py           # 新：MetaQAScoreLoader
├── datasets/metaqa.py         # 新：MetaQAAdapter
├── datasets/registry.py       # 改：注册 "metaqa"
├── models/metaqa.py           # 新：MetaQAScoreProducer
└── cli/dump_scores.py         # 改：数据集分发 + hop 字段
tests/kgqa/
├── test_global_kg_metaqa.py       # Task 1
├── test_scores_metaqa.py          # Task 2
├── test_dataset_metaqa.py         # Task 3
├── test_models_metaqa.py          # Task 4（ckpt 存在才跑）
├── test_dump_metaqa.py            # Task 5（ckpt 存在才跑）
└── test_metaqa_end_to_end.py      # Task 6（ckpt 存在才跑）
```

---

### Task 1: GlobalKG.from_metaqa_npy（MetaQA KG 边来源）

**Files:**
- Modify: `kgqa/kg/global_kg.py`
- Test: `tests/kgqa/test_global_kg_metaqa.py`

**Interfaces:**
- Consumes: `utils.path_utils.build_valid_edges_dict`（已存在）、`GlobalKG.from_triples`（已存在）
- Produces: `GlobalKG.from_metaqa_npy(input_dir: str) -> GlobalKG`（读三个 npy → triples → `from_triples`，不补反向边）

- [x] **Step 1: 写失败测试** `tests/kgqa/test_global_kg_metaqa.py`

```python
import os
import tempfile
import unittest

import numpy as np

from kgqa.kg.global_kg import GlobalKG


class TestGlobalKGMetaQA(unittest.TestCase):
    def _write_npy_dir(self):
        d = tempfile.mkdtemp()
        # 两条三元组: (0)-[10]->(1), (1)-[11]->(2)；npy 形状 (T,2)，第 1 列是 id
        subj = np.array([[0, 0], [1, 1]])
        rel = np.array([[0, 10], [1, 11]])
        obj = np.array([[0, 1], [1, 2]])
        np.save(os.path.join(d, "Msubj.npy"), subj)
        np.save(os.path.join(d, "Mrel.npy"), rel)
        np.save(os.path.join(d, "Mobj.npy"), obj)
        return d

    def test_from_metaqa_npy_builds_edges(self):
        d = self._write_npy_dir()
        kg = GlobalKG.from_metaqa_npy(d)
        self.assertCountEqual(kg.neighbors(0), [(10, 1)])
        self.assertCountEqual(kg.neighbors(1), [(11, 2)])
        self.assertEqual(kg.neighbors(2), [])

    def test_no_reverse_edges_added(self):
        d = self._write_npy_dir()
        kg = GlobalKG.from_metaqa_npy(d)
        # 不应凭空生成 1->0 的反向边（MetaQA npy 已含所需双向边，不额外补）
        self.assertNotIn((10, 0), kg.neighbors(1))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_global_kg_metaqa -v`
Expected: FAIL（`AttributeError: type object 'GlobalKG' has no attribute 'from_metaqa_npy'`）

- [x] **Step 3: 写实现** — 在 `kgqa/kg/global_kg.py` 顶部 import 处加 `import numpy as np`（若无），在 `from_input_dir` 之后新增：

```python
    @classmethod
    def from_metaqa_npy(cls, input_dir: str) -> "GlobalKG":
        """从 MetaQA_KB 的 Msubj/Mobj/Mrel.npy 重建全局邻接表。

        逻辑迁移自 MetaQA_KB/predict.py：三个 npy 形状 (Tsize, 2)，第 1 列是
        entity/relation id，按行 zip 成 (subj, rel, obj)。MetaQA KG 已含反向边，
        不再补 _reverse。"""
        import numpy as np
        from pathlib import Path
        d = Path(input_dir)
        subj = np.load(d / "Msubj.npy")
        rel = np.load(d / "Mrel.npy")
        obj = np.load(d / "Mobj.npy")
        stacked = np.stack([subj[:, 1], rel[:, 1], obj[:, 1]], axis=1).tolist()
        triples = [[int(s), int(r), int(o)] for s, r, o in stacked]
        return cls.from_triples(triples)
```

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_global_kg_metaqa -v`
Expected: PASS（2 tests）

- [x] **Step 5: Commit**

```bash
git add kgqa/kg/global_kg.py tests/kgqa/test_global_kg_metaqa.py
git commit -m "$(cat <<'EOF'
feat(kgqa): GlobalKG 支持 MetaQA npy 邻接重建

- kgqa/kg/global_kg.py: 新增 from_metaqa_npy（读 Msubj/Mobj/Mrel.npy，不补反向边）
- tests/kgqa/test_global_kg_metaqa.py: MetaQA 邻接单测

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: MetaQAScoreLoader（读缓存恢复 hop）

**Files:**
- Create: `kgqa/scores/metaqa.py`
- Test: `tests/kgqa/test_scores_metaqa.py`

**Interfaces:**
- Consumes: `kgqa.scores.base.{CacheMeta, SampleScore, ScoreBundle, ScoreLoader}`；缓存 schema 由 `kgqa/cli/dump_scores.py:_bundle_to_cache` 写出（Task 5 会补 `hop` 字段）。
- Produces: `MetaQAScoreLoader().load(cache_path: str) -> ScoreBundle`（每条 `SampleScore.hop` 从缓存恢复）

- [x] **Step 1: 写失败测试** `tests/kgqa/test_scores_metaqa.py`

```python
import os
import tempfile
import unittest

import torch

from kgqa.scores.metaqa import MetaQAScoreLoader


class TestMetaQAScoreLoader(unittest.TestCase):
    def _write_cache(self):
        cache = {
            "version": 1,
            "meta": {"dataset": "MetaQA", "split": "test", "num_samples": 1,
                     "topk_entities": 500, "input_dir": "data/input/MetaQA_KB",
                     "qa_file": "data/input/MetaQA_KB/test.pt",
                     "id2ent": {0: "DUMMY", 1: "Movie A"}, "id2rel": {10: "starred_actors"}},
            "samples": [{
                "question": "what movie", "topic_ids": [1], "gold_ids": [1],
                "hop_attn": torch.tensor([1.0, 0.0, 0.0]),
                "rel_probs": [torch.tensor([0.0, 0.9]), torch.tensor([0.0, 0.0]),
                              torch.tensor([0.0, 0.0])],
                "ent_indices": [torch.tensor([1]), torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)],
                "ent_scores": [torch.tensor([0.8]), torch.tensor([]), torch.tensor([])],
                "e_score_indices": torch.tensor([1]),
                "e_score_values": torch.tensor([0.95]),
                "hop": 1,
            }],
        }
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(cache, path)
        return path

    def test_load_restores_hop(self):
        path = self._write_cache()
        try:
            bundle = MetaQAScoreLoader().load(path)
        finally:
            os.unlink(path)
        self.assertEqual(bundle.meta.dataset, "MetaQA")
        self.assertEqual(len(bundle.samples), 1)
        s = bundle.samples[0]
        self.assertEqual(s.hop, 1)
        self.assertEqual(s.sample_index, 0)
        self.assertEqual(s.gold_ids, [1])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_scores_metaqa -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.scores.metaqa'`）

- [x] **Step 3: 写实现** `kgqa/scores/metaqa.py`

```python
"""MetaQA 得分缓存加载：dump_scores 的 dict 缓存 → ScoreBundle（含 hop）。"""
from __future__ import annotations

import torch

from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle, ScoreLoader


class MetaQAScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        cache = torch.load(cache_path, weights_only=False)
        meta_d = cache["meta"]
        meta = CacheMeta(
            dataset=meta_d.get("dataset", "MetaQA"),
            split=meta_d.get("split", ""),
            id2ent=meta_d.get("id2ent", {}),
            id2rel=meta_d.get("id2rel", {}),
            num_samples=meta_d.get("num_samples", len(cache["samples"])),
            topk_entities=meta_d.get("topk_entities", 500),
            input_dir=meta_d.get("input_dir"),
            qa_file=meta_d.get("qa_file"),
        )
        samples = [
            SampleScore(
                question=s["question"],
                topic_ids=list(s["topic_ids"]),
                gold_ids=list(s["gold_ids"]),
                hop_attn=s["hop_attn"],
                rel_probs=s["rel_probs"],
                ent_indices=s["ent_indices"],
                ent_scores=s["ent_scores"],
                e_score_indices=s["e_score_indices"],
                e_score_values=s["e_score_values"],
                sample_index=i,
                hop=s.get("hop"),
                triples=s.get("triples"),
            )
            for i, s in enumerate(cache["samples"])
        ]
        return ScoreBundle(meta=meta, samples=samples)
```

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_scores_metaqa -v`
Expected: PASS（1 test）

- [x] **Step 5: Commit**

```bash
git add kgqa/scores/metaqa.py tests/kgqa/test_scores_metaqa.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 MetaQAScoreLoader（缓存恢复 hop）

- kgqa/scores/metaqa.py: 读 dict 缓存 → ScoreBundle，恢复每条 hop
- tests/kgqa/test_scores_metaqa.py: loader 单测

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: MetaQAAdapter + registry 注册

**Files:**
- Create: `kgqa/datasets/metaqa.py`
- Modify: `kgqa/datasets/registry.py`
- Test: `tests/kgqa/test_dataset_metaqa.py`

**Interfaces:**
- Consumes: `GlobalKG.from_metaqa_npy`（Task 1）、`MetaQAScoreLoader`（Task 2）、`kgqa.types.{QASample, MetricSpec}`、`kgqa.datasets.base.DatasetAdapter`
- Produces:
  - `MetaQAAdapter(input_dir="data/input/MetaQA_KB")`，属性 `name="metaqa"`、`max_hop=3`
  - `load_qa(path, limit=0) -> list[QASample]`（读 test.json：名字 topic/gold + hop）
  - `entity_name(entity_id) -> str`（恒等）、`kg_edge_source(sample=None) -> GlobalKG`、`score_loader() -> MetaQAScoreLoader`、`metric_spec() -> MetricSpec(gold_key="name", group_by="hop")`
  - registry 注册 `"metaqa"`

- [x] **Step 1: 写失败测试** `tests/kgqa/test_dataset_metaqa.py`

```python
import json
import os
import tempfile
import unittest

from kgqa.datasets.metaqa import MetaQAAdapter
from kgqa.datasets.registry import get_adapter
from kgqa.types import MetricSpec, QASample


class TestMetaQAAdapter(unittest.TestCase):
    def _write_test_json(self):
        data = [
            {"question": "what does E_S appear in", "topic_entity": "Grégoire Colin",
             "answers": ["Before the Rain"], "hop": 1},
            {"question": "who directed the movies", "topic_entity": "Joe",
             "answers": ["A", "B"], "hop": 3},
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return path

    def test_load_qa_parses_names_and_hop(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        path = self._write_test_json()
        try:
            samples = adapter.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], QASample)
        self.assertEqual(samples[0].topic_ids, ["Grégoire Colin"])
        self.assertEqual(samples[0].gold_ids, ["Before the Rain"])
        self.assertEqual(samples[0].hop, 1)
        self.assertEqual(samples[1].hop, 3)
        self.assertEqual(samples[1].gold_ids, ["A", "B"])

    def test_load_qa_limit(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        path = self._write_test_json()
        try:
            samples = adapter.load_qa(path, limit=1)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)

    def test_entity_name_identity(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        self.assertEqual(adapter.entity_name("Before the Rain"), "Before the Rain")

    def test_metric_spec_name_and_hop(self):
        adapter = MetaQAAdapter(input_dir="data/input/MetaQA_KB")
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "name")
        self.assertEqual(spec.group_by, "hop")
        self.assertEqual(adapter.max_hop, 3)

    def test_registry_returns_metaqa(self):
        adapter = get_adapter("metaqa", input_dir="data/input/MetaQA_KB")
        self.assertEqual(adapter.name, "metaqa")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_dataset_metaqa -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.datasets.metaqa'`）

- [x] **Step 3: 写实现** `kgqa/datasets/metaqa.py`

```python
"""MetaQA_KB 适配器（实体即名字、3-hop、分跳评测）。"""
from __future__ import annotations

import json

from kgqa.datasets.base import DatasetAdapter
from kgqa.kg.global_kg import GlobalKG
from kgqa.scores.base import ScoreLoader
from kgqa.scores.metaqa import MetaQAScoreLoader
from kgqa.types import MetricSpec, QASample


class MetaQAAdapter(DatasetAdapter):
    name = "metaqa"
    max_hop = 3

    def __init__(self, input_dir: str = "data/input/MetaQA_KB"):
        self.input_dir = input_dir
        self._kg: GlobalKG | None = None

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        samples: list[QASample] = []
        for item in data:
            topic = item.get("topic_entity")
            samples.append(QASample(
                question=item["question"],
                topic_ids=[topic] if topic else [],
                gold_ids=list(item.get("answers", [])),
                sample_index=len(samples),
                hop=item.get("hop"),
                extra={"topic_entity": topic},
            ))
            if limit and len(samples) >= limit:
                break
        return samples

    def entity_name(self, entity_id: str) -> str:
        return entity_id  # MetaQA 实体本身即名字

    def kg_edge_source(self, sample: QASample | None = None) -> GlobalKG:
        if self._kg is None:
            self._kg = GlobalKG.from_metaqa_npy(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return MetaQAScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="name", group_by="hop",
                          answer_metrics=True, path_metrics=True)
```

在 `kgqa/datasets/registry.py` 注册 MetaQA。当前内容：

```python
_REGISTRY: dict[str, type[DatasetAdapter]] = {"webqsp": WebQSPAdapter}
```

改为在文件顶部 import 后，把注册表补上 MetaQA：

```python
from kgqa.datasets.metaqa import MetaQAAdapter
from kgqa.datasets.webqsp import WebQSPAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "webqsp": WebQSPAdapter,
    "metaqa": MetaQAAdapter,
}
```

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_dataset_metaqa -v`
Expected: PASS（5 tests）

- [x] **Step 5: Commit**

```bash
git add kgqa/datasets/metaqa.py kgqa/datasets/registry.py tests/kgqa/test_dataset_metaqa.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 MetaQAAdapter 并注册 metaqa

- kgqa/datasets/metaqa.py: 读 test.json（名字 topic/gold + hop）、恒等 entity_name、npy KG、name+hop metric_spec
- kgqa/datasets/registry.py: 注册 "metaqa"
- tests/kgqa/test_dataset_metaqa.py: 适配器单测

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: MetaQAScoreProducer（前向产 SampleScore + 合成 hop_attn + 分层抽样）

**Files:**
- Create: `kgqa/models/metaqa.py`
- Test: `tests/kgqa/test_models_metaqa.py`（ckpt 存在才跑）

**Interfaces:**
- Consumes: `MetaQA_KB.data.DataLoader`、`MetaQA_KB.model.TransferNet`、`utils.misc.idx_to_one_hot`、`utils.path_utils.filter_tensor`、`kgqa.scores.base.{CacheMeta, SampleScore, ScoreBundle}`、`kgqa.models.base.ScoreProducer`
- Produces:
  - `MetaQAScoreProducer(num_steps=3, dim_word=300, dim_hidden=1024, aux_hop=1, per_hop_limit=0)`
  - `load_checkpoint(ckpt_path)`、`produce(input_dir, qa_file, *, split="test", batch_size=64, topk=500) -> ScoreBundle`
  - `qa_file` = MetaQA 预处理 `.pt`（如 `data/input/MetaQA_KB/test.pt`）；`per_hop_limit>0` 时每跳只保留前 `per_hop_limit` 条（覆盖 3-hop）
  - 每条 `SampleScore.hop_attn` = gold hop 的 one-hot（长度 num_steps），`SampleScore.hop` = gold hop

- [x] **Step 1: 写失败测试** `tests/kgqa/test_models_metaqa.py`

```python
import os
import unittest

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(TEST_PT), "ckpt/数据缺失，跳过")
class TestMetaQAScoreProducer(unittest.TestCase):
    def test_produce_small_stratified(self):
        from kgqa.models.metaqa import MetaQAScoreProducer
        producer = MetaQAScoreProducer(per_hop_limit=2)
        producer.load_checkpoint(CKPT)
        bundle = producer.produce(INPUT_DIR, TEST_PT, split="test", batch_size=64, topk=500)
        # 每跳 2 条 → 覆盖三个 hop
        hops = sorted({s.hop for s in bundle.samples})
        self.assertEqual(hops, [1, 2, 3])
        s = bundle.samples[0]
        # 合成 hop_attn：argmax()+1 == gold hop
        self.assertEqual(int(s.hop_attn.argmax().item()) + 1, s.hop)
        self.assertEqual(len(s.rel_probs), 3)
        self.assertGreater(s.e_score_values.numel(), 0)
        self.assertIsInstance(s.question, str)
        self.assertTrue(bundle.meta.id2ent)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_models_metaqa -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.models.metaqa'`；若无 ckpt 则 skip）

- [x] **Step 3: 写实现** `kgqa/models/metaqa.py`

```python
"""MetaQA_KB 在线得分生产（前向逻辑迁移自 MetaQA_KB/predict.py）。"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch

from utils.misc import idx_to_one_hot
from utils.path_utils import filter_tensor
from MetaQA_KB.data import DataLoader
from MetaQA_KB.model import TransferNet
from kgqa.models.base import ScoreProducer
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


class MetaQAScoreProducer(ScoreProducer):
    def __init__(self, num_steps: int = 3, dim_word: int = 300, dim_hidden: int = 1024,
                 aux_hop: int = 1, per_hop_limit: int = 0):
        self.num_steps = num_steps
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.aux_hop = aux_hop
        self.per_hop_limit = per_hop_limit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 64, topk: int = 500) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        import os
        vocab_json = os.path.join(input_dir, "vocab.json")
        loader = DataLoader(vocab_json, qa_file, batch_size)
        vocab = loader.vocab
        num_ent = len(vocab["entity2id"])

        # TransferNet.__init__(args, dim_word, dim_hidden, vocab)；args 需 num_steps/aux_hop/input_dir
        args = SimpleNamespace(num_steps=self.num_steps, aux_hop=self.aux_hop,
                               input_dir=input_dir)
        model = TransferNet(args, self.dim_word, self.dim_hidden, vocab)
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"), strict=False)
        model = model.to(self.device)
        model.kg.Msubj = model.kg.Msubj.to(self.device)
        model.kg.Mobj = model.kg.Mobj.to(self.device)
        model.kg.Mrel = model.kg.Mrel.to(self.device)
        model.eval()

        kept = defaultdict(int)
        samples: list[SampleScore] = []
        with torch.no_grad():
            for batch in loader:
                questions, topic_entities, answers, hops = batch
                topic_onehot = idx_to_one_hot(topic_entities, num_ent).to(self.device)
                answers_onehot = idx_to_one_hot(answers, num_ent)
                answers_onehot[:, 0] = 0  # 排除 DUMMY_ENTITY
                outputs = model(questions.to(self.device), topic_onehot)
                e_score = outputs["e_score"].cpu()
                rel_probs = [t.cpu() for t in outputs["rel_probs"]]
                ent_probs = [t.cpu() for t in outputs["ent_probs"]]
                hops_list = hops.tolist()
                for i in range(e_score.shape[0]):
                    hop = int(hops_list[i])
                    if self.per_hop_limit and kept[hop] >= self.per_hop_limit:
                        continue
                    kept[hop] += 1
                    topic_ids = [int(x) for (x, _) in filter_tensor(topic_onehot[i].cpu(), 1)]
                    gold_ids = answers_onehot[i].gt(0.5).nonzero().squeeze(1).tolist()
                    question_str = " ".join(
                        vocab["id2word"][w] for w in questions[i].cpu().tolist() if w > 0)
                    ent_idx_hop, ent_sc_hop = [], []
                    for t in range(self.num_steps):
                        vec = ent_probs[t][i]
                        k = min(topk, vec.shape[0])
                        vals, idxs = vec.topk(k)
                        mask = vals > 0
                        ent_idx_hop.append(idxs[mask])
                        ent_sc_hop.append(vals[mask])
                    ev = e_score[i]
                    k = min(topk, ev.shape[0])
                    evals, eidxs = ev.topk(k)
                    emask = evals > 0
                    hop_attn = torch.zeros(self.num_steps)
                    hop_attn[hop - 1] = 1.0
                    samples.append(SampleScore(
                        question=question_str,
                        topic_ids=topic_ids, gold_ids=[int(g) for g in gold_ids],
                        hop_attn=hop_attn,
                        rel_probs=[rel_probs[t][i].clone() for t in range(self.num_steps)],
                        ent_indices=ent_idx_hop, ent_scores=ent_sc_hop,
                        e_score_indices=eidxs[emask], e_score_values=evals[emask],
                        sample_index=len(samples), hop=hop,
                    ))
                if self.per_hop_limit and all(
                        kept[h] >= self.per_hop_limit for h in range(1, self.num_steps + 1)):
                    break  # hop 分块有序，三跳配额都满即可提前停

        meta = CacheMeta(dataset="MetaQA", split=split, id2ent=vocab["id2entity"],
                         id2rel=vocab["id2relation"], num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
```

> 说明：`hops` 分块有序（hop1→hop2→hop3），故三跳配额都满时可 `break`。因 hop3 在末段，小 `per_hop_limit` 仍需前向到 hop3 起始区，但模型极小，代价可接受。

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_models_metaqa -v`
Expected: PASS（1 test；无 ckpt 则 skip）

- [x] **Step 5: Commit**

```bash
git add kgqa/models/metaqa.py tests/kgqa/test_models_metaqa.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 MetaQAScoreProducer（前向 + 合成 hop_attn + 分层抽样）

- kgqa/models/metaqa.py: MetaQA_KB 前向产 SampleScore，gold hop 合成 one-hot hop_attn，per_hop_limit 分层小子集
- tests/kgqa/test_models_metaqa.py: producer 单测（ckpt 存在才跑）

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: dump_scores CLI 数据集分发 + hop 字段

**Files:**
- Modify: `kgqa/cli/dump_scores.py`
- Test: `tests/kgqa/test_dump_metaqa.py`（ckpt 存在才跑）

**Interfaces:**
- Consumes: `MetaQAScoreProducer`（Task 4）、`WebQSPScoreProducer`（已存在）
- Produces: `python -m kgqa.cli.dump_scores --dataset metaqa ...` 写出含 `hop` 的缓存；新增 `--per_hop_limit`。`_bundle_to_cache` 的 samples 增加 `hop`（`s.hop is not None` 才写，兼容 WebQSP）。

- [x] **Step 1: 写失败测试** `tests/kgqa/test_dump_metaqa.py`

```python
import os
import tempfile
import unittest

import torch

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


class TestDumpBundleHopField(unittest.TestCase):
    def test_bundle_to_cache_writes_hop(self):
        # 纯函数测试，无需 ckpt
        from kgqa.cli.dump_scores import _bundle_to_cache
        from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle
        s = SampleScore(question="q", topic_ids=[1], gold_ids=[1],
                        hop_attn=torch.tensor([1.0, 0.0, 0.0]),
                        rel_probs=[torch.tensor([0.0])], ent_indices=[torch.tensor([1])],
                        ent_scores=[torch.tensor([0.5])],
                        e_score_indices=torch.tensor([1]), e_score_values=torch.tensor([0.9]),
                        sample_index=0, hop=2)
        meta = CacheMeta(dataset="MetaQA", split="test", id2ent={}, id2rel={}, num_samples=1)
        cache = _bundle_to_cache(ScoreBundle(meta=meta, samples=[s]))
        self.assertEqual(cache["samples"][0]["hop"], 2)


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(TEST_PT), "ckpt/数据缺失，跳过")
class TestDumpMetaQAEndToEnd(unittest.TestCase):
    def test_dump_metaqa_small(self):
        from kgqa.cli.dump_scores import main
        out = os.path.join(tempfile.mkdtemp(), "metaqa_small.pt")
        main(["--dataset", "metaqa", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
              "--qa_file", TEST_PT, "--output", out, "--per_hop_limit", "2",
              "--batch_size", "64"])
        cache = torch.load(out, weights_only=False)
        self.assertEqual(cache["meta"]["dataset"], "MetaQA")
        self.assertTrue(all("hop" in s for s in cache["samples"]))
        self.assertEqual(sorted({s["hop"] for s in cache["samples"]}), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_dump_metaqa -v`
Expected: FAIL（`_bundle_to_cache` 不写 hop → 第一个断言失败；metaqa 分发未实现）

- [x] **Step 3: 写实现** — 改 `kgqa/cli/dump_scores.py`：

（a）`_bundle_to_cache` 的 sample dict 增加 hop（在 `**({"triples": ...})` 那行同级追加）：

```python
        "samples": [{
            "question": s.question, "topic_ids": s.topic_ids, "gold_ids": s.gold_ids,
            "hop_attn": s.hop_attn, "rel_probs": s.rel_probs,
            "ent_indices": s.ent_indices, "ent_scores": s.ent_scores,
            "e_score_indices": s.e_score_indices, "e_score_values": s.e_score_values,
            **({"hop": s.hop} if s.hop is not None else {}),
            **({"triples": s.triples} if s.triples is not None else {}),
        } for s in bundle.samples],
```

（b）`build_parser` 增加 `--per_hop_limit`：

```python
    p.add_argument("--per_hop_limit", type=int, default=0,
                   help="MetaQA 每跳保留前 N 条（分层小子集），0=全量")
```

（c）`main` 用数据集分发替换「仅支持 webqsp」硬拒绝：

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dataset == "webqsp":
        from kgqa.models.webqsp import WebQSPScoreProducer
        producer = WebQSPScoreProducer()
    elif args.dataset == "metaqa":
        from kgqa.models.metaqa import MetaQAScoreProducer
        producer = MetaQAScoreProducer(per_hop_limit=args.per_hop_limit)
    else:
        raise SystemExit(f"未支持的 dump 数据集: {args.dataset}")
    producer.load_checkpoint(args.ckpt)
    bundle = producer.produce(args.input_dir, args.qa_file, split=args.split,
                              batch_size=args.batch_size, topk=args.topk)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(_bundle_to_cache(bundle), args.output)
    print(f"[INFO] dump 完成 {len(bundle.samples)} 条 → {args.output}", flush=True)
```

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_dump_metaqa -v`
Expected: PASS（无 ckpt 时 hop 字段单测通过、end-to-end skip）

同时确认 WebQSP dump 未回归：

Run: `python -m unittest tests.kgqa.test_cli -v`
Expected: PASS（WebQSP dump 缓存兼容，hop 字段缺省不写）

- [x] **Step 5: Commit**

```bash
git add kgqa/cli/dump_scores.py tests/kgqa/test_dump_metaqa.py
git commit -m "$(cat <<'EOF'
feat(kgqa): dump_scores 支持 metaqa 分发与 hop 字段

- kgqa/cli/dump_scores.py: 数据集分发（webqsp/metaqa）、缓存增 hop 字段（兼容 WebQSP）、新增 --per_hop_limit
- tests/kgqa/test_dump_metaqa.py: hop 字段单测 + metaqa 小子集 dump（ckpt 存在才跑）

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 端到端集成（offline 分跳评测 + 3-hop 验证 + online/offline parity）

**Files:**
- Test: `tests/kgqa/test_metaqa_end_to_end.py`（ckpt 存在才跑）

**Interfaces:**
- Consumes: `kgqa.cli.dump_scores.main`（Task 5）、`kgqa.datasets.registry.get_adapter`、`kgqa.retrieve.backends.offline.OfflineBackend`、`kgqa.retrieve.backends.online.OnlineBackend`、`kgqa.models.metaqa.MetaQAScoreProducer`、`kgqa.cli.eval`（复用 Plan1）
- Produces: 无新代码，仅集成验证——证明 MetaQA 走通「dump→offline 检索→分跳评测」且 online/offline parity；显式验证 engine 在 3-hop 输入下产出路径。

- [x] **Step 1: 写失败测试** `tests/kgqa/test_metaqa_end_to_end.py`

```python
import os
import tempfile
import unittest

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(TEST_PT), "ckpt/数据缺失，跳过")
class TestMetaQAEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgqa.cli.dump_scores import main as dump_main
        cls.cache = os.path.join(tempfile.mkdtemp(), "metaqa_small.pt")
        dump_main(["--dataset", "metaqa", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
                   "--qa_file", TEST_PT, "--output", cls.cache,
                   "--per_hop_limit", "3", "--batch_size", "64"])

    def _offline(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        adapter = get_adapter("metaqa", input_dir=INPUT_DIR)
        return OfflineBackend(adapter, cache_path=self.cache)

    def test_offline_retrieves_paths_all_hops(self):
        backend = self._offline()
        results = backend.retrieve_all()
        self.assertTrue(results)
        # 覆盖 3 个 hop，且 3-hop 样本能产出路径（验证 engine 在 3 跳下工作）
        by_hop = {}
        for r, s in zip(results, backend.bundle.samples):
            by_hop.setdefault(s.hop, []).append(r)
        self.assertEqual(sorted(by_hop), [1, 2, 3])
        self.assertTrue(any(r.paths for r in by_hop[3]))

    def test_answer_eval_by_hop(self):
        from kgqa.cli.eval import _gold_strings
        from kgqa.eval.answer_eval import answer_record, answer_summary
        backend = self._offline()
        adapter = backend.adapter
        spec = adapter.metric_spec()
        id2ent = backend.bundle.meta.id2ent
        results = backend.retrieve_all()
        records = []
        for r, s in zip(results, backend.bundle.samples):
            gold = _gold_strings(s, adapter, id2ent, spec.gold_key)
            records.append(answer_record(pred=list(r.prediction.keys()),
                                         gold=sorted(gold), hop=s.hop))
        summary = answer_summary(records, spec)
        self.assertEqual(set(summary["by_hop"]), {"1", "2", "3"})
        self.assertIn("hit1", summary["overall"])

    def test_online_offline_parity_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.models.metaqa import MetaQAScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = get_adapter("metaqa", input_dir=INPUT_DIR)
        online = OnlineBackend(adapter, MetaQAScoreProducer(per_hop_limit=3),
                               ckpt_path=CKPT, input_dir=INPUT_DIR, qa_file=TEST_PT,
                               batch_size=64, limit=0)
        off = self._offline()
        for idx in range(3):
            ro = online.retrieve(idx)
            rf = off.retrieve(idx)
            self.assertEqual([p["path"] for p in ro.paths], [p["path"] for p in rf.paths])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行确认失败/或直接跑**

Run: `python -m unittest tests.kgqa.test_metaqa_end_to_end -v`
Expected: 有 ckpt 时应 PASS（前置任务都实现后）；若 `test_offline_retrieves_paths_all_hops` 因 engine 在 3-hop 下有隐含 2-hop 假设而失败，见 Step 3。

- [x] **Step 3: （条件）修引擎 3-hop 边界**

若 Step 2 中 3-hop 检索无路径或报错，定位 `kgqa/retrieve/engine.py:_method_hop_numbers` 及 `search_path_candidates` 是否假设 `hop<=2`。**仅修边界、不改打分公式**（数值红线）。若无问题，跳过本步。记录结论到 commit message。

- [x] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_metaqa_end_to_end -v`
Expected: PASS（3 tests；无 ckpt 则 skip）

全量回归确认无破坏：

Run: `python -m unittest discover -s tests/kgqa -p 'test*.py'`
Expected: OK（WebQSP 全部照旧 + MetaQA 新增；联网/ckpt 相关按需 skip）

- [x] **Step 5: Commit**

```bash
git add tests/kgqa/test_metaqa_end_to_end.py
# 若 Step 3 改了 engine，一并 add kgqa/retrieve/engine.py
git commit -m "$(cat <<'EOF'
test(kgqa): MetaQA 端到端集成（分跳评测 + 3-hop + parity）

- tests/kgqa/test_metaqa_end_to_end.py: dump→offline 分跳评测、3-hop 路径、online/offline parity（ckpt 存在才跑）

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: 全量 dump 与正式 overall/by_hop 数字（执行步，非 TDD）

**Files:**
- 无代码变更；产出物写入 `data/output/MetaQA_KB/`（gitignored）与文档。

**Interfaces:**
- Consumes: 前六个 Task 的全部实现。

- [x] **Step 1: 全量 dump（39093 条，per_hop_limit=0）**

Run:
```bash
python -m kgqa.cli.dump_scores --dataset metaqa \
  --ckpt data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt \
  --input_dir data/input/MetaQA_KB \
  --qa_file data/input/MetaQA_KB/test.pt \
  --output data/output/MetaQA_KB/score_cache/metaqa_test_full.pt \
  --batch_size 64
```
Expected: `[INFO] dump 完成 39093 条 → ...`

- [x] **Step 2: 全量 offline 评测（overall + by_hop）**

Run:
```bash
python -m kgqa.cli.eval --dataset metaqa --backend offline \
  --input_dir data/input/MetaQA_KB \
  --cache data/output/MetaQA_KB/score_cache/metaqa_test_full.pt \
  --qa_file data/input/MetaQA_KB/test.pt \
  --summary data/output/MetaQA_KB/eval/metaqa_test_summary.json
```
Expected: stdout 打印 overall；`metaqa_test_summary.json` 含 `answer.by_hop{"1","2","3"}` 与 `path.by_hop`。

- [x] **Step 3: 记录数字与内核收敛差异**

在 `docs/experiments_*` 或本 plan 末尾追加：全量 overall hit1/F1/EM + 分跳三档；并注明「单一 WebQSP 内核收敛，未逐条复现 Ch3 的 `mmr_diversity_beam_search` 数值，差异属预期」。

- [x] **Step 4: Commit（仅文档/summary 指针，不提交大缓存）**

```bash
git add docs/  # 仅新增/改动的实验记录文档
git commit -m "$(cat <<'EOF'
docs(kgqa): 记录 MetaQA 全量分跳评测数字

- MetaQA test 39093 全量 overall + 1/2/3-hop 指标；说明单一内核收敛与 Ch3 差异

Co-Authored-By: <git config user.name> <<git config user.email>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage：**
- 组件 1 GlobalKG.from_metaqa_npy → Task 1 ✓
- 组件 2 MetaQAScoreProducer（前向+hop+分层）→ Task 4 ✓
- 组件 3 MetaQAScoreLoader（恢复 hop）→ Task 2 ✓
- 组件 4 MetaQAAdapter + registry → Task 3 ✓
- 组件 5 dump_scores 分发 + hop 字段 → Task 5 ✓
- 保真红线：online/offline parity → Task 6 `test_online_offline_parity_first3` ✓；答案 sanity（hit1）→ Task 6 `test_answer_eval_by_hop` + Task 7 全量 ✓；分跳 sanity → Task 6/Task 7 ✓；engine 3-hop 适配点 → Task 6 Step 3 ✓
- 先小后全量：Task 1-6 用 per_hop_limit 小子集；Task 7 全量 ✓
- 范围外（不改 engine 公式/WebQSP/训练）：约束已在 Global Constraints，Task 6 Step 3 限定「仅修边界不改公式」✓

**Placeholder scan：** 无 TBD/TODO；每个代码步给出完整代码；Task 6 Step 3 为条件分支但给出明确定位对象与红线，非占位。

**Type consistency：**
- `MetaQAScoreProducer(per_hop_limit=...)` 构造参数在 Task 4 定义、Task 5/6 一致使用 ✓
- `GlobalKG.from_metaqa_npy(input_dir)` Task 1 定义、Task 3 使用 ✓
- `MetaQAScoreLoader().load()` Task 2 定义、Task 3 `score_loader()` 返回、Task 6 offline 使用 ✓
- `SampleScore.hop` Task 4 写入、Task 5 缓存、Task 2 恢复、Task 6 评测分组 ✓
- `_bundle_to_cache` 增 hop 字段与 loader 读 `s.get("hop")` 一致 ✓
- `MetricSpec(gold_key="name", group_by="hop")` 与 `kgqa.cli.eval._gold_strings` 的 name 分支（Plan1 已实现）一致 ✓
