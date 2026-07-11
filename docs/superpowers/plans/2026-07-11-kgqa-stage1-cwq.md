# kgqa Stage1 Plan3 — CWQ 端到端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 把 CompWebQ(CWQ) 接入 kgqa 统一框架，跑通「dump → 检索 → 评测」端到端，核心是逐样本子图的 KG 边来源分发。

**Architecture:** 沿用方案 C（共享 engine + 策略口 + 薄适配器）。两个 backend 从「init 取一次全局 edge_source」改为「每次检索调 `adapter.kg_edge_source(sample)`」；CWQ 的子图 tuples 由 producer 写入 `SampleScore.triples`（Plan1 预留字段）随缓存自包含；`CWQAdapter.kg_edge_source(sample)` 用 `GlobalKG.from_triples` 逐样本现建邻接。engine 公式、`types.py`、`eval/` 不动。

**Tech Stack:** PyTorch、transformers(bert-base-cased)、unittest；复用 `CompWebQ/{model,data}.py` 做前向（不改动）。

**Spec:** `docs/superpowers/specs/2026-07-11-kgqa-stage1-cwq-design.md`

## Global Constraints

- 本地执行统一用 conda 环境 `py312_t271_cuda`（先激活再跑 `python`，不用 `conda run`）。
- engine 检索内核是数值红线：不改 `kgqa/retrieve/engine.py` 任何公式。
- 不动 `kgqa/types.py`、`kgqa/eval/`；不改 WebQSP/MetaQA 适配器行为（仅放宽注解）。
- 不训练；不碰 `CompWebQ/{model,data,predict}.py`（只读复用）。
- ckpt 固定 `data/ckpt/CWQ/model-29-0.4206.pt`（`num_ways=1, num_steps=2, bert-base-cased, rev=False`）。
- 提交按文件名暂存（禁 `git add -A`），消息用中文 Conventional Commits + HEREDOC，Co-Authored-By 按 `git config user.name/user.email` 读取（当前为 `jsh-smi-wsl <1099048889@qq.com>`）+ `Claude Fable 5 <noreply@anthropic.com>`。
- 每个 Task 结束跑 `python -m unittest discover -s tests/kgqa -p 'test*.py'` 确认零回归。

---

### Task 1: backend 逐样本 edge_source 分发

**Files:**
- Modify: `kgqa/retrieve/backends/offline.py`
- Modify: `kgqa/retrieve/backends/online.py`
- Modify: `kgqa/datasets/base.py:22`（注解放宽）
- Modify: `kgqa/datasets/webqsp.py:51`、`kgqa/datasets/metaqa.py:42`（注解同步放宽，行为不变）
- Test: `tests/kgqa/test_backends_per_sample.py`

**Interfaces:**
- Consumes: `DatasetAdapter.kg_edge_source(sample=None)`（Plan1 已有可选参数）；`engine.retrieve_one(sample, edge_source, id2ent, id2rel, **params)`。
- Produces: `OfflineBackend`/`OnlineBackend` 每次检索调用 `self.adapter.kg_edge_source(sample)`（sample 为 `SampleScore`，鸭子类型）；后续 CWQAdapter 依赖此分发。

- [x] **Step 1: 写失败测试**

创建 `tests/kgqa/test_backends_per_sample.py`：

```python
import unittest
from unittest import mock

import torch

from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


def _fake_bundle(n=2):
    samples = [
        SampleScore(
            question=f"q{i}", topic_ids=[0], gold_ids=[1],
            hop_attn=torch.tensor([1.0, 0.0]),
            rel_probs=[torch.zeros(3), torch.zeros(3)],
            ent_indices=[torch.tensor([1]), torch.tensor([1])],
            ent_scores=[torch.tensor([0.5]), torch.tensor([0.5])],
            e_score_indices=torch.tensor([1]),
            e_score_values=torch.tensor([0.9]),
            sample_index=i,
            triples=[[0, 0, 1]],
        )
        for i in range(n)
    ]
    meta = CacheMeta(dataset="fake", split="test", id2ent={}, id2rel={}, num_samples=n)
    return ScoreBundle(meta=meta, samples=samples)


class _RecordingAdapter:
    """记录 kg_edge_source 收到的 sample；load 返回内存 bundle。"""

    def __init__(self, bundle):
        self._bundle = bundle
        self.calls = []

    def score_loader(self):
        outer = self

        class _Loader:
            def load(self, path):
                return outer._bundle

        return _Loader()

    def kg_edge_source(self, sample=None):
        self.calls.append(sample)
        return f"kg-for-{getattr(sample, 'sample_index', None)}"


class _FakeProducer:
    def load_checkpoint(self, ckpt_path):
        pass

    def produce(self, input_dir, qa_file, *, split="test", batch_size=16, topk=500):
        return _fake_bundle()


class TestOfflinePerSample(unittest.TestCase):
    def test_each_sample_gets_own_edge_source(self):
        from kgqa.retrieve.backends.offline import OfflineBackend
        bundle = _fake_bundle()
        adapter = _RecordingAdapter(bundle)
        backend = OfflineBackend(adapter, cache_path="unused")
        # 旧实现在 __init__ 里就调 kg_edge_source()（无 sample）——新实现不应有该调用
        self.assertEqual(adapter.calls, [])
        with mock.patch("kgqa.retrieve.backends.offline.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve_all()
        self.assertEqual(adapter.calls, bundle.samples)
        got = [c.args[1] for c in m.call_args_list]
        self.assertEqual(got, ["kg-for-0", "kg-for-1"])

    def test_retrieve_single_passes_sample(self):
        from kgqa.retrieve.backends.offline import OfflineBackend
        bundle = _fake_bundle()
        adapter = _RecordingAdapter(bundle)
        backend = OfflineBackend(adapter, cache_path="unused")
        with mock.patch("kgqa.retrieve.backends.offline.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve(1)
        self.assertEqual(adapter.calls, [bundle.samples[1]])


class TestOnlinePerSample(unittest.TestCase):
    def test_each_sample_gets_own_edge_source(self):
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = _RecordingAdapter(_fake_bundle())
        backend = OnlineBackend(adapter, _FakeProducer(), ckpt_path="x",
                                input_dir="d", qa_file="q")
        self.assertEqual(adapter.calls, [])
        with mock.patch("kgqa.retrieve.backends.online.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve_all()
        self.assertEqual([getattr(s, "sample_index", None) for s in adapter.calls], [0, 1])
        got = [c.args[1] for c in m.call_args_list]
        self.assertEqual(got, ["kg-for-0", "kg-for-1"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.kgqa.test_backends_per_sample -v`
Expected: FAIL——`test_each_sample_gets_own_edge_source` 断言 `adapter.calls == []` 失败（旧实现 `__init__` 里调了 `kg_edge_source()`，calls 为 `[None]`）。

- [x] **Step 3: 改 offline.py**

`kgqa/retrieve/backends/offline.py` 全文替换为：

```python
"""离线后端：读得分缓存 → engine（edge source 逐样本分发）。"""
from __future__ import annotations

from kgqa.datasets.base import DatasetAdapter
from kgqa.retrieve import engine
from kgqa.retrieve.backends.base import RetrieveBackend, RetrieveParams
from kgqa.types import RetrieveResult


class OfflineBackend(RetrieveBackend):
    def __init__(self, adapter: DatasetAdapter, cache_path: str):
        self.adapter = adapter
        self.bundle = adapter.score_loader().load(cache_path)

    def _one(self, sample, params: dict) -> RetrieveResult:
        return engine.retrieve_one(
            sample, self.adapter.kg_edge_source(sample),
            self.bundle.meta.id2ent, self.bundle.meta.id2rel, **params,
        )

    def retrieve(self, sample_index: int, **params) -> RetrieveResult:
        merged = {**RetrieveParams().as_kwargs(), **params}
        return self._one(self.bundle.samples[sample_index], merged)

    def retrieve_all(self, *, limit: int = 0, **params) -> list[RetrieveResult]:
        merged = {**RetrieveParams().as_kwargs(), **params}
        samples = self.bundle.samples[:limit] if limit else self.bundle.samples
        return [self._one(s, merged) for s in samples]
```

- [x] **Step 4: 改 online.py**

`kgqa/retrieve/backends/online.py` 全文替换为：

```python
"""在线后端：ScoreProducer 实时前向 → 同一 engine（edge source 逐样本分发）。"""
from __future__ import annotations

from kgqa.datasets.base import DatasetAdapter
from kgqa.models.base import ScoreProducer
from kgqa.retrieve import engine
from kgqa.retrieve.backends.base import RetrieveParams


class OnlineBackend:
    def __init__(self, adapter: DatasetAdapter, producer: ScoreProducer, *,
                 ckpt_path: str, input_dir: str, qa_file: str,
                 split: str = "test", batch_size: int = 16, topk: int = 500, limit: int = 0):
        producer.load_checkpoint(ckpt_path)
        self.adapter = adapter
        self.bundle = producer.produce(input_dir, qa_file, split=split,
                                       batch_size=batch_size, topk=topk)
        if limit:
            self.bundle.samples = self.bundle.samples[:limit]

    def _one(self, sample, params: dict):
        return engine.retrieve_one(
            sample, self.adapter.kg_edge_source(sample),
            self.bundle.meta.id2ent, self.bundle.meta.id2rel, **params,
        )

    def retrieve(self, sample_index: int, **params):
        merged = {**RetrieveParams().as_kwargs(), **params}
        return self._one(self.bundle.samples[sample_index], merged)

    def retrieve_all(self, *, limit: int = 0, **params):
        merged = {**RetrieveParams().as_kwargs(), **params}
        samples = self.bundle.samples[:limit] if limit else self.bundle.samples
        return [self._one(s, merged) for s in samples]
```

- [x] **Step 5: 放宽注解**

`kgqa/datasets/base.py` 中 `kg_edge_source` 改为（QASample import 保留，`load_qa` 仍用）：

```python
    @abstractmethod
    def kg_edge_source(self, sample=None) -> KGEdgeSource:
        """sample 为鸭子类型：逐样本子图数据集（CWQ）传带 .triples 的 SampleScore；
        全局图数据集（WebQSP/MetaQA）忽略该参数。"""
```

`kgqa/datasets/webqsp.py:51` 与 `kgqa/datasets/metaqa.py:42` 的签名同步改为 `def kg_edge_source(self, sample=None) -> GlobalKG:`（实现不变）。

- [x] **Step 6: 跑测试确认通过 + 全量零回归**

Run: `python -m unittest tests.kgqa.test_backends_per_sample -v`
Expected: PASS（4 个测试）
Run: `python -m unittest discover -s tests/kgqa -p 'test*.py'`
Expected: 全部 PASS（WebQSP/MetaQA 离线回归锁不依赖 ckpt，必须全绿）

- [x] **Step 7: Commit**

```bash
git add tests/kgqa/test_backends_per_sample.py kgqa/retrieve/backends/offline.py kgqa/retrieve/backends/online.py kgqa/datasets/base.py kgqa/datasets/webqsp.py kgqa/datasets/metaqa.py
git commit -m "$(cat <<'EOF'
refactor(kgqa): backend 改为逐样本 kg_edge_source 分发

- retrieve/backends/{offline,online}.py: 去掉 init 期全局 edge_source，每次检索传 sample
- datasets/{base,webqsp,metaqa}.py: kg_edge_source 注解放宽为鸭子类型，行为不变
- tests/kgqa/test_backends_per_sample.py: mock 适配器验证逐样本分发（免数据）

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: CWQScoreLoader

**Files:**
- Create: `kgqa/scores/cwq.py`
- Test: `tests/kgqa/test_scores_cwq.py`

**Interfaces:**
- Consumes: `kgqa/scores/base.py` 的 `SampleScore/CacheMeta/ScoreBundle/ScoreLoader`；dump 缓存 dict 格式（`cli/dump_scores._bundle_to_cache`，triples 分支已存在）。
- Produces: `CWQScoreLoader().load(cache_path) -> ScoreBundle`，每条 sample 恢复 `triples`；Task 3 的 `CWQAdapter.score_loader()` 返回它。

- [x] **Step 1: 写失败测试**

创建 `tests/kgqa/test_scores_cwq.py`（合成缓存，免真实数据）：

```python
import os
import tempfile
import unittest

import torch


def _write_fake_cache(path):
    cache = {
        "version": 1,
        "meta": {"dataset": "CWQ", "split": "test", "num_samples": 1,
                 "topk_entities": 500, "input_dir": "in", "qa_file": "qa",
                 "id2ent": {0: "m.0a", 1: "m.0b"}, "id2rel": {0: "r.loc"}},
        "samples": [{
            "question": "who?", "topic_ids": [0], "gold_ids": [1],
            "hop_attn": torch.tensor([1.0, 0.0]),
            "rel_probs": [torch.zeros(2), torch.zeros(2)],
            "ent_indices": [torch.tensor([1]), torch.tensor([1])],
            "ent_scores": [torch.tensor([0.5]), torch.tensor([0.5])],
            "e_score_indices": torch.tensor([1]),
            "e_score_values": torch.tensor([0.9]),
            "triples": [[0, 0, 1]],
        }],
    }
    torch.save(cache, path)


class TestCWQScoreLoader(unittest.TestCase):
    def test_load_restores_triples_and_meta(self):
        from kgqa.scores.cwq import CWQScoreLoader
        path = os.path.join(tempfile.mkdtemp(), "fake_cwq.pt")
        _write_fake_cache(path)
        bundle = CWQScoreLoader().load(path)
        self.assertEqual(bundle.meta.dataset, "CWQ")
        self.assertEqual(len(bundle.samples), 1)
        s = bundle.samples[0]
        self.assertEqual(s.triples, [[0, 0, 1]])
        self.assertIsNone(s.hop)
        self.assertEqual(s.sample_index, 0)
        self.assertEqual(s.topic_ids, [0])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.kgqa.test_scores_cwq -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'kgqa.scores.cwq'`

- [x] **Step 3: 写实现**

创建 `kgqa/scores/cwq.py`：

```python
"""CWQ 得分缓存加载：dump_scores 的 dict 缓存 → ScoreBundle（含逐样本 triples）。"""
from __future__ import annotations

import torch

from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle, ScoreLoader


class CWQScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        cache = torch.load(cache_path, weights_only=False)
        meta_d = cache["meta"]
        meta = CacheMeta(
            dataset=meta_d.get("dataset", "CWQ"),
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

- [x] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.kgqa.test_scores_cwq -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add kgqa/scores/cwq.py tests/kgqa/test_scores_cwq.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 CWQScoreLoader（缓存恢复逐样本 triples）

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: CWQAdapter + registry 注册

**Files:**
- Create: `kgqa/datasets/cwq.py`
- Modify: `kgqa/datasets/registry.py`
- Test: `tests/kgqa/test_dataset_cwq.py`

**Interfaces:**
- Consumes: `GlobalKG.from_triples(triples)`（已有）；`CWQScoreLoader`（Task 2）；`MetricSpec/QASample`。
- Produces: `CWQAdapter`——`name="cwq"`、`max_hop=2`、`load_qa(path, limit)`、`kg_edge_source(sample)`（sample 需带 `.triples`，None/缺 triples 抛 `ValueError`）、`metric_spec()`；registry `get_adapter("cwq", input_dir=...)`。

- [x] **Step 1: 写失败测试**

创建 `tests/kgqa/test_dataset_cwq.py`：

```python
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from kgqa.types import MetricSpec, QASample


def _write_test_jsonl():
    rows = [
        {"id": "WebQTest-1", "question": "who is A?",
         "answers": [{"kb_id": "m.0b", "text": "B"}], "entities": [0],
         "subgraph": {"tuples": [[0, 0, 1]], "entities": [0, 1]}},
        {"id": "WebQTest-2", "question": "empty subgraph, should skip",
         "answers": [{"kb_id": "m.0c", "text": "C"}], "entities": [2],
         "subgraph": {"tuples": [], "entities": []}},
        {"id": "WebQTest-3", "question": "who is D?",
         "answers": [{"kb_id": "m.0d", "text": "D"}, {"kb_id": "m.0e", "text": "E"}],
         "entities": [3], "subgraph": {"tuples": [[3, 1, 4]], "entities": [3, 4]}},
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


class TestCWQAdapter(unittest.TestCase):
    def _adapter(self):
        from kgqa.datasets.cwq import CWQAdapter
        return CWQAdapter(input_dir="data/input/CWQ")

    def test_load_qa_parses_and_skips_empty_subgraph(self):
        path = _write_test_jsonl()
        try:
            samples = self._adapter().load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 2)  # 空子图样本被跳过（对齐 CompWebQ DataLoader）
        self.assertIsInstance(samples[0], QASample)
        self.assertEqual(samples[0].topic_ids, [0])
        self.assertEqual(samples[0].gold_ids, ["m.0b"])
        self.assertEqual(samples[1].gold_ids, ["m.0d", "m.0e"])
        self.assertEqual(samples[1].sample_index, 1)

    def test_load_qa_limit(self):
        path = _write_test_jsonl()
        try:
            samples = self._adapter().load_qa(path, limit=1)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)

    def test_kg_edge_source_builds_per_sample(self):
        adapter = self._adapter()
        kg = adapter.kg_edge_source(SimpleNamespace(triples=[[0, 0, 1], [1, 1, 2]]))
        self.assertEqual(kg.neighbors(0), [(0, 1)])
        self.assertEqual(kg.neighbors(1), [(1, 2)])

    def test_kg_edge_source_requires_sample_triples(self):
        adapter = self._adapter()
        with self.assertRaises(ValueError):
            adapter.kg_edge_source(None)
        with self.assertRaises(ValueError):
            adapter.kg_edge_source(SimpleNamespace(triples=None))

    def test_metric_spec_mid_no_group(self):
        adapter = self._adapter()
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)
        self.assertEqual(adapter.max_hop, 2)
        self.assertEqual(adapter.entity_name("m.0b"), "m.0b")

    def test_registry_returns_cwq(self):
        from kgqa.datasets.registry import get_adapter
        adapter = get_adapter("cwq", input_dir="data/input/CWQ")
        self.assertEqual(adapter.name, "cwq")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.kgqa.test_dataset_cwq -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'kgqa.datasets.cwq'`

- [x] **Step 3: 写实现**

创建 `kgqa/datasets/cwq.py`：

```python
"""CWQ 适配器（MID 口径、2-hop、逐样本子图）。"""
from __future__ import annotations

import json

from kgqa.datasets.base import DatasetAdapter
from kgqa.kg.global_kg import GlobalKG
from kgqa.scores.base import ScoreLoader
from kgqa.scores.cwq import CWQScoreLoader
from kgqa.types import MetricSpec, QASample


class CWQAdapter(DatasetAdapter):
    name = "cwq"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/CWQ"):
        self.input_dir = input_dir

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        samples: list[QASample] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("subgraph", {}).get("tuples"):
                    continue  # 与 CompWebQ DataLoader 跳过空子图的规则对齐
                samples.append(QASample(
                    question=item["question"].strip(),
                    topic_ids=[int(e) for e in item.get("entities", [])],
                    gold_ids=[a["kb_id"] for a in item.get("answers", [])],
                    sample_index=len(samples),
                    extra={"id": item.get("id")},
                ))
                if limit and len(samples) >= limit:
                    break
        return samples

    def entity_name(self, entity_id: str) -> str:
        return entity_id  # MID 口径，同 WebQSP 不在 eval 链路做名字映射

    def kg_edge_source(self, sample=None) -> GlobalKG:
        triples = getattr(sample, "triples", None)
        if triples is None:
            raise ValueError("CWQ 为逐样本子图，kg_edge_source 需要带 triples 的 sample")
        return GlobalKG.from_triples(triples)

    def score_loader(self) -> ScoreLoader:
        return CWQScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None,
                          answer_metrics=True, path_metrics=True)
```

`kgqa/datasets/registry.py` 加 import 与注册项：

```python
"""数据集适配器注册表。"""
from __future__ import annotations

from kgqa.datasets.base import DatasetAdapter
from kgqa.datasets.cwq import CWQAdapter
from kgqa.datasets.metaqa import MetaQAAdapter
from kgqa.datasets.webqsp import WebQSPAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "webqsp": WebQSPAdapter,
    "metaqa": MetaQAAdapter,
    "cwq": CWQAdapter,
}
```

（`register_adapter`/`get_adapter` 不变。）

- [x] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.kgqa.test_dataset_cwq -v`
Expected: PASS（6 个测试）

- [x] **Step 5: Commit**

```bash
git add kgqa/datasets/cwq.py kgqa/datasets/registry.py tests/kgqa/test_dataset_cwq.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 CWQAdapter 并注册 cwq

- datasets/cwq.py: MID 口径 load_qa（空子图跳过对齐 DataLoader）、逐样本 kg_edge_source
- datasets/registry.py: 注册 cwq

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: CWQScoreProducer

**Files:**
- Create: `kgqa/models/cwq.py`
- Test: `tests/kgqa/test_models_cwq.py`

**Interfaces:**
- Consumes: `CompWebQ.data.DataLoader(fn, bert_name, ent2id, rel2id, batch_size)`（batch = `[topic_onehot, question, answer_onehot, triples(list of LongTensor), entity_range]`）；`CompWebQ.model.TransferNet(args, ent2id, rel2id)`（args 需 `bert_name/num_steps/num_ways`）；`utils.path_utils.filter_tensor`、`utils.misc.{batch_device, invert_dict}`。
- Produces: `CWQScoreProducer(bert_name="bert-base-cased", num_steps=2, num_ways=1, limit=0)`，`produce(...) -> ScoreBundle`，每条 `SampleScore` 带 `triples`；模块级辅助 `_read_vocab(path)`、`_valid_lines(qa_file, limit)`（Task 5 dump 分发与 Task 6 端到端依赖 producer）。

- [x] **Step 1: 写失败测试（纯函数部分，免数据）**

创建 `tests/kgqa/test_models_cwq.py`：

```python
import json
import os
import tempfile
import unittest


class TestCWQProducerHelpers(unittest.TestCase):
    def test_read_vocab_line_order(self):
        from kgqa.models.cwq import _read_vocab
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("m.0a\nm.0b\n")
        try:
            vocab = _read_vocab(path)
        finally:
            os.unlink(path)
        self.assertEqual(vocab, {"m.0a": 0, "m.0b": 1})

    def test_valid_lines_skips_empty_subgraph_and_limits(self):
        from kgqa.models.cwq import _valid_lines
        rows = [
            {"question": "q1", "subgraph": {"tuples": [[0, 0, 1]]}},
            {"question": "q2", "subgraph": {"tuples": []}},
            {"question": "q3", "subgraph": {"tuples": [[1, 0, 2]]}},
            {"question": "q4", "subgraph": {"tuples": [[2, 0, 3]]}},
        ]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        try:
            all_lines = _valid_lines(path)
            limited = _valid_lines(path, limit=2)
        finally:
            os.unlink(path)
        self.assertEqual([json.loads(l)["question"] for l in all_lines], ["q1", "q3", "q4"])
        self.assertEqual([json.loads(l)["question"] for l in limited], ["q1", "q3"])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.kgqa.test_models_cwq -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'kgqa.models.cwq'`

- [x] **Step 3: 写实现**

创建 `kgqa/models/cwq.py`：

```python
"""CWQ 在线得分生产（前向逻辑迁移自 CompWebQ/predict.py，子图逐样本内嵌）。

不走 CompWebQ.data.load_data（它会 tokenize train 2.6GB 并整体 pickle），
直接读 entities.txt/relations.txt 建词表、仅对 qa_file 构造 DataLoader。
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import torch

from utils.misc import batch_device, invert_dict
from utils.path_utils import filter_tensor
from CompWebQ.data import DataLoader
from CompWebQ.model import TransferNet
from kgqa.models.base import ScoreProducer
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


def _read_vocab(path: str) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            vocab[line.strip()] = len(vocab)
    return vocab


def _valid_lines(qa_file: str, limit: int = 0) -> list[str]:
    """取非空子图样本的原始行（与 CompWebQ DataLoader 跳过空子图的规则对齐）。"""
    lines: list[str] = []
    with open(qa_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if not json.loads(line).get("subgraph", {}).get("tuples"):
                continue
            lines.append(line)
            if limit and len(lines) >= limit:
                break
    return lines


class CWQScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "bert-base-cased", num_steps: int = 2,
                 num_ways: int = 1, limit: int = 0):
        self.bert_name = bert_name
        self.num_steps = num_steps
        self.num_ways = num_ways
        self.limit = limit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        ent2id = _read_vocab(os.path.join(input_dir, "entities.txt"))
        rel2id = _read_vocab(os.path.join(input_dir, "relations.txt"))

        lines = _valid_lines(qa_file, self.limit)
        raw_questions = [json.loads(l)["question"].strip() for l in lines]
        if self.limit:
            # 小子集截断成临时文件，DataLoader 免读全量 358MB
            fd, qa_path = tempfile.mkstemp(suffix=".jsonl")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
        else:
            qa_path = qa_file
        try:
            loader = DataLoader(qa_path, self.bert_name, ent2id, rel2id, batch_size)
        finally:
            if self.limit:
                os.unlink(qa_path)

        args = SimpleNamespace(bert_name=self.bert_name, num_steps=self.num_steps,
                               num_ways=self.num_ways)
        model = TransferNet(args, ent2id, rel2id)
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"), strict=False)
        model = model.to(self.device)
        model.eval()

        samples: list[SampleScore] = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(*batch_device(batch, self.device))
                e_score = outputs["e_score"].cpu()
                hop_attn = outputs["hop_attn"].cpu()
                rel_probs = [t.cpu() for t in outputs["rel_probs"]]
                ent_probs = [t.cpu() for t in outputs["ent_probs"]]
                num_steps = len(rel_probs)
                for i in range(e_score.shape[0]):
                    topic_ids = [int(x) for (x, _) in filter_tensor(batch[0][i], 1)]
                    gold_ids = [int(x) for (x, _) in filter_tensor(batch[2][i], 0.5)]
                    ent_idx_hop, ent_sc_hop = [], []
                    for t in range(num_steps):
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
                    samples.append(SampleScore(
                        question=raw_questions[len(samples)],
                        topic_ids=topic_ids, gold_ids=gold_ids,
                        hop_attn=hop_attn[i].clone(),
                        rel_probs=[rel_probs[t][i].clone() for t in range(num_steps)],
                        ent_indices=ent_idx_hop, ent_scores=ent_sc_hop,
                        e_score_indices=eidxs[emask], e_score_values=evals[emask],
                        sample_index=len(samples),
                        triples=batch[3][i].tolist(),
                    ))
        meta = CacheMeta(dataset="CWQ", split=split, id2ent=invert_dict(ent2id),
                         id2rel=invert_dict(rel2id), num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
```

注意：若 BERT 加载因代理报错，参考 Plan1 经验设置 `NO_PROXY` / `HF_HUB_OFFLINE=1`（`bert-base-cased` 已有本地缓存，训练时用过）。

- [x] **Step 4: 跑测试确认通过**

Run: `python -m unittest tests.kgqa.test_models_cwq -v`
Expected: PASS（2 个测试；producer 前向部分由 Task 6 端到端覆盖）

- [x] **Step 5: Commit**

```bash
git add kgqa/models/cwq.py tests/kgqa/test_models_cwq.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 CWQScoreProducer（仅 test 词表 + 子图内嵌 + limit 小子集）

- models/cwq.py: 绕开 load_data 全量 tokenize，triples 写入 SampleScore，limit 截断临时文件
- tests/kgqa/test_models_cwq.py: _read_vocab/_valid_lines 纯函数单测

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: CLI 分发（dump_scores + retrieve online producer）

**Files:**
- Modify: `kgqa/cli/dump_scores.py`
- Modify: `kgqa/cli/retrieve.py`
- Test: `tests/kgqa/test_cli_dispatch.py`

**Interfaces:**
- Consumes: `CWQScoreProducer(limit=...)`（Task 4）；已有 `WebQSPScoreProducer`/`MetaQAScoreProducer`。
- Produces: `dump_scores --dataset cwq [--limit N]`；`retrieve.py` 的 `_make_producer(dataset)`（online 分支按 dataset 分发，Task 6 parity 依赖）。

- [x] **Step 1: 写失败测试**

创建 `tests/kgqa/test_cli_dispatch.py`：

```python
import unittest


class TestRetrieveProducerDispatch(unittest.TestCase):
    def test_make_producer_by_dataset(self):
        from kgqa.cli.retrieve import _make_producer
        from kgqa.models.cwq import CWQScoreProducer
        from kgqa.models.metaqa import MetaQAScoreProducer
        from kgqa.models.webqsp import WebQSPScoreProducer
        self.assertIsInstance(_make_producer("webqsp"), WebQSPScoreProducer)
        self.assertIsInstance(_make_producer("metaqa"), MetaQAScoreProducer)
        self.assertIsInstance(_make_producer("cwq"), CWQScoreProducer)

    def test_make_producer_unknown_raises(self):
        from kgqa.cli.retrieve import _make_producer
        with self.assertRaises(SystemExit):
            _make_producer("nope")


class TestDumpParserLimit(unittest.TestCase):
    def test_limit_arg(self):
        from kgqa.cli.dump_scores import build_parser
        args = build_parser().parse_args(
            ["--dataset", "cwq", "--ckpt", "c", "--input_dir", "d",
             "--qa_file", "q", "--output", "o", "--limit", "4"])
        self.assertEqual(args.limit, 4)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑测试确认失败**

Run: `python -m unittest tests.kgqa.test_cli_dispatch -v`
Expected: FAIL——`ImportError: cannot import name '_make_producer'`；`--limit` 未定义报 `SystemExit`。

- [x] **Step 3: 改 dump_scores.py**

`build_parser` 增加（放在 `--per_hop_limit` 之后）：

```python
    p.add_argument("--limit", type=int, default=0,
                   help="CWQ 取前 N 条非空子图样本（小子集），0=全量")
```

`main` 分发表增加 cwq 分支（`metaqa` 分支之后）：

```python
    elif args.dataset == "cwq":
        from kgqa.models.cwq import CWQScoreProducer
        producer = CWQScoreProducer(limit=args.limit)
```

- [x] **Step 4: 改 retrieve.py**

新增模块级函数（`build_parser` 之后）并替换 online 分支的硬编码：

```python
def _make_producer(dataset: str):
    if dataset == "webqsp":
        from kgqa.models.webqsp import WebQSPScoreProducer
        return WebQSPScoreProducer()
    if dataset == "metaqa":
        from kgqa.models.metaqa import MetaQAScoreProducer
        return MetaQAScoreProducer()
    if dataset == "cwq":
        from kgqa.models.cwq import CWQScoreProducer
        return CWQScoreProducer()
    raise SystemExit(f"未支持的 online producer: {dataset}")
```

`build_backend` 的 online 分支改为：

```python
    from kgqa.retrieve.backends.online import OnlineBackend
    if not (args.ckpt and args.qa_file):
        raise SystemExit("--backend online 需要 --ckpt 和 --qa_file")
    backend = OnlineBackend(adapter, _make_producer(args.dataset), ckpt_path=args.ckpt,
                            input_dir=args.input_dir, qa_file=args.qa_file,
                            split=args.split, limit=args.limit)
    return adapter, backend
```

（删除原 `from kgqa.models.webqsp import WebQSPScoreProducer` 顶部分支内 import。）

- [x] **Step 5: 跑测试确认通过 + 全量零回归**

Run: `python -m unittest tests.kgqa.test_cli_dispatch -v`
Expected: PASS（3 个测试）
Run: `python -m unittest discover -s tests/kgqa -p 'test*.py'`
Expected: 全部 PASS

- [x] **Step 6: Commit**

```bash
git add kgqa/cli/dump_scores.py kgqa/cli/retrieve.py tests/kgqa/test_cli_dispatch.py
git commit -m "$(cat <<'EOF'
feat(kgqa): CLI 支持 cwq 分发与 --limit 小子集

- cli/dump_scores.py: cwq→CWQScoreProducer 分发 + --limit 参数
- cli/retrieve.py: online 分支 producer 硬编码改为 _make_producer 按 dataset 分发
- tests/kgqa/test_cli_dispatch.py: 分发与参数单测（免数据）

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: CWQ 端到端集成测试（dump → 检索 → 评测 → parity）

**Files:**
- Test: `tests/kgqa/test_cwq_end_to_end.py`（含 dump 小子集，即 spec 中 `test_dump_cwq.py` 的覆盖点，合并进本文件避免重复 dump）

**Interfaces:**
- Consumes: Task 1-5 全部产物；`kgqa.cli.eval._gold_strings`、`kgqa.eval.answer_eval.{answer_record, answer_summary}`。
- Produces: 集成级保真证据（缓存含 triples、检索出路径、hit1>0、online/offline parity）。

- [x] **Step 1: 写测试（ckpt/数据存在才跑）**

创建 `tests/kgqa/test_cwq_end_to_end.py`：

```python
import os
import tempfile
import unittest

import torch

CKPT = "data/ckpt/CWQ/model-29-0.4206.pt"
INPUT_DIR = "data/input/CWQ"
QA_FILE = "data/input/CWQ/test_simple.json"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(QA_FILE), "ckpt/数据缺失，跳过")
class TestCWQEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgqa.cli.dump_scores import main as dump_main
        cls.cache = os.path.join(tempfile.mkdtemp(), "cwq_small.pt")
        dump_main(["--dataset", "cwq", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
                   "--qa_file", QA_FILE, "--output", cls.cache, "--limit", "20"])

    def _offline(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        adapter = get_adapter("cwq", input_dir=INPUT_DIR)
        return OfflineBackend(adapter, cache_path=self.cache)

    def test_cache_contains_triples(self):
        cache = torch.load(self.cache, weights_only=False)
        self.assertEqual(len(cache["samples"]), 20)
        self.assertTrue(all(s.get("triples") for s in cache["samples"]))

    def test_offline_retrieves_paths(self):
        backend = self._offline()
        results = backend.retrieve_all()
        self.assertEqual(len(results), len(backend.bundle.samples))
        self.assertTrue(any(r.paths for r in results))

    def test_answer_eval_hit1_positive(self):
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
        self.assertIn("hit1", summary["overall"])
        # 口径一致时 hit1 不应为 0（ckpt acc 0.42，20 条全 miss 概率约 1.8e-5）
        self.assertGreater(summary["overall"]["hit1"], 0.0)

    def test_online_offline_parity_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.models.cwq import CWQScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = get_adapter("cwq", input_dir=INPUT_DIR)
        online = OnlineBackend(adapter, CWQScoreProducer(limit=20), ckpt_path=CKPT,
                               input_dir=INPUT_DIR, qa_file=QA_FILE)
        off = self._offline()
        for idx in range(3):
            ro = online.retrieve(idx)
            rf = off.retrieve(idx)
            self.assertEqual([p["path"] for p in ro.paths], [p["path"] for p in rf.paths])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 跑集成测试**

Run: `python -m unittest tests.kgqa.test_cwq_end_to_end -v`
Expected: PASS（4 个测试；首次跑含 BERT 加载与小子集前向，约 1-3 分钟）。若 hit1 断言失败，优先排查 gold/pred 口径（Plan2 的 name 口径全 0 教训——CWQ 应为 int gold id 经 id2ent → MID，与 prediction 的 MID 键同口径）。

- [x] **Step 3: 全量零回归**

Run: `python -m unittest discover -s tests/kgqa -p 'test*.py'`
Expected: 全部 PASS

- [x] **Step 4: Commit**

```bash
git add tests/kgqa/test_cwq_end_to_end.py
git commit -m "$(cat <<'EOF'
test(kgqa): CWQ 端到端集成（dump 小子集 + 检索 + hit1>0 + parity）

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: 全量 dump + 正式评测 + 实验记录

**Files:**
- Create: `docs/experiments_kgqa_stage1_cwq_20260711.md`（若实际执行日不同，按当日命名）
- 产物（gitignored）：`data/output/CWQ/score_cache/cwq_test_full.pt`、`data/output/CWQ/eval/cwq_test_summary.json`

**Interfaces:**
- Consumes: Task 1-6 全部产物。
- Produces: CWQ 正式 overall 数字（answer + path 指标）与实验记录文档。

- [x] **Step 1: 全量 dump（test 3531 条，扣除空子图）**

```bash
python -m kgqa.cli.dump_scores --dataset cwq \
    --ckpt data/ckpt/CWQ/model-29-0.4206.pt \
    --input_dir data/input/CWQ \
    --qa_file data/input/CWQ/test_simple.json \
    --output data/output/CWQ/score_cache/cwq_test_full.pt
```

Expected: `[INFO] dump 完成 N 条 → ...`（N ≤ 3531；预计 10-30 分钟前向 + 缓存约 300-400MB。若显存不足，加 `--batch_size 8`）。

- [x] **Step 2: 全量评测**

```bash
python -m kgqa.cli.eval --dataset cwq --backend offline \
    --cache data/output/CWQ/score_cache/cwq_test_full.pt \
    --input_dir data/input/CWQ \
    --summary data/output/CWQ/eval/cwq_test_summary.json
```

Expected: stdout 打印 overall 指标 JSON；`hit1` 与 ckpt acc 0.4206 量级吻合（±0.02 内视为通过；显著偏离则排查口径后再判断）。

- [x] **Step 3: 写实验记录**

创建 `docs/experiments_kgqa_stage1_cwq_20260711.md`，结构 mirror `docs/experiments_kgqa_stage1_metaqa_20260710.md`：配置（ckpt/数据/流程/内核）、answer 指标表（overall 一行）、path 指标表、与旧 CompWebQ predict 内核的差异说明（单一内核收敛，不逐条复现）、过程记录（有效样本数 N、空子图扣除数、耗时、遇到的坑）。数字以实际运行输出为准填入。

- [x] **Step 4: 全量测试收尾 + 回填 plan checkbox**

Run: `python -m unittest discover -s tests/kgqa -p 'test*.py' -v`
Expected: 全部 PASS。随后把本 plan 文件所有已完成 checkbox 勾选。

- [x] **Step 5: Commit**

```bash
git add docs/experiments_kgqa_stage1_cwq_20260711.md docs/superpowers/plans/2026-07-11-kgqa-stage1-cwq.md
git commit -m "$(cat <<'EOF'
docs(kgqa): 记录 CWQ 全量评测数字并回填 plan

Co-Authored-By: jsh-smi-wsl <1099048889@qq.com>
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```
