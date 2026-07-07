# kgqa Stage 1 · 核心骨架 + WebQSP 端到端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `kgqa/` 总包，把 WebQSP 的路径检索+评测从散落的旧代码迁成「共享引擎 + 可插拔策略 + 双后端」结构，跑通端到端且指标与旧实现逐条一致。

**Architecture:** 方案 C——`retrieve/engine.py` 消费三个接口（`KGEdgeSource`、`SampleScore`、`id2ent/id2rel`），不认识具体数据集；WebQSP 差异收进 `datasets/webqsp.py`、`kg/global_kg.py`、`scores/webqsp.py`。检索双后端（offline 读缓存 / online 实时前向）产出同一 `SampleScore`，共用同一 engine。

**Tech Stack:** Python 3.12（conda `py312_t271_cuda`）、PyTorch 2.7、unittest、FastAPI/uvicorn（server，与现有 `oh_my_agent` 一致）。

## Global Constraints

- Python 3.12，本地环境 `py312_t271_cuda`；测试用 `python -m unittest`。
- 数值保真红线：engine 内核（MMR、beam 搜索、稀疏重建、LogNorm 打分）**逐字迁移**自 `scripts/offline_path_search.py`，不得改写公式；由回归测试锁定。
- 迁移期旧代码（`scripts/offline_path_search.py`、`oh_my_agent/path_retrieve_server/`、`oh_my_agent/tools/path_retrieve.py`）**保留不删**，本 plan 只新增 `kgqa/`；退役在三数据集全部迁完后单独处理。
- 不训练、不改 4 个 TransferNet 模块、不碰 SFT/agent。
- 提交遵循 CLAUDE.md：Conventional Commits + 中文正文；`Co-Authored-By` 用 `git config` 读取；按文件名暂存。**仅在本 plan 步骤明确到「Commit」时提交。**
- 现有资产：WebQSP ckpt `data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt`；输入 `data/input/WebQSP`（含 `fbwq_full/`）；缓存 `data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt`；QA `data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt`；名字映射 `data/resources/WebQSP/fbwq_full/mid2name.txt`。

---

## 文件结构（本 plan 落地范围）

```
kgqa/
├── __init__.py
├── types.py                       # Task 1
├── kg/
│   ├── __init__.py  base.py       # Task 2 KGEdgeSource 接口
│   └── global_kg.py               # Task 2 全局邻接表
├── retrieve/
│   ├── __init__.py
│   ├── engine.py                  # Task 3 迁移内核 + retrieve_one()
│   └── backends/
│       ├── __init__.py  base.py   # Task 7 RetrieveBackend 接口
│       ├── offline.py             # Task 7
│       └── online.py              # Task 8
├── scores/
│   ├── __init__.py  base.py       # Task 4 ScoreLoader/Dumper 接口 + SampleScore/CacheMeta/ScoreBundle
│   └── webqsp.py                  # Task 4 WebQSP 加载
├── models/
│   ├── __init__.py  base.py       # Task 8 ScoreProducer 接口
│   └── webqsp.py                  # Task 8 WebQSP 模型封装
├── datasets/
│   ├── __init__.py  base.py       # Task 5 DatasetAdapter 接口
│   ├── webqsp.py                  # Task 5 WebQSP 适配器
│   └── registry.py                # Task 5
├── eval/
│   ├── __init__.py
│   ├── answer_eval.py             # Task 6 答案级 + group_by
│   └── path_eval.py               # Task 6 路径级
├── server/
│   ├── __init__.py
│   └── path_retrieve_server.py    # Task 10
└── cli/
    ├── __init__.py
    ├── dump_scores.py             # Task 9
    ├── retrieve.py                # Task 9
    └── eval.py                    # Task 9
tests/kgqa/
├── __init__.py
├── fixtures.py                    # Task 1 合成 SampleScore/edge_source
├── test_types.py                  # Task 1
├── test_global_kg.py             # Task 2
├── test_engine.py                # Task 3
├── test_scores_webqsp.py         # Task 4
├── test_dataset_webqsp.py        # Task 5
├── test_answer_eval.py           # Task 6
├── test_path_eval.py             # Task 6
├── test_backend_offline.py       # Task 7
├── test_backend_online.py        # Task 8 (ckpt 存在才跑)
├── test_cli.py                   # Task 9
├── test_server.py                # Task 10
├── test_backend_parity.py        # Task 11 (ckpt 存在才跑)
└── test_webqsp_regression.py     # Task 11 (缓存存在才跑)
```

---

### Task 1: 核心类型与测试脚手架

**Files:**
- Create: `kgqa/__init__.py`（空）, `kgqa/types.py`, `tests/kgqa/__init__.py`（空）, `tests/kgqa/fixtures.py`, `tests/kgqa/test_types.py`

**Interfaces:**
- Produces:
  - `QASample(question:str, topic_ids:list[int], gold_ids:list[int], sample_index:int=-1, hop:int|None=None, extra:dict=...)`（frozen dataclass）
  - `ReasonPath(nodes:list[int], rels:list[int], score:float)`，方法 `to_triples(id2ent:dict, id2rel:dict) -> list[list[str]]`
  - `RetrieveResult(question:str, topics:list[str], hop:int, paths:list[dict], prediction:dict[str,float], elapsed_ms:float, sample_index:int=-1)`
  - `MetricSpec(gold_key:str="mid", group_by:str|None=None, answer_metrics:bool=True, path_metrics:bool=True)`（frozen）
  - `tests/kgqa/fixtures.py`：`toy_sample_score()`（Task 3 复用）、`toy_edge_source()`（Task 3 复用）——本任务先建 `RetrieveResult`/`ReasonPath` 相关 fixture，`SampleScore` fixture 留空注释占位由 Task 4 补。

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_types.py`

```python
import unittest
from kgqa.types import QASample, ReasonPath, RetrieveResult, MetricSpec


class TestTypes(unittest.TestCase):
    def test_qasample_defaults(self):
        s = QASample(question="q", topic_ids=[1], gold_ids=[2, 3])
        self.assertEqual(s.sample_index, -1)
        self.assertIsNone(s.hop)
        self.assertEqual(s.extra, {})

    def test_reasonpath_to_triples(self):
        p = ReasonPath(nodes=[10, 11, 12], rels=[5, 6], score=-1.5)
        id2ent = {10: "m.a", 11: "m.b", 12: "m.c"}
        id2rel = {5: "r1", 6: "r2"}
        self.assertEqual(
            p.to_triples(id2ent, id2rel),
            [["m.a", "r1", "m.b"], ["m.b", "r2", "m.c"]],
        )

    def test_reasonpath_to_triples_missing_id_falls_back_to_str(self):
        p = ReasonPath(nodes=[10, 99], rels=[5], score=0.0)
        self.assertEqual(p.to_triples({10: "m.a"}, {}), [["m.a", "5", "99"]])

    def test_metricspec_defaults(self):
        spec = MetricSpec()
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)

    def test_retrieve_result_holds_paths(self):
        r = RetrieveResult(question="q", topics=["m.a"], hop=1,
                           paths=[{"path": [["m.a", "r", "m.b"]], "log_score": -0.1}],
                           prediction={"m.b": 0.9}, elapsed_ms=1.2)
        self.assertEqual(r.paths[0]["log_score"], -0.1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_types -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa'`）

- [ ] **Step 3: 写实现** `kgqa/types.py`

```python
"""kgqa 全包共享数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class QASample:
    question: str
    topic_ids: list[int]
    gold_ids: list[int]
    sample_index: int = -1
    hop: Optional[int] = None            # MetaQA 分跳标签；未知为 None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasonPath:
    nodes: list[int]
    rels: list[int]
    score: float

    def to_triples(self, id2ent: dict, id2rel: dict) -> list[list[str]]:
        return [
            [id2ent.get(self.nodes[i], str(self.nodes[i])),
             id2rel.get(self.rels[i], str(self.rels[i])),
             id2ent.get(self.nodes[i + 1], str(self.nodes[i + 1]))]
            for i in range(len(self.rels))
        ]


@dataclass
class RetrieveResult:
    question: str
    topics: list[str]
    hop: int
    paths: list[dict]                    # {"path": [[h,r,t],...], "log_score": float}
    prediction: dict[str, float]
    elapsed_ms: float
    sample_index: int = -1


@dataclass(frozen=True)
class MetricSpec:
    gold_key: str = "mid"                # "mid" | "name"
    group_by: Optional[str] = None       # None | "hop"
    answer_metrics: bool = True
    path_metrics: bool = True
```

也创建空文件：`kgqa/__init__.py`、`tests/kgqa/__init__.py`。创建 `tests/kgqa/fixtures.py`：

```python
"""kgqa 测试合成夹具。"""
from __future__ import annotations

# SampleScore 夹具在 Task 4 定义 SampleScore 后补充（toy_sample_score/toy_edge_source）。
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_types -v`
Expected: PASS（5 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/__init__.py kgqa/types.py tests/kgqa/__init__.py tests/kgqa/fixtures.py tests/kgqa/test_types.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增统一包核心数据类型

- kgqa/types.py: QASample/ReasonPath/RetrieveResult/MetricSpec
- tests/kgqa/test_types.py: 类型单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: KGEdgeSource 策略口 + 全局邻接表

**Files:**
- Create: `kgqa/kg/__init__.py`（空）, `kgqa/kg/base.py`, `kgqa/kg/global_kg.py`, `tests/kgqa/test_global_kg.py`

**Interfaces:**
- Consumes: `utils.path_utils.build_valid_edges_dict(triples:list[list[int]]) -> dict[int,list[tuple[int,int]]]`（已存在）
- Produces:
  - `KGEdgeSource`（ABC）：`neighbors(node_id:int) -> list[tuple[int,int]]`（返回 `(rel_id, tail_id)` 列表）、`all_edges() -> Iterable[tuple[int,int,int]]`
  - `GlobalKG(valid_edges_dict:dict[int,list[tuple[int,int]]])`，实现上述接口
  - `GlobalKG.from_input_dir(input_dir:str) -> GlobalKG`（迁移 `scripts.offline_path_search.rebuild_valid_edges_dict` 的读边逻辑）
  - 属性 `GlobalKG.valid_edges_dict`（engine 直接按 `dict.get(node, [])` 消费）

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_global_kg.py`

```python
import unittest
from kgqa.kg.global_kg import GlobalKG


class TestGlobalKG(unittest.TestCase):
    def _kg(self):
        # 三元组 (subj, rel, obj)
        return GlobalKG.from_triples([[0, 100, 1], [0, 101, 2], [1, 100, 3]])

    def test_neighbors(self):
        kg = self._kg()
        self.assertCountEqual(kg.neighbors(0), [(100, 1), (101, 2)])
        self.assertEqual(kg.neighbors(1), [(100, 3)])
        self.assertEqual(kg.neighbors(999), [])

    def test_all_edges(self):
        kg = self._kg()
        self.assertCountEqual(
            list(kg.all_edges()), [(0, 100, 1), (0, 101, 2), (1, 100, 3)]
        )

    def test_valid_edges_dict_attr_matches_neighbors(self):
        kg = self._kg()
        self.assertEqual(kg.valid_edges_dict.get(0, []), kg.neighbors(0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_global_kg -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.kg'`）

- [ ] **Step 3: 写实现**

`kgqa/kg/base.py`：
```python
"""KG 边来源策略口（方案 C 发散点之一）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class KGEdgeSource(ABC):
    @abstractmethod
    def neighbors(self, node_id: int) -> list[tuple[int, int]]:
        """返回从 node_id 出发的 (rel_id, tail_id) 列表。"""

    @abstractmethod
    def all_edges(self) -> Iterable[tuple[int, int, int]]:
        """遍历全部 (subj_id, rel_id, obj_id)。"""
```

`kgqa/kg/global_kg.py`：
```python
"""全局邻接表（WebQSP / MetaQA 共用）。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.path_utils import build_valid_edges_dict
from kgqa.kg.base import KGEdgeSource


class GlobalKG(KGEdgeSource):
    def __init__(self, valid_edges_dict: dict[int, list[tuple[int, int]]]):
        self.valid_edges_dict = valid_edges_dict

    @classmethod
    def from_triples(cls, triples: list[list[int]]) -> "GlobalKG":
        return cls(build_valid_edges_dict(triples))

    @classmethod
    def from_input_dir(cls, input_dir: str) -> "GlobalKG":
        """从 fbwq_full/{entities.dict,relations.dict,train.txt} 重建（含 _reverse 边）。

        逻辑迁移自 scripts.offline_path_search.rebuild_valid_edges_dict，逐字保留。"""
        fb_dir = Path(input_dir) / "fbwq_full"

        ent2id: dict[str, int] = {}
        with (fb_dir / "entities.dict").open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    ent2id[parts[0].strip()] = len(ent2id)

        rel2id: dict[str, int] = {}
        with (fb_dir / "relations.dict").open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel2id[parts[0].strip()] = int(parts[1])

        triples: list[list[int]] = []
        with (fb_dir / "train.txt").open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if s not in ent2id or r not in rel2id or o not in ent2id:
                    continue
                sid, rid, oid = ent2id[s], rel2id[r], ent2id[o]
                triples.append([sid, rid, oid])
                rev = r + "_reverse"
                if rev in rel2id:
                    triples.append([oid, rel2id[rev], sid])
        return cls.from_triples(triples)

    def neighbors(self, node_id: int) -> list[tuple[int, int]]:
        return self.valid_edges_dict.get(node_id, [])

    def all_edges(self) -> Iterable[tuple[int, int, int]]:
        for subj, edges in self.valid_edges_dict.items():
            for rel, obj in edges:
                yield (subj, rel, obj)
```

创建空 `kgqa/kg/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_global_kg -v`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/kg/__init__.py kgqa/kg/base.py kgqa/kg/global_kg.py tests/kgqa/test_global_kg.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 KGEdgeSource 策略口与全局邻接表

- kgqa/kg/base.py: KGEdgeSource 抽象接口
- kgqa/kg/global_kg.py: GlobalKG（from_triples/from_input_dir），迁移 rebuild 逻辑
- tests/kgqa/test_global_kg.py: 邻接表单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 检索引擎（逐字迁移内核 + retrieve_one 编排）

**Files:**
- Create: `kgqa/retrieve/__init__.py`（空）, `kgqa/retrieve/engine.py`, `tests/kgqa/test_engine.py`
- Modify: `tests/kgqa/fixtures.py`（补 `SampleScore` 相关夹具占位 → 实际在 Task 4 定义 SampleScore；本任务的 fixture 直接用 dict 形态，见下）

**Interfaces:**
- Consumes: `KGEdgeSource.neighbors`（Task 2）、`utils.path_utils.{path_to_rel_set}`；内核函数逐字迁移自 `scripts/offline_path_search.py`。
- Produces（engine 公开 API）：
  - 逐字迁移函数（保持同名同签名）：`PathCandidate`、`compute_candidate_score`、`score_path_candidates`、`candidate_to_tuple`、`select_path_candidates`、`search_path_candidates`、`reconstruct_ent_dict`、`reconstruct_rel_dict`、`LogNormStrategy`、`_method_hop_numbers`
  - 新增编排：`retrieve_one(sample:SampleScoreLike, edge_source:KGEdgeSource, id2ent:dict, id2rel:dict, *, method:str="tail_blend", alpha_final:float=1.0, threshold:float=0.01, beam_size:int=50, lambda_val:float=0.2, drop_loopback:bool=True) -> RetrieveResult`
  - 辅助：`drop_loopback_paths(paths) -> list`、`final_ent_score_dict(sample) -> dict[int,float]`、`build_prediction(sample, id2ent, score_threshold=0.9) -> dict[str,float]`
  - `SampleScoreLike` = 具有属性 `question / topic_ids / gold_ids / hop_attn / rel_probs / ent_indices / ent_scores / e_score_indices / e_score_values` 的对象（Task 4 的 `SampleScore` 满足）。

> **迁移说明（数值红线）**：`PathCandidate`、`compute_candidate_score`、`score_path_candidates`、`_ranked_candidates`、`candidate_to_tuple`、`select_path_candidates`、`search_path_candidates`、`reconstruct_ent_dict`、`reconstruct_rel_dict`、`LogNormStrategy`、`_method_hop_numbers`、`_path_to_triples`、`final_ent_score_dict` 全部**从 `scripts/offline_path_search.py` 逐字复制**（对应源行：42-141、157-169、228-249、305-355、362-387）到 `kgqa/retrieve/engine.py`，仅去掉脚本 `sys.path` 注入。不得改动任何公式/常量（`EPS=1e-9` 一并搬来）。`drop_loopback_paths` 逐字迁移自 `oh_my_agent/path_retrieve_server/service.py:43-56`。

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_engine.py`

```python
import unittest
import torch

from kgqa.kg.global_kg import GlobalKG
from kgqa.retrieve import engine


class _Sample:
    """最小 SampleScoreLike：单跳，两个候选尾。"""
    question = "toy question"
    topic_ids = [0]
    gold_ids = [1]
    # hop_attn argmax=0 → hop_num=1
    hop_attn = torch.tensor([0.9, 0.1])
    rel_probs = [torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])]
    ent_indices = [torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)]
    ent_scores = [torch.tensor([0.6, 0.5]), torch.tensor([])]
    e_score_indices = torch.tensor([1, 2])
    e_score_values = torch.tensor([0.95, 0.4])


class TestEngine(unittest.TestCase):
    def setUp(self):
        # 边：0 --rel1--> 1, 0 --rel2--> 2
        self.kg = GlobalKG.from_triples([[0, 1, 1], [0, 2, 2]])
        self.id2ent = {0: "m.topic", 1: "m.gold", 2: "m.other"}
        self.id2rel = {1: "rel.one", 2: "rel.two"}

    def test_reconstruct_rel_dict_threshold(self):
        d = engine.reconstruct_rel_dict(torch.tensor([0.0, 0.8, 0.005]), 0.01)
        self.assertEqual(d, {1: 0.800000011920929} if False else {1: float(torch.tensor(0.8))})

    def test_retrieve_one_returns_paths_and_prediction(self):
        r = engine.retrieve_one(
            _Sample(), self.kg, self.id2ent, self.id2rel,
            method="tail_blend", beam_size=10, threshold=0.01, lambda_val=0.2,
        )
        self.assertEqual(r.question, "toy question")
        self.assertEqual(r.hop, 1)
        self.assertTrue(r.paths)
        # 首条路径应命中 gold 尾 m.gold
        tails = [p["path"][-1][2] for p in r.paths]
        self.assertIn("m.gold", tails)
        # prediction 按 e_score>=0.9 过滤 → 只含 m.gold
        self.assertEqual(set(r.prediction), {"m.gold"})

    def test_drop_loopback_removes_self_return(self):
        paths = [([0, 1], [1], -0.1), ([0, 0], [1], -0.2)]
        kept = engine.drop_loopback_paths(paths)
        self.assertEqual(kept, [([0, 1], [1], -0.1)])


if __name__ == "__main__":
    unittest.main()
```

> 注：`test_reconstruct_rel_dict_threshold` 的断言写法仅示意浮点，实现步用更稳的断言（见 Step 3 附带修订）。

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_engine -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.retrieve'`）

- [ ] **Step 3: 写实现** `kgqa/retrieve/engine.py`

先把「迁移说明」列出的所有函数/类从 `scripts/offline_path_search.py` 逐字复制进来（含 `EPS`、`import math`、`from dataclasses import dataclass, replace`、`from typing import Optional`、`from utils.path_utils import path_to_rel_set`）。然后追加新增编排代码：

```python
from kgqa.types import RetrieveResult
from kgqa.kg.base import KGEdgeSource

PREDICTION_SCORE_THRESHOLD = 0.9


def drop_loopback_paths(paths):
    """剔除尾==topic 的自指路径（迁移自 path_retrieve_server/service.py）。"""
    return [
        (node_ids, rel_ids, score)
        for node_ids, rel_ids, score in paths
        if not node_ids or node_ids[-1] != node_ids[0]
    ]


def final_ent_score_dict(sample) -> dict[int, float]:
    return {
        int(idx): float(val)
        for idx, val in zip(sample.e_score_indices.tolist(), sample.e_score_values.tolist())
    }


def build_prediction(sample, id2ent: dict, score_threshold: float = PREDICTION_SCORE_THRESHOLD) -> dict[str, float]:
    prediction: dict[str, float] = {}
    for idx, val in zip(sample.e_score_indices.tolist(), sample.e_score_values.tolist()):
        if float(val) >= score_threshold:
            prediction[id2ent.get(int(idx), str(int(idx)))] = round(float(val), 4)
    return prediction


def _serialize_paths(paths, id2ent: dict, id2rel: dict) -> list[dict]:
    return [
        {"path": _path_to_triples(nodes, rels, id2ent, id2rel),
         "log_score": round(float(score), 6)}
        for nodes, rels, score in paths
    ]


def retrieve_one(sample, edge_source: KGEdgeSource, id2ent: dict, id2rel: dict, *,
                 method: str = "tail_blend", alpha_final: float = 1.0,
                 threshold: float = 0.01, beam_size: int = 50,
                 lambda_val: float = 0.2, drop_loopback: bool = True) -> RetrieveResult:
    """单样本检索：稀疏重建 → 逐跳 beam 展开 → MMR 选择 → 序列化。

    与 offline_path_search.run_experiment 的单样本分支逻辑等价（Task 11 回归锁定）。"""
    import time
    t0 = time.perf_counter()
    valid_edges_dict = edge_source.valid_edges_dict if hasattr(edge_source, "valid_edges_dict") else None

    hop_num = int(sample.hop_attn.argmax().item()) + 1
    hop_nums = _method_hop_numbers(method, hop_num, len(sample.rel_probs))

    rel_dicts, ent_dicts = [], []
    for t in range(max(hop_nums)):
        rel_dicts.append(reconstruct_rel_dict(sample.rel_probs[t], threshold))
        ent_dicts.append(reconstruct_ent_dict(sample.ent_indices[t], sample.ent_scores[t], threshold))

    scoring = LogNormStrategy()
    final_scores = final_ent_score_dict(sample)
    path_candidates = []
    for candidate_hop in hop_nums:
        path_candidates.extend(search_path_candidates(
            sample.topic_ids, rel_dicts, ent_dicts, candidate_hop,
            valid_edges_dict, scoring, beam_size,
            final_ent_scores=final_scores, order_start=len(path_candidates),
        ))

    selected = select_path_candidates(
        path_candidates, beam_size, method=method,
        alpha_final=alpha_final, lambda_val=lambda_val,
    )
    candidates = [candidate_to_tuple(c) for c in selected]
    if drop_loopback:
        candidates = drop_loopback_paths(candidates)

    topics = [id2ent.get(int(t), str(int(t))) for t in sample.topic_ids]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return RetrieveResult(
        question=sample.question,
        topics=topics,
        hop=hop_num,
        paths=_serialize_paths(candidates, id2ent, id2rel),
        prediction=build_prediction(sample, id2ent),
        elapsed_ms=elapsed_ms,
        sample_index=getattr(sample, "sample_index", -1),
    )
```

同时把 Step 1 里 `test_reconstruct_rel_dict_threshold` 的断言改稳：
```python
    def test_reconstruct_rel_dict_threshold(self):
        d = engine.reconstruct_rel_dict(torch.tensor([0.0, 0.8, 0.005]), 0.01)
        self.assertEqual(set(d), {1})
        self.assertAlmostEqual(d[1], 0.8, places=5)
```

创建空 `kgqa/retrieve/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_engine -v`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/retrieve/__init__.py kgqa/retrieve/engine.py tests/kgqa/test_engine.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 迁移检索内核并新增 retrieve_one 编排

- kgqa/retrieve/engine.py: 逐字迁移 MMR/beam/稀疏重建内核 + retrieve_one/build_prediction
- tests/kgqa/test_engine.py: 单样本检索单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: ScoreLoader 策略口 + SampleScore 类型 + WebQSP 加载

**Files:**
- Create: `kgqa/scores/__init__.py`（空）, `kgqa/scores/base.py`, `kgqa/scores/webqsp.py`, `tests/kgqa/test_scores_webqsp.py`
- Modify: `tests/kgqa/fixtures.py`（补 `toy_sample_score()`）

**Interfaces:**
- Consumes: `scripts.offline_path_search.load_score_cache`（读 .pt，逐字复用，不迁移）
- Produces:
  - `SampleScore`（dataclass）：字段 `question:str, topic_ids:list[int], gold_ids:list[int], hop_attn, rel_probs, ent_indices, ent_scores, e_score_indices, e_score_values, sample_index:int=-1, hop:int|None=None, triples:list[list[int]]|None=None`。满足 Task 3 的 `SampleScoreLike`。
  - `CacheMeta`（dataclass）：`dataset:str, split:str, id2ent:dict, id2rel:dict, num_samples:int, topk_entities:int, input_dir:str|None, qa_file:str|None`
  - `ScoreBundle`（dataclass）：`meta:CacheMeta, samples:list[SampleScore]`
  - `ScoreLoader`（ABC）：`load(cache_path:str) -> ScoreBundle`
  - `ScoreDumper`（ABC）：`dump(bundle:ScoreBundle, out_path:str) -> None`（本任务只实现 `WebQSPScoreLoader`；dumper 接口留给 Task 8/9）
  - `WebQSPScoreLoader().load(path) -> ScoreBundle`

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_scores_webqsp.py`

```python
import os
import unittest

from kgqa.scores.webqsp import WebQSPScoreLoader

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestWebQSPScoreLoader(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
    def test_load_bundle(self):
        bundle = WebQSPScoreLoader().load(CACHE)
        self.assertEqual(bundle.meta.dataset, "WebQSP")
        self.assertEqual(bundle.meta.num_samples, len(bundle.samples))
        self.assertGreater(len(bundle.samples), 0)
        s = bundle.samples[0]
        self.assertIsInstance(s.question, str)
        self.assertTrue(hasattr(s, "hop_attn"))
        self.assertTrue(hasattr(s, "e_score_values"))
        self.assertEqual(s.sample_index, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_scores_webqsp -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.scores'`）

- [ ] **Step 3: 写实现**

`kgqa/scores/base.py`：
```python
"""得分 dump/load 策略口（方案 C 发散点之二）+ 统一 SampleScore/CacheMeta/ScoreBundle。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SampleScore:
    question: str
    topic_ids: list[int]
    gold_ids: list[int]
    hop_attn: Any
    rel_probs: list[Any]
    ent_indices: list[Any]
    ent_scores: list[Any]
    e_score_indices: Any
    e_score_values: Any
    sample_index: int = -1
    hop: Optional[int] = None
    triples: Optional[list[list[int]]] = None


@dataclass
class CacheMeta:
    dataset: str
    split: str
    id2ent: dict
    id2rel: dict
    num_samples: int
    topk_entities: int = 500
    input_dir: Optional[str] = None
    qa_file: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBundle:
    meta: CacheMeta
    samples: list[SampleScore]


class ScoreLoader(ABC):
    @abstractmethod
    def load(self, cache_path: str) -> ScoreBundle: ...


class ScoreDumper(ABC):
    @abstractmethod
    def dump(self, bundle: ScoreBundle, out_path: str) -> None: ...
```

`kgqa/scores/webqsp.py`：
```python
"""WebQSP 得分缓存加载：把 dump_scores.py 的 dict 缓存转成 ScoreBundle。"""
from __future__ import annotations

from scripts.offline_path_search import load_score_cache
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle, ScoreLoader


class WebQSPScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        cache = load_score_cache(cache_path)
        meta_d = cache["meta"]
        meta = CacheMeta(
            dataset=meta_d.get("dataset", "WebQSP"),
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
                triples=s.get("triples"),
            )
            for i, s in enumerate(cache["samples"])
        ]
        return ScoreBundle(meta=meta, samples=samples)
```

创建空 `kgqa/scores/__init__.py`。在 `tests/kgqa/fixtures.py` 追加：
```python
import torch
from kgqa.scores.base import SampleScore


def toy_sample_score() -> SampleScore:
    return SampleScore(
        question="toy question",
        topic_ids=[0],
        gold_ids=[1],
        hop_attn=torch.tensor([0.9, 0.1]),
        rel_probs=[torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])],
        ent_indices=[torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)],
        ent_scores=[torch.tensor([0.6, 0.5]), torch.tensor([])],
        e_score_indices=torch.tensor([1, 2]),
        e_score_values=torch.tensor([0.95, 0.4]),
        sample_index=0,
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_scores_webqsp -v`
Expected: PASS（缓存存在则 1 test 通过；否则 skip）

- [ ] **Step 5: Commit**

```bash
git add kgqa/scores/__init__.py kgqa/scores/base.py kgqa/scores/webqsp.py tests/kgqa/fixtures.py tests/kgqa/test_scores_webqsp.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 ScoreLoader 策略口与 WebQSP 加载

- kgqa/scores/base.py: SampleScore/CacheMeta/ScoreBundle + ScoreLoader/ScoreDumper 接口
- kgqa/scores/webqsp.py: WebQSPScoreLoader（dict 缓存→ScoreBundle）
- tests/kgqa/fixtures.py: toy_sample_score 夹具

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: DatasetAdapter 接口 + WebQSP 适配器 + registry

**Files:**
- Create: `kgqa/datasets/__init__.py`（空）, `kgqa/datasets/base.py`, `kgqa/datasets/webqsp.py`, `kgqa/datasets/registry.py`, `tests/kgqa/test_dataset_webqsp.py`

**Interfaces:**
- Consumes: `oh_my_agent.common.qa_data.parse_webqsp_qa_line`（复用解析）、`oh_my_agent.common.entity_mapping.load_entity_map`、`GlobalKG`（Task 2）、`WebQSPScoreLoader`（Task 4）、`MetricSpec`（Task 1）
- Produces:
  - `DatasetAdapter`（ABC）：属性 `name:str`、`max_hop:int`；方法 `load_qa(path:str, limit:int=0) -> list[QASample]`、`entity_name(entity_id:str) -> str`、`kg_edge_source(sample:QASample|None=None) -> KGEdgeSource`、`score_loader() -> ScoreLoader`、`metric_spec() -> MetricSpec`
  - `WebQSPAdapter(input_dir:str, entity_map_path:str|None=None)`
  - `get_adapter(name:str, **kwargs) -> DatasetAdapter`、`register_adapter(name, cls)`；内置注册 `"webqsp"`

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_dataset_webqsp.py`

```python
import os
import tempfile
import unittest

from kgqa.datasets.registry import get_adapter
from kgqa.datasets.webqsp import WebQSPAdapter
from kgqa.types import QASample, MetricSpec


class TestWebQSPAdapter(unittest.TestCase):
    def test_load_qa_parses_topic_and_gold(self):
        adapter = WebQSPAdapter(input_dir="data/input/WebQSP")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("what is the language spoken in france [m.0f8l9c]\tm.04306rv|m.02bv9\n")
            path = fh.name
        try:
            samples = adapter.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertIsInstance(s, QASample)
        self.assertEqual(s.extra["topic_mid"], "m.0f8l9c")
        self.assertEqual(s.gold_ids, ["m.04306rv", "m.02bv9"])

    def test_metric_spec_defaults_mid(self):
        adapter = WebQSPAdapter(input_dir="data/input/WebQSP")
        spec = adapter.metric_spec()
        self.assertIsInstance(spec, MetricSpec)
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)
        self.assertEqual(adapter.max_hop, 2)

    def test_registry_returns_webqsp(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        self.assertEqual(adapter.name, "webqsp")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_dataset_webqsp -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.datasets'`）

- [ ] **Step 3: 写实现**

`kgqa/datasets/base.py`：
```python
"""数据集适配器接口（薄数据提供者）。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from kgqa.kg.base import KGEdgeSource
from kgqa.scores.base import ScoreLoader
from kgqa.types import MetricSpec, QASample


class DatasetAdapter(ABC):
    name: str
    max_hop: int

    @abstractmethod
    def load_qa(self, path: str, limit: int = 0) -> list[QASample]: ...

    @abstractmethod
    def entity_name(self, entity_id: str) -> str: ...

    @abstractmethod
    def kg_edge_source(self, sample: QASample | None = None) -> KGEdgeSource: ...

    @abstractmethod
    def score_loader(self) -> ScoreLoader: ...

    @abstractmethod
    def metric_spec(self) -> MetricSpec: ...
```

`kgqa/datasets/webqsp.py`：
```python
"""WebQSP 适配器。"""
from __future__ import annotations

from functools import lru_cache

from oh_my_agent.common.qa_data import parse_webqsp_qa_line
from oh_my_agent.common.entity_mapping import load_entity_map
from kgqa.datasets.base import DatasetAdapter
from kgqa.kg.global_kg import GlobalKG
from kgqa.scores.base import ScoreLoader
from kgqa.scores.webqsp import WebQSPScoreLoader
from kgqa.types import MetricSpec, QASample

DEFAULT_ENTITY_MAP = "data/resources/WebQSP/fbwq_full/mid2name.txt"


class WebQSPAdapter(DatasetAdapter):
    name = "webqsp"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/WebQSP", entity_map_path: str | None = None):
        self.input_dir = input_dir
        self.entity_map_path = entity_map_path or DEFAULT_ENTITY_MAP
        self._entity_map: dict[str, str] | None = None
        self._kg: GlobalKG | None = None

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        samples: list[QASample] = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                parsed = parse_webqsp_qa_line(line)
                samples.append(QASample(
                    question=parsed.question,
                    topic_ids=[parsed.topic_mid] if parsed.topic_mid else [],
                    gold_ids=list(parsed.gold_mids),
                    sample_index=len(samples),
                    extra={"topic_mid": parsed.topic_mid, "question_raw": parsed.question_raw},
                ))
                if limit and len(samples) >= limit:
                    break
        return samples

    def _load_map(self) -> dict[str, str]:
        if self._entity_map is None:
            self._entity_map = load_entity_map(self.entity_map_path)
        return self._entity_map

    def entity_name(self, entity_id: str) -> str:
        return self._load_map().get(entity_id, entity_id)

    def kg_edge_source(self, sample: QASample | None = None) -> GlobalKG:
        if self._kg is None:
            self._kg = GlobalKG.from_input_dir(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return WebQSPScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None, answer_metrics=True, path_metrics=True)
```

`kgqa/datasets/registry.py`：
```python
"""数据集适配器注册表。"""
from __future__ import annotations

from kgqa.datasets.base import DatasetAdapter
from kgqa.datasets.webqsp import WebQSPAdapter

_REGISTRY: dict[str, type[DatasetAdapter]] = {"webqsp": WebQSPAdapter}


def register_adapter(name: str, cls: type[DatasetAdapter]) -> None:
    _REGISTRY[name] = cls


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的数据集适配器: {name}（已注册: {sorted(_REGISTRY)}）")
    return _REGISTRY[name](**kwargs)
```

创建空 `kgqa/datasets/__init__.py`。

> 注：`QASample.topic_ids` 在 WebQSP 存的是 MID 字符串（检索时以 `SampleScore.topic_ids` 的 int id 为准；QASample 的 topic 仅供评测/展示），类型为 `list[str]`。这与 Task 1 的 `list[int]` 声明放宽为 `list`——在 Task 1 实现中 `topic_ids` 未做类型强校验，运行期允许 str（MetaQA/CWQ 亦如此）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_dataset_webqsp -v`
Expected: PASS（3 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/datasets/ tests/kgqa/test_dataset_webqsp.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 DatasetAdapter 接口与 WebQSP 适配器

- kgqa/datasets/base.py: DatasetAdapter 抽象接口
- kgqa/datasets/webqsp.py: WebQSPAdapter（QA 解析/名字映射/KG/loader/metric_spec）
- kgqa/datasets/registry.py: 适配器注册表
- tests/kgqa/test_dataset_webqsp.py: 适配器单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 评测层（答案级 group_by + 路径级）

**Files:**
- Create: `kgqa/eval/__init__.py`（空）, `kgqa/eval/answer_eval.py`, `kgqa/eval/path_eval.py`, `tests/kgqa/test_answer_eval.py`, `tests/kgqa/test_path_eval.py`

**Interfaces:**
- Consumes: `oh_my_agent.common.metrics.{compute_answer_metrics, aggregate_metrics}`（逐字复用计算）、`utils.path_utils.{compute_path_metrics, compute_path_diversity}`、`MetricSpec`（Task 1）、`RetrieveResult`（Task 1）
- Produces:
  - `answer_summary(records:list[dict], spec:MetricSpec) -> dict`：`records` 每条含 `pred:list[str], gold:list[str], hop:int|None, format_ok:bool`；返回 `{"overall": {...}, "by_hop": {...}}`（`spec.group_by=="hop"` 时 `by_hop` 非空，否则为 `{}`）
  - `answer_record(pred:list[str], gold:list[str], hop:int|None=None, format_ok:bool=True) -> dict`（把 `compute_answer_metrics` 结果补 `hop/format_ok` 字段）
  - `path_summary(results:list[RetrieveResult], gold_by_index:dict[int,set[str]], spec:MetricSpec, id2rel:dict|None=None) -> dict`：路径级 hit/recall/precision/f1 + diversity，同样支持 `by_hop`

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_answer_eval.py`

```python
import unittest
from kgqa.eval.answer_eval import answer_record, answer_summary
from kgqa.types import MetricSpec


class TestAnswerEval(unittest.TestCase):
    def _records(self):
        return [
            answer_record(pred=["a"], gold=["a"], hop=1, format_ok=True),
            answer_record(pred=["x"], gold=["b"], hop=1, format_ok=True),
            answer_record(pred=["c", "d"], gold=["c"], hop=2, format_ok=True),
        ]

    def test_overall_hit1(self):
        out = answer_summary(self._records(), MetricSpec())
        self.assertIn("overall", out)
        self.assertEqual(out["by_hop"], {})
        self.assertAlmostEqual(out["overall"]["hit1"], 2 / 3, places=4)

    def test_group_by_hop(self):
        out = answer_summary(self._records(), MetricSpec(group_by="hop"))
        self.assertEqual(set(out["by_hop"]), {"1", "2"})
        self.assertAlmostEqual(out["by_hop"]["1"]["hit1"], 0.5, places=4)
        self.assertAlmostEqual(out["by_hop"]["2"]["hit1"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_answer_eval -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.eval'`）

- [ ] **Step 3: 写实现**

`kgqa/eval/answer_eval.py`：
```python
"""答案级评测：复用 oh_my_agent 指标，增加 group_by 分组视图。"""
from __future__ import annotations

from oh_my_agent.common.metrics import compute_answer_metrics, aggregate_metrics
from kgqa.types import MetricSpec


def answer_record(pred: list[str], gold: list[str], hop=None, format_ok: bool = True) -> dict:
    rec = dict(compute_answer_metrics(pred, gold))
    rec["hop"] = hop
    rec["format_ok"] = format_ok
    return rec


def answer_summary(records: list[dict], spec: MetricSpec) -> dict:
    overall = aggregate_metrics(records)
    by_hop: dict[str, dict] = {}
    if spec.group_by == "hop":
        groups: dict[str, list[dict]] = {}
        for rec in records:
            if rec.get("hop") is None:
                continue
            groups.setdefault(str(rec["hop"]), []).append(rec)
        by_hop = {hop: aggregate_metrics(recs) for hop, recs in sorted(groups.items())}
    return {"overall": overall, "by_hop": by_hop}
```

`kgqa/eval/path_eval.py`：
```python
"""路径级评测：复用 utils.path_utils，增加 group_by 分组视图。"""
from __future__ import annotations

from utils.path_utils import compute_path_metrics, compute_path_diversity
from kgqa.types import MetricSpec, RetrieveResult


def _paths_as_tuples(result: RetrieveResult) -> list[tuple[list[str], list[str], float]]:
    tuples = []
    for p in result.paths:
        edges = p["path"]
        nodes = ([edges[0][0]] + [e[2] for e in edges]) if edges else []
        rels = [e[1] for e in edges]
        tuples.append((nodes, rels, p.get("log_score", 0.0)))
    return tuples


def path_record(result: RetrieveResult, gold: set[str], id2rel=None) -> dict:
    selected = _paths_as_tuples(result)
    if not selected:
        return {"hop": result.hop, "answer_hit": 0, "top1_hit": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "jaccard_diversity": 0.0, "relation_jaccard_diversity": 0.0,
                "tail_diversity": 0.0, "relation_coverage": 0.0, "edge_coverage": 0.0}
    m = compute_path_metrics(selected, gold, id2rel=id2rel)
    d = compute_path_diversity(selected)
    return {
        "hop": result.hop,
        "answer_hit": int(m["answer_hit"]), "top1_hit": int(m["top1_hit"]),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "jaccard_diversity": d.get("jaccard_diversity", 0.0),
        "relation_jaccard_diversity": d.get("relation_jaccard_diversity", 0.0),
        "tail_diversity": d.get("tail_diversity", 0.0),
        "relation_coverage": d.get("relation_coverage", 0.0),
        "edge_coverage": d.get("edge_coverage", 0.0),
    }


def _mean(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ["answer_hit", "top1_hit", "precision", "recall", "f1",
            "jaccard_diversity", "relation_jaccard_diversity", "tail_diversity",
            "relation_coverage", "edge_coverage"]
    n = len(records)
    return {"n": n, **{k: round(sum(float(r[k]) for r in records) / n, 4) for k in keys}}


def path_summary(results: list[RetrieveResult], gold_by_index: dict, spec: MetricSpec,
                 id2rel=None) -> dict:
    records = [path_record(r, gold_by_index.get(r.sample_index, set()), id2rel=id2rel)
               for r in results]
    overall = _mean(records)
    by_hop: dict[str, dict] = {}
    if spec.group_by == "hop":
        groups: dict[str, list[dict]] = {}
        for rec in records:
            if rec.get("hop") is None:
                continue
            groups.setdefault(str(rec["hop"]), []).append(rec)
        by_hop = {hop: _mean(recs) for hop, recs in sorted(groups.items())}
    return {"overall": overall, "by_hop": by_hop}
```

`tests/kgqa/test_path_eval.py`：
```python
import unittest
from kgqa.eval.path_eval import path_record, path_summary
from kgqa.types import MetricSpec, RetrieveResult


def _result(idx, hop, tail):
    return RetrieveResult(
        question="q", topics=["m.t"], hop=hop,
        paths=[{"path": [["m.t", "r", tail]], "log_score": -0.1}],
        prediction={}, elapsed_ms=0.0, sample_index=idx,
    )


class TestPathEval(unittest.TestCase):
    def test_path_record_hit(self):
        rec = path_record(_result(0, 1, "m.gold"), {"m.gold"})
        self.assertEqual(rec["answer_hit"], 1)

    def test_path_summary_group_by_hop(self):
        results = [_result(0, 1, "m.gold"), _result(1, 2, "m.x")]
        gold = {0: {"m.gold"}, 1: {"m.gold"}}
        out = path_summary(results, gold, MetricSpec(group_by="hop"))
        self.assertEqual(set(out["by_hop"]), {"1", "2"})
        self.assertEqual(out["by_hop"]["1"]["answer_hit"], 1.0)
        self.assertEqual(out["by_hop"]["2"]["answer_hit"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

创建空 `kgqa/eval/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_answer_eval tests.kgqa.test_path_eval -v`
Expected: PASS（4 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/eval/ tests/kgqa/test_answer_eval.py tests/kgqa/test_path_eval.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增答案级/路径级评测（支持 group_by 分跳）

- kgqa/eval/answer_eval.py: answer_record/answer_summary，overall+by_hop
- kgqa/eval/path_eval.py: path_record/path_summary，复用 path_utils
- tests/kgqa: 评测单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: RetrieveBackend 接口 + 离线后端

**Files:**
- Create: `kgqa/retrieve/backends/__init__.py`（空）, `kgqa/retrieve/backends/base.py`, `kgqa/retrieve/backends/offline.py`, `tests/kgqa/test_backend_offline.py`

**Interfaces:**
- Consumes: `DatasetAdapter`（Task 5）、`ScoreBundle`（Task 4）、`engine.retrieve_one`（Task 3）
- Produces:
  - `RetrieveBackend`（ABC）：`retrieve(sample_index:int, **params) -> RetrieveResult`、`retrieve_all(**params) -> list[RetrieveResult]`、属性 `bundle:ScoreBundle`
  - `OfflineBackend(adapter:DatasetAdapter, cache_path:str)`：加载缓存到 `ScoreBundle`，`retrieve` 用 `adapter.kg_edge_source()` + `engine.retrieve_one`
  - 检索参数 `RetrieveParams`（dataclass，字段与 `retrieve_one` 对齐：method/alpha_final/threshold/beam_size/lambda_val/drop_loopback）

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_backend_offline.py`

```python
import os
import unittest

from kgqa.datasets.registry import get_adapter
from kgqa.retrieve.backends.offline import OfflineBackend

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestOfflineBackend(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
    def test_retrieve_single(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        backend = OfflineBackend(adapter, cache_path=CACHE)
        r = backend.retrieve(0, beam_size=50, method="tail_blend", lambda_val=0.2)
        self.assertEqual(r.sample_index, 0)
        self.assertGreaterEqual(len(r.paths), 1)
        self.assertTrue(all("path" in p and "log_score" in p for p in r.paths))

    @unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
    def test_retrieve_all_len(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        backend = OfflineBackend(adapter, cache_path=CACHE)
        results = backend.retrieve_all(beam_size=10, limit=5)
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_backend_offline -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.retrieve.backends'`）

- [ ] **Step 3: 写实现**

`kgqa/retrieve/backends/base.py`：
```python
"""检索后端接口 + 参数。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

from kgqa.types import RetrieveResult


@dataclass(frozen=True)
class RetrieveParams:
    method: str = "tail_blend"
    alpha_final: float = 1.0
    threshold: float = 0.01
    beam_size: int = 50
    lambda_val: float = 0.2
    drop_loopback: bool = True

    def as_kwargs(self) -> dict:
        return asdict(self)


class RetrieveBackend(ABC):
    @abstractmethod
    def retrieve(self, sample_index: int, **params) -> RetrieveResult: ...

    @abstractmethod
    def retrieve_all(self, *, limit: int = 0, **params) -> list[RetrieveResult]: ...
```

`kgqa/retrieve/backends/offline.py`：
```python
"""离线后端：读得分缓存 → engine。"""
from __future__ import annotations

from kgqa.datasets.base import DatasetAdapter
from kgqa.retrieve import engine
from kgqa.retrieve.backends.base import RetrieveBackend, RetrieveParams


class OfflineBackend(RetrieveBackend):
    def __init__(self, adapter: DatasetAdapter, cache_path: str):
        self.adapter = adapter
        self.bundle = adapter.score_loader().load(cache_path)
        self.edge_source = adapter.kg_edge_source()

    def _one(self, sample, params: dict):
        return engine.retrieve_one(
            sample, self.edge_source,
            self.bundle.meta.id2ent, self.bundle.meta.id2rel, **params,
        )

    def retrieve(self, sample_index: int, **params) -> object:
        merged = {**RetrieveParams().as_kwargs(), **params}
        return self._one(self.bundle.samples[sample_index], merged)

    def retrieve_all(self, *, limit: int = 0, **params) -> list:
        merged = {**RetrieveParams().as_kwargs(), **params}
        samples = self.bundle.samples[:limit] if limit else self.bundle.samples
        return [self._one(s, merged) for s in samples]
```

创建空 `kgqa/retrieve/backends/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_backend_offline -v`
Expected: PASS（缓存存在则 2 tests；否则 skip）

- [ ] **Step 5: Commit**

```bash
git add kgqa/retrieve/backends/ tests/kgqa/test_backend_offline.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 RetrieveBackend 接口与离线后端

- kgqa/retrieve/backends/base.py: RetrieveBackend 接口 + RetrieveParams
- kgqa/retrieve/backends/offline.py: OfflineBackend（缓存→engine）
- tests/kgqa/test_backend_offline.py: 离线后端单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: ScoreProducer 模型接口 + WebQSP 封装 + 在线后端

**Files:**
- Create: `kgqa/models/__init__.py`（空）, `kgqa/models/base.py`, `kgqa/models/webqsp.py`, `kgqa/retrieve/backends/online.py`, `tests/kgqa/test_backend_online.py`

**Interfaces:**
- Consumes: `WebQSP.model.TransferNet`、`WebQSP.data.{DataLoader, load_data}`、`WebQSP.predict.id_score_pairs`、`utils.misc.batch_device`（复现 `dump_scores` 前向）；`ScoreBundle/SampleScore`（Task 4）；`OfflineBackend` 结构（Task 7）
- Produces:
  - `ScoreProducer`（ABC）：`load_checkpoint(ckpt_path:str) -> None`、`produce(input_dir:str, qa_file:str, *, split:str="test", batch_size:int=16, topk:int=500) -> ScoreBundle`
  - `WebQSPScoreProducer(bert_name:str="bert-base-uncased")`：内部构建 model+loader，前向逐字复用 `WebQSP.dump_scores.dump_scores` 的采样逻辑，输出 `ScoreBundle`（而非写 .pt）
  - `OnlineBackend(adapter:DatasetAdapter, producer:ScoreProducer, ckpt_path:str, input_dir:str, qa_file:str, **produce_kwargs)`：`produce` 得到 `bundle` 后与 `OfflineBackend` 走同一 `engine.retrieve_one`

> **实现说明**：`WebQSPScoreProducer.produce` 把 `WebQSP/dump_scores.py:dump_scores` 的「前向 + 每样本 topk 采样」逻辑**逐字迁移**，唯一区别是最后不 `torch.save`，改为构造 `SampleScore` 列表并返回 `ScoreBundle`（`load_score_cache`→内存直连）。为 DRY，可让 `dump_scores.py` 与本类共享一个 `_extract_samples(outputs, batch, topk)` helper；但为遵守「迁移期不改旧代码」约束，本任务**复制**该逻辑，去重留待旧代码退役阶段。

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_backend_online.py`

```python
import os
import unittest

CKPT = "data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt"
INPUT_DIR = "data/input/WebQSP"
QA = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"


class TestOnlineBackend(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(QA), "ckpt/QA 缺失，跳过")
    def test_online_retrieve_smoke(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.models.webqsp import WebQSPScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend

        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        backend = OnlineBackend(
            adapter, WebQSPScoreProducer(), ckpt_path=CKPT,
            input_dir=INPUT_DIR, qa_file=QA, split="test", limit=3,
        )
        r = backend.retrieve(0, beam_size=50)
        self.assertGreaterEqual(len(r.paths), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_backend_online -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.models'`）或（无 ckpt 时）skip——若 skip 则先临时去掉 skip 装饰验证 import 失败，再复原。

- [ ] **Step 3: 写实现**

`kgqa/models/base.py`：
```python
"""模型接口：加载 ckpt → 前向 → 中间得分（训练循环不并入）。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from kgqa.scores.base import ScoreBundle


class ScoreProducer(ABC):
    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> None: ...

    @abstractmethod
    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle: ...
```

`kgqa/models/webqsp.py`：封装 model 构建与前向。核心 `produce`：
```python
"""WebQSP 在线得分生产（前向逻辑迁移自 WebQSP/dump_scores.py）。"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from utils.misc import batch_device
from WebQSP.data import DataLoader, load_data
from WebQSP.model import TransferNet
from WebQSP.predict import id_score_pairs
from kgqa.models.base import ScoreProducer
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


class WebQSPScoreProducer(ScoreProducer):
    def __init__(self, bert_name: str = "bert-base-uncased"):
        self.bert_name = bert_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ckpt_path: str | None = None

    def load_checkpoint(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle:
        assert self._ckpt_path, "先调用 load_checkpoint()"
        ent2id, rel2id, triples, _train, _val = load_data(input_dir, self.bert_name, batch_size)
        loader = DataLoader(input_dir, qa_file, self.bert_name, ent2id, rel2id, batch_size)
        args = SimpleNamespace(bert_name=self.bert_name)  # TransferNet 需要的最小 args
        model = TransferNet(args, ent2id, rel2id, triples)
        model.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"), strict=False)
        model = model.to(self.device)
        for attr in ("Msubj", "Mobj", "Mrel"):
            setattr(model, attr, getattr(model, attr).to(self.device))
        model.eval()

        raw_questions = getattr(loader, "qa_text", None)
        assert raw_questions is not None, "DataLoader 缺 qa_text"

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
                    topic_ids = [x for (x, _) in id_score_pairs(batch[0][i], 1)]
                    gold_ids = [x for (x, _) in id_score_pairs(batch[2][i], 1)]
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
                    ))
        meta = CacheMeta(dataset="WebQSP", split=split, id2ent=loader.id2ent,
                         id2rel=loader.id2rel, num_samples=len(samples),
                         topk_entities=topk, input_dir=input_dir, qa_file=qa_file)
        return ScoreBundle(meta=meta, samples=samples)
```

> 若 `TransferNet.__init__` 需要的 `args` 字段多于 `bert_name`，实现时对照 `WebQSP/train.py` 补齐 `SimpleNamespace` 字段（如 `dim_hidden/num_steps`）——以实际构造报错为准补全，保持与 `dump_scores.py` 一致的 `args` 来源（`dump_scores` 直接用 argparse 的 `args`，此处用等价 namespace）。

`kgqa/retrieve/backends/online.py`：
```python
"""在线后端：ScoreProducer 实时前向 → 同一 engine。"""
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
        self.edge_source = adapter.kg_edge_source()

    def retrieve(self, sample_index: int, **params):
        merged = {**RetrieveParams().as_kwargs(), **params}
        return engine.retrieve_one(self.bundle.samples[sample_index], self.edge_source,
                                   self.bundle.meta.id2ent, self.bundle.meta.id2rel, **merged)

    def retrieve_all(self, *, limit: int = 0, **params):
        merged = {**RetrieveParams().as_kwargs(), **params}
        samples = self.bundle.samples[:limit] if limit else self.bundle.samples
        return [engine.retrieve_one(s, self.edge_source, self.bundle.meta.id2ent,
                                    self.bundle.meta.id2rel, **merged) for s in samples]
```

创建空 `kgqa/models/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_backend_online -v`
Expected: PASS（有 ckpt+GPU 则 1 test；否则 skip）。若本机无 ckpt，至少确认 `python -c "import kgqa.models.webqsp, kgqa.retrieve.backends.online"` 无 import 错误。

- [ ] **Step 5: Commit**

```bash
git add kgqa/models/ kgqa/retrieve/backends/online.py tests/kgqa/test_backend_online.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增 ScoreProducer 模型接口与在线后端

- kgqa/models/base.py: ScoreProducer 接口
- kgqa/models/webqsp.py: WebQSPScoreProducer（前向产 ScoreBundle）
- kgqa/retrieve/backends/online.py: OnlineBackend（实时前向→同一 engine）
- tests/kgqa/test_backend_online.py: 在线后端 smoke（ckpt 存在才跑）

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: 统一 CLI（dump_scores / retrieve / eval）

**Files:**
- Create: `kgqa/cli/__init__.py`（空）, `kgqa/cli/dump_scores.py`, `kgqa/cli/retrieve.py`, `kgqa/cli/eval.py`, `tests/kgqa/test_cli.py`

**Interfaces:**
- Consumes: `get_adapter`（Task 5）、`OfflineBackend`（Task 7）、`WebQSPScoreProducer`（Task 8）、`answer_summary/path_summary`（Task 6）
- Produces:
  - `kgqa/cli/dump_scores.py`：`--dataset --ckpt --input_dir --qa_file --split --output --topk --batch_size`；调用 producer 产 `ScoreBundle` 后 `torch.save` 成与旧格式兼容的 dict（供 offline 复用）
  - `kgqa/cli/retrieve.py`：`--dataset --backend {offline,online} --cache/--ckpt --input_dir --qa_file --beam_size --method --lambda_val --threshold --alpha_final --limit --output(jsonl)`；写逐样本 jsonl
  - `kgqa/cli/eval.py`：在 retrieve 基础上，调用 `answer_summary`（用 prediction 当 pred）+ `path_summary`，写 `summary.json`
  - 三文件均提供 `build_parser()` 与 `main(argv=None)`，`main` 可被测试直接调用

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_cli.py`

```python
import os
import json
import tempfile
import unittest

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestCLI(unittest.TestCase):
    def test_parsers_build(self):
        from kgqa.cli import retrieve, eval as eval_cli, dump_scores
        self.assertIsNotNone(retrieve.build_parser())
        self.assertIsNotNone(eval_cli.build_parser())
        self.assertIsNotNone(dump_scores.build_parser())

    @unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
    def test_eval_writes_summary(self):
        from kgqa.cli import eval as eval_cli
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "summary.json")
            eval_cli.main([
                "--dataset", "webqsp", "--backend", "offline",
                "--cache", CACHE, "--input_dir", "data/input/WebQSP",
                "--limit", "20", "--beam_size", "50", "--summary", out,
            ])
            with open(out, encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertIn("answer", summary)
            self.assertIn("path", summary)
            self.assertIn("overall", summary["answer"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_cli -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.cli'`）

- [ ] **Step 3: 写实现**

`kgqa/cli/retrieve.py`（核心，`eval.py` 复用其 `build_backend`）：
```python
"""统一检索 CLI。"""
from __future__ import annotations

import argparse
import json
import os

from kgqa.datasets.registry import get_adapter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一路径检索")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["offline", "online"], default="offline")
    p.add_argument("--cache", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument("--method", default="tail_blend")
    p.add_argument("--lambda_val", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--alpha_final", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=None, help="逐样本 jsonl")
    return p


def build_backend(args):
    adapter = get_adapter(args.dataset, input_dir=args.input_dir)
    if args.backend == "offline":
        from kgqa.retrieve.backends.offline import OfflineBackend
        if not args.cache:
            raise SystemExit("--backend offline 需要 --cache")
        return adapter, OfflineBackend(adapter, cache_path=args.cache)
    from kgqa.models.webqsp import WebQSPScoreProducer
    from kgqa.retrieve.backends.online import OnlineBackend
    if not (args.ckpt and args.qa_file):
        raise SystemExit("--backend online 需要 --ckpt 和 --qa_file")
    backend = OnlineBackend(adapter, WebQSPScoreProducer(), ckpt_path=args.ckpt,
                            input_dir=args.input_dir, qa_file=args.qa_file,
                            split=args.split, limit=args.limit)
    return adapter, backend


def run_retrieval(args):
    _adapter, backend = build_backend(args)
    params = dict(beam_size=args.beam_size, method=args.method, lambda_val=args.lambda_val,
                  threshold=args.threshold, alpha_final=args.alpha_final)
    results = backend.retrieve_all(limit=args.limit, **params)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "sample_index": r.sample_index, "question": r.question,
                    "topics": r.topics, "hop": r.hop, "mmr_reason_paths": r.paths,
                    "prediction": r.prediction,
                }, ensure_ascii=False) + "\n")
    return backend, results


def main(argv=None):
    args = build_parser().parse_args(argv)
    _backend, results = run_retrieval(args)
    print(f"[INFO] 检索完成 {len(results)} 条", flush=True)


if __name__ == "__main__":
    main()
```

`kgqa/cli/eval.py`：
```python
"""统一评测 CLI：检索 + 答案级/路径级指标。"""
from __future__ import annotations

import argparse
import json
import os

from kgqa.cli.retrieve import build_parser as _retrieve_parser, run_retrieval
from kgqa.eval.answer_eval import answer_record, answer_summary
from kgqa.eval.path_eval import path_summary


def build_parser() -> argparse.ArgumentParser:
    p = _retrieve_parser()
    p.description = "kgqa 统一评测"
    p.add_argument("--summary", default=None, help="summary.json 输出路径")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    backend, results = run_retrieval(args)
    adapter = backend.adapter
    spec = adapter.metric_spec()
    id2rel = backend.bundle.meta.id2rel

    # gold（名称口径）：用 adapter.entity_name 映射 gold_ids
    gold_by_index: dict[int, set[str]] = {}
    ans_records = []
    for r, sample in zip(results, backend.bundle.samples):
        gold_names = {adapter.entity_name(str(g)) for g in sample.gold_ids}
        gold_by_index[r.sample_index] = gold_names
        pred_names = list(r.prediction.keys())  # prediction 已是名称口径（build_prediction 用 id2ent=MID；见注）
        ans_records.append(answer_record(pred=pred_names, gold=[adapter.entity_name(str(g)) for g in sample.gold_ids],
                                         hop=sample.hop, format_ok=True))

    summary = {
        "answer": answer_summary(ans_records, spec),
        "path": path_summary(results, gold_by_index, spec, id2rel=id2rel),
        "n": len(results),
    }
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary["answer"]["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
```

> **口径注（实现须处理）**：`engine.build_prediction` 用 `id2ent`（MID→MID，因 WebQSP 的 `id2ent` 值即 MID）产出 MID 键；而 `path_eval` 的 gold 与路径尾比较用的是 MID（路径三元组尾是 `id2ent[tail]`=MID）。因此**答案级与路径级统一用 MID 口径比对**：`eval.py` 中 gold 用 `str(g)`（MID）而非 `entity_name`，`pred` 用 `r.prediction` 的 MID 键。上面示例误用了 `entity_name`，实现时改为 MID 口径（WebQSP `gold_key="mid"`）；`entity_name` 仅用于展示层。MetaQA（`gold_key="name"`）时再走名称口径。此差异由 `spec.gold_key` 驱动，Task 6 的 eval 函数对字符串不敏感，只需保证 pred/gold 同口径。

`kgqa/cli/dump_scores.py`：
```python
"""统一 dump CLI：producer 产 ScoreBundle → 存兼容 .pt。"""
from __future__ import annotations

import argparse

import torch

from kgqa.datasets.registry import get_adapter  # noqa: F401  (保留扩展位)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一得分 dump")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    return p


def _bundle_to_cache(bundle) -> dict:
    meta = bundle.meta
    return {
        "version": 1,
        "meta": {"dataset": meta.dataset, "split": meta.split,
                 "num_samples": meta.num_samples, "topk_entities": meta.topk_entities,
                 "input_dir": meta.input_dir, "qa_file": meta.qa_file,
                 "id2ent": meta.id2ent, "id2rel": meta.id2rel},
        "samples": [{
            "question": s.question, "topic_ids": s.topic_ids, "gold_ids": s.gold_ids,
            "hop_attn": s.hop_attn, "rel_probs": s.rel_probs,
            "ent_indices": s.ent_indices, "ent_scores": s.ent_scores,
            "e_score_indices": s.e_score_indices, "e_score_values": s.e_score_values,
            **({"triples": s.triples} if s.triples is not None else {}),
        } for s in bundle.samples],
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dataset != "webqsp":
        raise SystemExit(f"stage1 仅支持 webqsp dump，收到: {args.dataset}")
    from kgqa.models.webqsp import WebQSPScoreProducer
    producer = WebQSPScoreProducer()
    producer.load_checkpoint(args.ckpt)
    bundle = producer.produce(args.input_dir, args.qa_file, split=args.split,
                              batch_size=args.batch_size, topk=args.topk)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(_bundle_to_cache(bundle), args.output)
    print(f"[INFO] dump 完成 {len(bundle.samples)} 条 → {args.output}", flush=True)


if __name__ == "__main__":
    main()
```

创建空 `kgqa/cli/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_cli -v`
Expected: PASS（parser 测试恒通过；summary 测试缓存存在则通过，否则 skip）

- [ ] **Step 5: Commit**

```bash
git add kgqa/cli/ tests/kgqa/test_cli.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增统一 CLI（dump_scores/retrieve/eval）

- kgqa/cli/retrieve.py: 双后端检索入口 + build_backend
- kgqa/cli/eval.py: 答案级/路径级 summary
- kgqa/cli/dump_scores.py: producer→兼容 .pt
- tests/kgqa/test_cli.py: CLI 单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: 常驻检索服务（薄壳）

**Files:**
- Create: `kgqa/server/__init__.py`（空）, `kgqa/server/path_retrieve_server.py`, `tests/kgqa/test_server.py`

**Interfaces:**
- Consumes: `OfflineBackend`（Task 7）、`RetrieveResult`（Task 1）；HTTP 用 FastAPI + `fastapi.testclient.TestClient`（与 `oh_my_agent` 现有依赖一致）
- Produces:
  - `create_app(backend) -> FastAPI`：`POST /retrieve {"sample_index": int, "beam_size": int, ...}` → `RetrieveResult` dict；`GET /health` → `{"status":"ok","n":int}`
  - `main(argv=None)`：`--dataset --backend --cache --input_dir --host --port`，启动 uvicorn

- [ ] **Step 1: 写失败测试** `tests/kgqa/test_server.py`

```python
import unittest
from kgqa.types import RetrieveResult


class _StubBackend:
    class _B:  # 模拟 bundle.samples 长度
        samples = [None, None, None]
    bundle = _B()

    def retrieve(self, sample_index, **params):
        return RetrieveResult(question="q", topics=["m.t"], hop=1,
                              paths=[{"path": [["m.t", "r", "m.a"]], "log_score": -0.1}],
                              prediction={"m.a": 0.9}, elapsed_ms=0.5, sample_index=sample_index)


class TestServer(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from kgqa.server.path_retrieve_server import create_app
        self.client = TestClient(create_app(_StubBackend()))

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["n"], 3)

    def test_retrieve(self):
        resp = self.client.post("/retrieve", json={"sample_index": 2, "beam_size": 10})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["sample_index"], 2)
        self.assertEqual(body["paths"][0]["log_score"], -0.1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.kgqa.test_server -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'kgqa.server'`）

- [ ] **Step 3: 写实现** `kgqa/server/path_retrieve_server.py`

```python
"""常驻检索服务（薄壳，持有一个后端）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    sample_index: int
    beam_size: int = 50
    method: str = "tail_blend"
    lambda_val: float = 0.2
    threshold: float = 0.01
    alpha_final: float = 1.0


def create_app(backend) -> FastAPI:
    app = FastAPI(title="kgqa path retrieve")

    @app.get("/health")
    def health():
        return {"status": "ok", "n": len(backend.bundle.samples)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        result = backend.retrieve(
            req.sample_index, beam_size=req.beam_size, method=req.method,
            lambda_val=req.lambda_val, threshold=req.threshold, alpha_final=req.alpha_final,
        )
        return asdict(result)

    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="kgqa 常驻检索服务")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["offline"], default="offline")
    p.add_argument("--cache", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8789)
    args = p.parse_args(argv)

    import uvicorn
    from kgqa.datasets.registry import get_adapter
    from kgqa.retrieve.backends.offline import OfflineBackend
    adapter = get_adapter(args.dataset, input_dir=args.input_dir)
    backend = OfflineBackend(adapter, cache_path=args.cache)
    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

创建空 `kgqa/server/__init__.py`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.kgqa.test_server -v`
Expected: PASS（2 tests）

- [ ] **Step 5: Commit**

```bash
git add kgqa/server/ tests/kgqa/test_server.py
git commit -m "$(cat <<'EOF'
feat(kgqa): 新增常驻检索服务薄壳

- kgqa/server/path_retrieve_server.py: create_app（/health,/retrieve）+ main
- tests/kgqa/test_server.py: TestClient 单测

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: 回归 + parity 验证（迁移保真锁）

**Files:**
- Create: `tests/kgqa/test_webqsp_regression.py`, `tests/kgqa/test_backend_parity.py`

**Interfaces:**
- Consumes: `scripts.offline_path_search.{load_score_cache, run_experiment, rebuild_valid_edges_dict}`（旧实现，作为 ground truth）、`OfflineBackend`（Task 7）、`OnlineBackend`（Task 8）
- Produces: 无新代码，仅测试

> **回归口径**：`OfflineBackend.retrieve_all` 对 `webqsp_test_1581.pt` 用 `method=tail_blend, alpha_final=1.0, threshold=0.01, beam_size=50, lambda_val=0.2` 检索；旧 `run_experiment` 用**同一缓存同一参数**（注意旧 `run_experiment` 无 `drop_loopback`，故新后端此对照须 `drop_loopback=False`）。逐样本比较「每条路径的 (nodes, rels) 序列」完全一致。这是免 ckpt 的强保真锁。

- [ ] **Step 1: 写回归测试** `tests/kgqa/test_webqsp_regression.py`

```python
import os
import unittest

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"


@unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
class TestWebQSPRegression(unittest.TestCase):
    def test_offline_paths_match_legacy(self):
        from scripts.offline_path_search import (
            load_score_cache, rebuild_valid_edges_dict, _method_hop_numbers,
            reconstruct_rel_dict, reconstruct_ent_dict, LogNormStrategy,
            search_path_candidates, select_path_candidates, candidate_to_tuple,
            final_ent_score_dict,
        )
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend

        params = dict(method="tail_blend", alpha_final=1.0, threshold=0.01,
                      beam_size=50, lambda_val=0.2)

        # 旧实现：直接复算前 N 条的选中路径 (nodes, rels)
        cache = load_score_cache(CACHE)
        ved = rebuild_valid_edges_dict(INPUT_DIR)
        N = 50

        def legacy_paths(sample):
            hop_num = int(sample["hop_attn"].argmax().item()) + 1
            hop_nums = _method_hop_numbers("tail_blend", hop_num, len(sample["rel_probs"]))
            rel_dicts, ent_dicts = [], []
            for t in range(max(hop_nums)):
                rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], 0.01))
                ent_dicts.append(reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], 0.01))
            scoring = LogNormStrategy()
            final_scores = final_ent_score_dict(sample)
            cands = []
            for ch in hop_nums:
                cands.extend(search_path_candidates(sample["topic_ids"], rel_dicts, ent_dicts, ch,
                                                    ved, scoring, 50, final_ent_scores=final_scores,
                                                    order_start=len(cands)))
            selected = select_path_candidates(cands, 50, method="tail_blend", alpha_final=1.0, lambda_val=0.2)
            return [candidate_to_tuple(c)[:2] for c in selected]

        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        backend = OfflineBackend(adapter, cache_path=CACHE)

        for i in range(N):
            legacy = legacy_paths(cache["samples"][i])
            r = backend.retrieve(i, drop_loopback=False, **params)
            new = [(p_nodes, p_rels) for (p_nodes, p_rels) in
                   [(_nodes_from(p), _rels_from(p)) for p in r.paths]]
            self.assertEqual(len(new), len(legacy), f"sample {i} 路径数不一致")
            # 逐条比对 rels 序列（nodes 与 rels 一一对应）
            self.assertEqual([rl for _, rl in new], [rl for _, rl in legacy], f"sample {i} rels 不一致")


def _rels_from(path_dict):
    return [e[1] for e in path_dict["path"]]


def _nodes_from(path_dict):
    edges = path_dict["path"]
    return ([edges[0][0]] + [e[2] for e in edges]) if edges else []


if __name__ == "__main__":
    unittest.main()
```

> 注：新后端序列化后是 MID/rel 名称串，legacy `candidate_to_tuple` 是 id。比对 **rels 名称序列**即可（新 = `id2rel[rid]`，legacy 需同样映射）。实现时在 legacy 侧用 `cache["meta"]["id2rel"]` 把 rels id 映射成名称再比对，保证两侧同口径；若逐条名称序列一致即判定保真。

- [ ] **Step 2: 运行回归（缓存存在）**

Run: `python -m unittest tests.kgqa.test_webqsp_regression -v`
Expected: PASS（前 50 条路径 rels 序列逐条一致）；若 FAIL 说明迁移改变了数值/顺序，须回到 Task 3 核对逐字迁移。

- [ ] **Step 3: 写 parity 测试** `tests/kgqa/test_backend_parity.py`

```python
import os
import unittest

CKPT = "data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt"
CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"
QA = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(CACHE) and os.path.isfile(QA),
                     "ckpt/缓存/QA 缺失，跳过")
class TestBackendParity(unittest.TestCase):
    def test_online_matches_offline_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        from kgqa.models.webqsp import WebQSPScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend

        params = dict(beam_size=50, method="tail_blend", lambda_val=0.2,
                      threshold=0.01, alpha_final=1.0)
        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        offline = OfflineBackend(adapter, cache_path=CACHE)
        online = OnlineBackend(adapter, WebQSPScoreProducer(), ckpt_path=CKPT,
                               input_dir=INPUT_DIR, qa_file=QA, split="test", limit=3)

        for i in range(3):
            ro = offline.retrieve(i, **params)
            rn = online.retrieve(i, **params)
            rels_o = [[e[1] for e in p["path"]] for p in ro.paths]
            rels_n = [[e[1] for e in p["path"]] for p in rn.paths]
            self.assertEqual(rels_n, rels_o, f"sample {i} online/offline rels 不一致")


if __name__ == "__main__":
    unittest.main()
```

> 前提：离线缓存 `webqsp_test_1581.pt` 由**同一 ckpt** dump 得到。若缓存来自不同 ckpt，parity 可能因浮点/topk 截断有微差——此时把断言放宽为「gold 命中一致 + 路径数差 ≤1」，并在测试注释标注缓存与 ckpt 来源。实现时先确认二者同源（`data/output/.../path_retrieve_server` 缓存对应 `model-49-0.7154.pt`）。

- [ ] **Step 4: 运行 parity（有 ckpt+GPU）**

Run: `python -m unittest tests.kgqa.test_backend_parity -v`
Expected: PASS（前 3 条 online/offline rels 一致）；无 ckpt 环境自动 skip。

- [ ] **Step 5: 跑全套 kgqa 测试 + Commit**

Run: `python -m unittest discover -s tests/kgqa -p 'test*.py' -v`
Expected: 全绿（无 ckpt/缓存的用例 skip，其余 PASS）

```bash
git add tests/kgqa/test_webqsp_regression.py tests/kgqa/test_backend_parity.py
git commit -m "$(cat <<'EOF'
test(kgqa): 新增迁移回归与在线/离线 parity 验证

- tests/kgqa/test_webqsp_regression.py: 离线路径与旧 run_experiment 逐条一致
- tests/kgqa/test_backend_parity.py: online 与 offline 路径一致（ckpt 存在才跑）

Co-Authored-By: <从 git config 读取> <<从 git config 读取>>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## 验收标准（Definition of Done）

1. `python -m unittest discover -s tests/kgqa -p 'test*.py'` 全绿（无资源用例 skip）。
2. `python -m kgqa.cli.eval --dataset webqsp --backend offline --cache <webqsp_test_1581.pt> --input_dir data/input/WebQSP --beam_size 50 --summary /tmp/s.json` 产出含 `answer.overall`、`path.overall` 的 summary。
3. 回归测试证明离线检索路径与旧 `offline_path_search` 逐条一致。
4. `kgqa/` 结构与本 plan 文件树一致；旧代码未删除、未修改。
5. Plan 1 完成后，Plan 2（MetaQA）/Plan 3（CWQ）只需新增各自 `datasets/models/scores` 实现并注册，不改 `engine`/`eval`/`backends`。

## Self-Review 记录

- **Spec 覆盖**：§2 结构→文件树；§3 三接口→Task 2/4/5；§3.4 ScoreProducer→Task 8；§4 双后端+parity→Task 7/8/11；§5 指标（全套+by_hop）→Task 6；§6 数据缺口→本 plan 仅 WebQSP（MetaQA/CWQ 缺口留 Plan 2/3，spec §6 明列）；§7 测试→Task 11 + 各 Task 单测。
- **占位扫描**：无 TODO/TBD；`Co-Authored-By` 占位是**故意**留给执行者用 `git config` 填（Global Constraints 已强制），非内容占位。
- **类型一致性**：`SampleScore` 字段在 Task 4 定义，Task 3 的 `retrieve_one`、Task 8 的 producer、回归测试均按同名字段消费；`RetrieveResult` 字段在 Task 1 定义，server/CLI/eval 一致引用；`RetrieveParams` 字段与 `retrieve_one` 关键字对齐。
- **已知实现注意点（非占位，执行者须落实）**：① Task 8 `SimpleNamespace(args)` 字段可能需按 `TransferNet.__init__` 补齐；② Task 9 eval 的 pred/gold 口径须按 `spec.gold_key` 统一为 MID（注释已明确）；③ Task 11 parity 依赖缓存与 ckpt 同源，否则放宽断言。
