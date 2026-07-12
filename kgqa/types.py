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
    golden: list[str] = field(default_factory=list)   # gold_ids 经 id2ent 还原,与 topics/paths 同空间


@dataclass(frozen=True)
class MetricSpec:
    gold_key: str = "mid"                # "mid" | "name"
    group_by: Optional[str] = None       # None | "hop"
    answer_metrics: bool = True
    path_metrics: bool = True
