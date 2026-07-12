"""kgqa 跨能力域共享的数据契约。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class QASample:
    question: str
    topic_ids: list[int]
    gold_ids: list[int]
    sample_index: int = -1
    hop: Optional[int] = None
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
    paths: list[dict]
    prediction: dict[str, float]
    elapsed_ms: float
    sample_index: int = -1
    golden: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetricSpec:
    gold_key: str = "mid"
    group_by: Optional[str] = None
    answer_metrics: bool = True
    path_metrics: bool = True


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


@dataclass
class ScoreBundle:
    meta: CacheMeta
    samples: list[SampleScore]


class ScoreLoader(ABC):
    @abstractmethod
    def load(self, cache_path: str) -> ScoreBundle: ...


class ScoreProducer(ABC):
    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> None: ...

    @abstractmethod
    def produce(self, input_dir: str, qa_file: str, *, split: str = "test",
                batch_size: int = 16, topk: int = 500) -> ScoreBundle: ...
