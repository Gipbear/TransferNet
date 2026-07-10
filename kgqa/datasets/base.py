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
    def kg_edge_source(self, sample=None) -> KGEdgeSource:
        """sample 为鸭子类型：逐样本子图数据集（CWQ）传带 .triples 的 SampleScore；
        全局图数据集（WebQSP/MetaQA）忽略该参数。"""

    @abstractmethod
    def score_loader(self) -> ScoreLoader: ...

    @abstractmethod
    def metric_spec(self) -> MetricSpec: ...
