"""ReaRev-WebQSP 离线得分缓存的检索适配器。"""
from __future__ import annotations

from kgqa.core.contracts import MetricSpec, QASample, ScoreBundle, ScoreLoader
from kgqa.retrieve.cache.base import score_bundle_from_cache
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.global_kg import GlobalKG


class ReaRevWebQSPScoreLoader(ScoreLoader):
    def load(self, cache_path: str) -> ScoreBundle:
        return score_bundle_from_cache(cache_path, "webqsp-rearev")


class ReaRevWebQSPAdapter(DatasetAdapter):
    """复用 ReaRev dump 的逐样本子图和多跳得分，接入 offline backend。"""

    name = "webqsp-rearev"
    max_hop = 3  # ReaRev num_gnn=3，dump 缓存含 3 个 hop 的分布

    def __init__(self, input_dir: str = ""):
        self.input_dir = input_dir

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        raise NotImplementedError("离线链路的 QA 信息都在得分缓存里，无需单独加载")

    def entity_name(self, entity_id: str) -> str:
        return entity_id  # MID 口径，同 WebQSP/CWQ

    def kg_edge_source(self, sample=None) -> GlobalKG:
        triples = getattr(sample, "triples", None)
        if triples is None:
            raise ValueError("webqsp-rearev 为逐样本子图，kg_edge_source 需要带 triples 的 sample")
        return GlobalKG.from_triples(triples)

    def score_loader(self) -> ScoreLoader:
        return ReaRevWebQSPScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None,
                          answer_metrics=True, path_metrics=True)
