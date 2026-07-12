"""离线后端：读得分缓存 → engine（edge source 逐样本分发）。"""
from __future__ import annotations

from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve import engine
from kgqa.retrieve.backends.base import RetrieveBackend, RetrieveParams
from kgqa.core.contracts import RetrieveResult


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
