"""在线后端：ScoreProducer 实时前向 → 同一 engine（edge source 逐样本分发）。"""
from __future__ import annotations

from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.backbone.base import ScoreProducer
from kgqa.retrieve import engine
from kgqa.retrieve.backends.base import RetrieveParams


class OnlineBackend:
    def __init__(self, adapter: DatasetAdapter, producer: ScoreProducer, *,
                 ckpt_path: str, input_dir: str, qa_file: str,
                 split: str = "test", batch_size: int = 16, topk: int = 500):
        producer.load_checkpoint(ckpt_path)
        self.adapter = adapter
        self.bundle = producer.produce(input_dir, qa_file, split=split,
                                       batch_size=batch_size, topk=topk)

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
