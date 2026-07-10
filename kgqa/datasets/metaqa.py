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

    def kg_edge_source(self, sample=None) -> GlobalKG:
        if self._kg is None:
            self._kg = GlobalKG.from_metaqa_npy(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return MetaQAScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="name", group_by="hop",
                          answer_metrics=True, path_metrics=True)
