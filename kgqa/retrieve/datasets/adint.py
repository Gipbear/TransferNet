"""名称原生 ADInt 数据集适配器。"""

from __future__ import annotations

import json

from kgqa.core.contracts import MetricSpec, QASample, ScoreLoader
from kgqa.retrieve.cache.adint import ADIntScoreLoader
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.global_kg import GlobalKG


class ADIntAdapter(DatasetAdapter):
    name = "adint"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/ADInt"):
        self.input_dir = input_dir
        self._kg: GlobalKG | None = None

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        with open(path, encoding="utf-8") as source:
            rows = json.load(source)
        samples: list[QASample] = []
        for row in rows:
            topic = row.get("topic_entity")
            samples.append(QASample(
                question=row["question"],
                topic_ids=[topic] if topic else [],
                gold_ids=list(row.get("answers", [])),
                sample_index=len(samples),
                hop=row.get("hop"),
                extra={"question_id": row.get("question_id"), "topic_entity": topic},
            ))
            if limit and len(samples) >= limit:
                break
        return samples

    def entity_name(self, entity_id: str) -> str:
        return entity_id

    def kg_edge_source(self, sample=None) -> GlobalKG:
        if self._kg is None:
            self._kg = GlobalKG.from_adint_tsv(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return ADIntScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="name", group_by="hop", answer_metrics=True, path_metrics=True)
