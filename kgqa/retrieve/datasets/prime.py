"""Prime 原生实体 ID 数据集适配器。"""

from __future__ import annotations

import json
from pathlib import Path

from kgqa.core.contracts import MetricSpec, QASample, ScoreLoader
from kgqa.retrieve.cache.prime import PrimeScoreLoader
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.prime import PrimeKG


class PrimeAdapter(DatasetAdapter):
    name = "prime"
    max_hop = 3

    def __init__(self, input_dir: str = "data/input/Prime"):
        self.input_dir = input_dir
        self._kg: PrimeKG | None = None
        names_path = Path(input_dir) / "entity_names.json"
        self._entity_names: list[str] = json.loads(names_path.read_text(encoding="utf-8"))

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if limit:
            rows = rows[:limit]
        return [
            QASample(
                question=row["question"],
                topic_ids=[int(row["topic_entity"])],
                gold_ids=[int(answer) for answer in row["answers"]],
                sample_index=index,
                hop=row["hop"],
                extra={"question_id": row["question_id"]},
            )
            for index, row in enumerate(rows)
        ]

    def entity_name(self, entity_id: str) -> str:
        return self._entity_names[int(entity_id)]

    def kg_edge_source(self, sample=None) -> PrimeKG:
        if self._kg is None:
            self._kg = PrimeKG(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return PrimeScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by="hop", answer_metrics=True, path_metrics=True)
