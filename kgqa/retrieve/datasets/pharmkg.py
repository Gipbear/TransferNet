"""PharmKG 数据集适配器。"""
from __future__ import annotations

from pathlib import Path

from kgqa.core.contracts import MetricSpec, QASample, ScoreLoader
from kgqa.core.entity_map import load_entity_map
from kgqa.core.qa_formats import parse_webqsp_qa_line
from kgqa.retrieve.cache.pharmkg import PharmKGScoreLoader
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.global_kg import GlobalKG


class PharmKGAdapter(DatasetAdapter):
    name = "pharmkg"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/PharmKG", entity_map_path: str | None = None):
        self.input_dir = input_dir
        self.entity_map_path = entity_map_path or str(Path(input_dir) / "fbwq_full" / "mid2name.txt")
        self._entity_map: dict[str, str] | None = None
        self._kg: GlobalKG | None = None

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        samples: list[QASample] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = parse_webqsp_qa_line(line)
                samples.append(QASample(
                    question=parsed.question,
                    topic_ids=[parsed.topic_mid] if parsed.topic_mid else [],
                    gold_ids=list(parsed.gold_mids),
                    sample_index=len(samples),
                    extra={"topic_id": parsed.topic_mid, "question_raw": parsed.question_raw},
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

    def kg_edge_source(self, sample=None) -> GlobalKG:
        if self._kg is None:
            self._kg = GlobalKG.from_input_dir(self.input_dir)
        return self._kg

    def score_loader(self) -> ScoreLoader:
        return PharmKGScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None, answer_metrics=True, path_metrics=True)
