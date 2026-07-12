"""WebQSP 适配器。"""
from __future__ import annotations

from kgqa.agent.common.qa_data import parse_webqsp_qa_line
from kgqa.agent.common.entity_mapping import load_entity_map
from kgqa.datasets.base import DatasetAdapter
from kgqa.kg.global_kg import GlobalKG
from kgqa.scores.base import ScoreLoader
from kgqa.scores.webqsp import WebQSPScoreLoader
from kgqa.types import MetricSpec, QASample

DEFAULT_ENTITY_MAP = "data/resources/WebQSP/fbwq_full/mid2name.txt"


class WebQSPAdapter(DatasetAdapter):
    name = "webqsp"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/WebQSP", entity_map_path: str | None = None):
        self.input_dir = input_dir
        self.entity_map_path = entity_map_path or DEFAULT_ENTITY_MAP
        self._entity_map: dict[str, str] | None = None
        self._kg: GlobalKG | None = None

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        samples: list[QASample] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                parsed = parse_webqsp_qa_line(line)
                samples.append(QASample(
                    question=parsed.question,
                    topic_ids=[parsed.topic_mid] if parsed.topic_mid else [],
                    gold_ids=list(parsed.gold_mids),
                    sample_index=len(samples),
                    extra={"topic_mid": parsed.topic_mid, "question_raw": parsed.question_raw},
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
        return WebQSPScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None, answer_metrics=True, path_metrics=True)
