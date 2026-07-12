"""CWQ 适配器（MID 口径、2-hop、逐样本子图）。"""
from __future__ import annotations

import json

from kgqa.core.contracts import MetricSpec, QASample, ScoreLoader
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.global_kg import GlobalKG
from kgqa.retrieve.cache.cwq import CWQScoreLoader


class CWQAdapter(DatasetAdapter):
    name = "cwq"
    max_hop = 2

    def __init__(self, input_dir: str = "data/input/CWQ"):
        self.input_dir = input_dir

    def load_qa(self, path: str, limit: int = 0) -> list[QASample]:
        samples: list[QASample] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("subgraph", {}).get("tuples"):
                    continue  # 与 CompWebQ DataLoader 跳过空子图的规则对齐
                samples.append(QASample(
                    question=item["question"].strip(),
                    topic_ids=[int(e) for e in item.get("entities", [])],
                    gold_ids=[a["kb_id"] for a in item.get("answers", [])],
                    sample_index=len(samples),
                    extra={"id": item.get("id")},
                ))
                if limit and len(samples) >= limit:
                    break
        return samples

    def entity_name(self, entity_id: str) -> str:
        return entity_id  # MID 口径，同 WebQSP 不在 eval 链路做名字映射

    def kg_edge_source(self, sample=None) -> GlobalKG:
        triples = getattr(sample, "triples", None)
        if triples is None:
            raise ValueError("CWQ 为逐样本子图，kg_edge_source 需要带 triples 的 sample")
        return GlobalKG.from_triples(triples)

    def score_loader(self) -> ScoreLoader:
        return CWQScoreLoader()

    def metric_spec(self) -> MetricSpec:
        return MetricSpec(gold_key="mid", group_by=None,
                          answer_metrics=True, path_metrics=True)
