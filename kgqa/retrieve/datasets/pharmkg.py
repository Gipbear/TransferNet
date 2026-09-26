"""PharmKG 数据集适配器。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final, Literal, Mapping, TypedDict, assert_never

from kgqa.core.contracts import MetricSpec, QASample, ScoreLoader
from kgqa.core.entity_map import load_entity_map
from kgqa.core.qa_formats import parse_webqsp_qa_line
from kgqa.retrieve.cache.pharmkg import PharmKGScoreLoader
from kgqa.retrieve.datasets.base import DatasetAdapter
from kgqa.retrieve.graph.global_kg import GlobalKG

AnswerType = Literal["gene", "disease", "chemical"]


class ReasonPath(TypedDict):
    path: list[list[str]]
    log_score: float


TYPE_CUE_PATTERNS: Final[tuple[tuple[AnswerType, re.Pattern[str]], ...]] = (
    ("gene", re.compile(r"\b(?:genes?|genetic)\b", re.IGNORECASE)),
    ("disease", re.compile(r"\b(?:diseases?|disorders?|syndromes?)\b", re.IGNORECASE)),
    ("chemical", re.compile(r"\b(?:chemicals?|drugs?|compounds?|medications?)\b", re.IGNORECASE)),
)
BIOMARKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bbiomarkers?\b", re.IGNORECASE)


def infer_answer_type(question: str) -> AnswerType | None:
    """仅在医学实体类型线索唯一且可靠时返回答案类型。"""
    inferred = {answer_type for answer_type, pattern in TYPE_CUE_PATTERNS if pattern.search(question)}
    if len(inferred) != 1:
        return None
    answer_type = next(iter(inferred))
    match answer_type:
        case "disease" if BIOMARKER_PATTERN.search(question):
            return None
        case "gene" | "disease" | "chemical":
            return answer_type
        case unreachable:
            assert_never(unreachable)


def rerank_and_dedupe_paths(
    question: str,
    paths: list[ReasonPath],
    entity_types: Mapping[str, str],
    *,
    limit: int,
) -> list[ReasonPath]:
    """按可靠答案类型稳定重排，再按尾实体去重并截断。"""
    if limit <= 0:
        return []
    answer_type = infer_answer_type(question)
    ranked = paths
    if answer_type is not None:
        matched = [path for path in paths if entity_types.get(path["path"][-1][-1]) == answer_type]
        unmatched = [path for path in paths if entity_types.get(path["path"][-1][-1]) != answer_type]
        ranked = matched + unmatched

    selected: list[ReasonPath] = []
    seen_tails: set[str] = set()
    for path in ranked:
        tail = path["path"][-1][-1]
        if tail in seen_tails:
            continue
        seen_tails.add(tail)
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def load_entity_types(path: str | Path) -> dict[str, str]:
    """加载转换后 PharmKG 实体 token 到规范类型的映射。"""
    entity_types: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            entity_types[str(item["token"])] = str(item["type"])
    return entity_types


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
