"""题库索引与回放轨迹的加载。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from kgqa.agent.common.qa_data import WebQSPQASample, load_webqsp_qa_samples


@dataclass(frozen=True)
class QuestionEntry:
    sample_index: int
    question: str


class QuestionIndex:
    """1581 题的内存索引：下拉联想用子串搜索，回放用下标定位。"""

    def __init__(self, samples: list[WebQSPQASample]) -> None:
        self._samples = list(samples)

    @classmethod
    def from_file(cls, path: str) -> "QuestionIndex":
        return cls(load_webqsp_qa_samples(path))

    def __len__(self) -> int:
        return len(self._samples)

    def get(self, sample_index: int) -> WebQSPQASample:
        if not 0 <= sample_index < len(self._samples):
            raise IndexError(f"sample_index 越界: {sample_index}")
        return self._samples[sample_index]

    def search(self, query: str, limit: int = 20) -> list[QuestionEntry]:
        needle = query.strip().lower()
        hits: list[QuestionEntry] = []
        for i, sample in enumerate(self._samples):
            if not needle or needle in sample.question.lower():
                hits.append(QuestionEntry(i, sample.question))
                if len(hits) >= limit:
                    break
        return hits


def load_trace_index(path: str) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                index[record["sample_index"]] = record
    return index
