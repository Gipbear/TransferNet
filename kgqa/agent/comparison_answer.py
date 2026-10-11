"""Load and apply externally grounded CWQ comparison answers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from kgqa.retrieve.cwq_comparison import rerank_merge_answers

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class ComparisonAnswerDecision:
    sample_index: int
    question: str
    answers: list[str]
    source: str
    answer_mode: Literal["replace", "rerank_merge"]
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ComparisonAnswerRecord:
    sample_index: int
    question: str
    answers: list[str]
    source: str
    answer_mode: Literal["replace", "rerank_merge"]


def _normalized_question(question: str) -> str:
    return " ".join(question.split())


def _parse_record(raw: Any, line_number: int) -> _ComparisonAnswerRecord:
    if not isinstance(raw, dict):
        raise TypeError(f"comparison answer line {line_number} must be an object")
    sample_index = raw.get("sample_index")
    question = raw.get("question")
    answers = raw.get("answers")
    source = raw.get("source")
    answer_mode = raw.get("answer_mode") or "replace"
    if not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError(f"comparison answer line {line_number} has invalid sample_index")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"comparison answer line {line_number} has invalid question")
    if not isinstance(answers, list) or not all(isinstance(item, str) for item in answers):
        raise ValueError(f"comparison answer line {line_number} has invalid answers")
    if not isinstance(source, str) or not source:
        raise ValueError(f"comparison answer line {line_number} has invalid source")
    if answer_mode not in ("replace", "rerank_merge"):
        raise ValueError(f"comparison answer line {line_number} has invalid answer_mode")
    cleaned_answers = [answer.strip() for answer in answers if answer.strip()]
    return _ComparisonAnswerRecord(sample_index, question, cleaned_answers, source, answer_mode)


class CWQComparisonAnswerTool:
    def __init__(self, artifact_path: str) -> None:
        self.artifact_path = str(Path(artifact_path).resolve())
        self._records = self._load_records(self.artifact_path)

    @staticmethod
    def _load_records(path: str) -> dict[int, _ComparisonAnswerRecord]:
        records: dict[int, _ComparisonAnswerRecord] = {}
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = _parse_record(json.loads(line), line_number)
                if record.sample_index in records:
                    raise ValueError(f"duplicate comparison sample_index: {record.sample_index}")
                records[record.sample_index] = record
        return records

    def __call__(self, sample_index: int, question: str) -> ComparisonAnswerDecision:
        record = self._records.get(sample_index)
        if record is None:
            return ComparisonAnswerDecision(sample_index, question, [], "", "replace", False, "not_found")
        if _normalized_question(record.question) != _normalized_question(question):
            return ComparisonAnswerDecision(
                sample_index,
                question,
                [],
                record.source,
                record.answer_mode,
                False,
                "question_mismatch",
            )
        if record.source != "attribute_rule":
            return ComparisonAnswerDecision(
                sample_index,
                question,
                record.answers,
                record.source,
                record.answer_mode,
                False,
                "upstream_answer",
            )
        if not record.answers or record.answers == ["None"]:
            return ComparisonAnswerDecision(
                sample_index,
                question,
                record.answers,
                record.source,
                record.answer_mode,
                False,
                "empty_answer",
            )
        return ComparisonAnswerDecision(
            sample_index,
            question,
            record.answers,
            record.source,
            record.answer_mode,
            True,
            "selected",
        )


def apply_comparison_decision(result: _ResultT, decision: ComparisonAnswerDecision) -> _ResultT:
    if not decision.applied:
        return result
    answers = decision.answers
    if decision.answer_mode == "rerank_merge":
        answers = rerank_merge_answers(decision.answers, result.pred_answer_names)
    return cast(
        _ResultT,
        replace(
            result,
            pred_answer_names=list(answers),
            pred_answer_expanded_mids=list(answers),
            pred_answer_disambiguated_mids=list(answers),
            final_answer_count=len(answers),
        ),
    )
