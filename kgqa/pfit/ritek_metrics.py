"""按 RiTeK 论文表格口径复算 Exact Match 与 ROUGE-1 指标。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict


TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", flags=re.UNICODE)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    llm_pred: list[str]
    llm_pred_disambiguated_mids: list[str] | None = None
    golden: list[str]

    def effective_prediction(self) -> Sequence[str]:
        if self.llm_pred_disambiguated_mids is not None:
            return self.llm_pred_disambiguated_mids
        return self.llm_pred


@dataclass(frozen=True, slots=True)
class Score:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class RiTeKMetrics:
    n: int
    exact_match: Score
    rouge1: Score


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def _normalized_entities(entities: Sequence[str]) -> set[str]:
    return {entity.casefold().strip() for entity in entities if entity.strip()}


def score_exact_match(predicted: Sequence[str], golden: Sequence[str]) -> Score:
    predicted_set = _normalized_entities(predicted)
    golden_set = _normalized_entities(golden)
    if not predicted_set and not golden_set:
        return Score(precision=1.0, recall=1.0, f1=1.0)
    overlap = len(predicted_set & golden_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(golden_set) if golden_set else 0.0
    return Score(precision=precision, recall=recall, f1=_f1(precision, recall))


def _unigrams(entities: Sequence[str]) -> Counter[str]:
    text = "\n".join(entities).casefold()
    return Counter(TOKEN_PATTERN.findall(text))


def score_rouge1(predicted: Sequence[str], golden: Sequence[str]) -> Score:
    predicted_tokens = _unigrams(predicted)
    golden_tokens = _unigrams(golden)
    if not predicted_tokens and not golden_tokens:
        return Score(precision=1.0, recall=1.0, f1=1.0)
    overlap = sum((predicted_tokens & golden_tokens).values())
    predicted_count = sum(predicted_tokens.values())
    golden_count = sum(golden_tokens.values())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / golden_count if golden_count else 0.0
    return Score(precision=precision, recall=recall, f1=_f1(precision, recall))


def _mean_score(scores: Sequence[Score]) -> Score:
    count = len(scores)
    if count == 0:
        return Score(precision=0.0, recall=0.0, f1=0.0)
    return Score(
        precision=sum(score.precision for score in scores) / count,
        recall=sum(score.recall for score in scores) / count,
        f1=sum(score.f1 for score in scores) / count,
    )


def aggregate_ritek_metrics(records: Sequence[tuple[Sequence[str], Sequence[str]]]) -> RiTeKMetrics:
    exact_scores = [score_exact_match(predicted, golden) for predicted, golden in records]
    rouge_scores = [score_rouge1(predicted, golden) for predicted, golden in records]
    return RiTeKMetrics(
        n=len(records),
        exact_match=_mean_score(exact_scores),
        rouge1=_mean_score(rouge_scores),
    )


def _load_entity_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            entity_id, separator, name = line.rstrip("\n").partition("\t")
            if separator:
                mapping[entity_id] = name
    return mapping


def evaluate_predictions(input_path: Path, entity_map_path: Path | None = None) -> RiTeKMetrics:
    entity_map = _load_entity_map(entity_map_path)
    exact_scores: list[Score] = []
    rouge_scores: list[Score] = []
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = PredictionRecord.model_validate_json(line)
            exact_scores.append(score_exact_match(record.effective_prediction(), record.golden))
            rouge_golden = [entity_map.get(answer, answer) for answer in record.golden]
            rouge_scores.append(score_rouge1(record.llm_pred, rouge_golden))
    return RiTeKMetrics(
        n=len(exact_scores),
        exact_match=_mean_score(exact_scores),
        rouge1=_mean_score(rouge_scores),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复算 RiTeK Exact Match 与 ROUGE-1 P/R/F1")
    parser.add_argument("--input", type=Path, required=True, help="pfit predictions.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="指标 JSON 输出路径")
    parser.add_argument("--entity_map", type=Path, help="可选的实体 ID 到名称映射")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_predictions(args.input, entity_map_path=args.entity_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(asdict(metrics), destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
