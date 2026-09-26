"""对 PharmKG 检索结果执行类型感知重排与固定预算尾实体去重。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from kgqa.retrieve.datasets.pharmkg import (
    ReasonPath,
    infer_answer_type,
    load_entity_types,
    rerank_and_dedupe_paths,
)


class RetrievalRecord(TypedDict):
    sample_index: int
    question: str
    topics: list[str]
    hop: int
    golden: list[str]
    mmr_reason_paths: list[ReasonPath]
    prediction: dict[str, float]


class PostprocessSummary(TypedDict):
    n: int
    path_limit: int
    type_inferred: int
    type_coverage: float
    duplicate_tail_rows: int
    changed_top1_rows: int
    answer_hit_gained: int
    answer_hit_lost: int
    top1_rescued: int
    top1_harmed: int
    before_answer_hit: float
    after_answer_hit: float
    before_top1_hit: float
    after_top1_hit: float
    average_output_paths: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PharmKG 类型感知路径后处理")
    parser.add_argument("--input", required=True, help="至少包含 beam-50 候选路径的检索 JSONL")
    parser.add_argument("--entity_info", required=True, help="转换后的 entity_info.jsonl")
    parser.add_argument("--output", required=True, help="后处理 JSONL 输出路径")
    parser.add_argument("--summary", required=True, help="前后对照指标 JSON 输出路径")
    parser.add_argument("--path_limit", type=int, default=20, help="去重后的固定路径预算")
    return parser


def _load_records(path: Path) -> list[RetrievalRecord]:
    records: list[RetrievalRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _path_tail(path: ReasonPath) -> str:
    return path["path"][-1][-1]


def _has_answer(paths: list[ReasonPath], golden: list[str]) -> bool:
    gold = set(golden)
    return any(_path_tail(path) in gold for path in paths)


def _top1_hit(paths: list[ReasonPath], golden: list[str]) -> bool:
    return bool(paths) and _path_tail(paths[0]) in set(golden)


def process_records(
    records: list[RetrievalRecord],
    entity_types: dict[str, str],
    *,
    path_limit: int,
) -> tuple[list[RetrievalRecord], PostprocessSummary]:
    """后处理检索记录并返回可直接复核的前后对照指标。"""
    output: list[RetrievalRecord] = []
    type_inferred = duplicate_rows = changed_top1 = 0
    answer_gained = answer_lost = top1_rescued = top1_harmed = 0
    before_answer = after_answer = before_top1 = after_top1 = output_paths = 0
    for record in records:
        before = record["mmr_reason_paths"]
        after = rerank_and_dedupe_paths(
            record["question"], before, entity_types, limit=path_limit,
        )
        type_inferred += infer_answer_type(record["question"]) is not None
        duplicate_rows += len({_path_tail(path) for path in before}) < len(before)
        changed_top1 += bool(before and after and _path_tail(before[0]) != _path_tail(after[0]))
        before_answer_hit = _has_answer(before[:path_limit], record["golden"])
        after_answer_hit = _has_answer(after, record["golden"])
        before_top1_hit = _top1_hit(before, record["golden"])
        after_top1_hit = _top1_hit(after, record["golden"])
        answer_gained += after_answer_hit and not before_answer_hit
        answer_lost += before_answer_hit and not after_answer_hit
        top1_rescued += after_top1_hit and not before_top1_hit
        top1_harmed += before_top1_hit and not after_top1_hit
        before_answer += before_answer_hit
        after_answer += after_answer_hit
        before_top1 += before_top1_hit
        after_top1 += after_top1_hit
        output_paths += len(after)
        output.append({**record, "mmr_reason_paths": after})

    total = len(records)
    denominator = total or 1
    summary: PostprocessSummary = {
        "n": total,
        "path_limit": path_limit,
        "type_inferred": type_inferred,
        "type_coverage": type_inferred / denominator,
        "duplicate_tail_rows": duplicate_rows,
        "changed_top1_rows": changed_top1,
        "answer_hit_gained": answer_gained,
        "answer_hit_lost": answer_lost,
        "top1_rescued": top1_rescued,
        "top1_harmed": top1_harmed,
        "before_answer_hit": before_answer / denominator,
        "after_answer_hit": after_answer / denominator,
        "before_top1_hit": before_top1 / denominator,
        "after_top1_hit": after_top1 / denominator,
        "average_output_paths": output_paths / denominator,
    }
    return output, summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.path_limit <= 0:
        raise SystemExit("--path_limit 必须为正整数")
    records = _load_records(Path(args.input))
    entity_types = load_entity_types(args.entity_info)
    output, summary = process_records(records, entity_types, path_limit=args.path_limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
