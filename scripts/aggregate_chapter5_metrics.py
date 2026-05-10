"""Aggregate Chapter-5 ablation metrics from initial_retrieval / initial_answer / final summary."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

EVAL_DIR = "data/output/WebQSP/checked_batch_agent/checked_batch_eval_20260505_0009"
OUTPUT = "data/analysis/chapter5_metrics.json"

RETRIEVAL_KEYS = [
    "mmr_top1_hit",
    "mmr_answer_path_hit",
    "mmr_answer_recall",
    "mmr_precision",
    "mmr_f1",
]

ANSWER_KEYS = [
    "hit1",
    "hit_any",
    "precision",
    "recall",
    "f1",
    "exact_match",
    "citation_accuracy",
    "citation_recall",
    "hallucination_rate",
    "format_ok",
]


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records if key in record]
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval(path: str) -> dict[str, float]:
    records = list(_iter_jsonl(path))
    return {key: round(_mean(records, key), 4) for key in RETRIEVAL_KEYS} | {"n": len(records)}


def aggregate_answer(path: str) -> dict[str, float]:
    records = list(_iter_jsonl(path))
    return {key: round(_mean(records, key), 4) for key in ANSWER_KEYS} | {"n": len(records)}


def main() -> int:
    retrieval = aggregate_retrieval(os.path.join(EVAL_DIR, "initial_retrieval.jsonl"))
    answer = aggregate_answer(os.path.join(EVAL_DIR, "initial_answer.jsonl"))
    with open(os.path.join(EVAL_DIR, "checked_batch_eval_summary.json"), "r", encoding="utf-8") as handle:
        final_summary = json.load(handle)

    output = {
        "source_dir": EVAL_DIR,
        "retrieval_only": retrieval,
        "first_batch_no_check": answer,
        "checked_batch_full": {
            key: final_summary.get(key)
            for key in [
                "n",
                "hit1",
                "hit_any",
                "macro_f1",
                "exact_match",
                "citation_accuracy",
                "hallucination_rate",
                "avg_batches_used",
                "avg_retrieval_elapsed_ms",
                "avg_answer_elapsed_ms",
                "avg_check_elapsed_ms",
                "stop_reason_counts",
            ]
        },
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
