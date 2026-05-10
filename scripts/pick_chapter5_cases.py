"""Pick representative case-study samples for Chapter 5.

Case A (early-stop success): batches_used == 1, hit1 == 1, stop_reason != path_exhausted,
    verifier rejected at least 1 cited path, question length 30-90 chars.
    Picks idx=4 ("where was george washington carver from") by default.

Case B (multi-batch rescue): batches_used == 2, hit1 == 1, batch 0 has all_wrong status
    (all cited paths rejected), batch 1 provides the correct answer.
    Picks idx=574 ("who was president in 1988 in the united states") by default.
"""

from __future__ import annotations

import json
import os
from typing import Any

EVAL_PATH = "data/output/WebQSP/checked_batch_agent/checked_batch_eval_20260505_0009/checked_batch_eval.jsonl"
OUTPUT = "data/analysis/chapter5_cases.json"


def _iter_records():
    with open(EVAL_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _question_len(record: dict[str, Any]) -> int:
    return len(record.get("question_raw", ""))


def _batch_rejected(batch: dict[str, Any]) -> set[int]:
    cited = set(batch.get("local_cited_path_indices") or [])
    accepted = set(batch.get("accepted_path_indices") or [])
    return cited - accepted


def find_case_a() -> dict | None:
    """First batch succeeds (hit1=1) in one shot; verifier rejects >= 1 noisy citation."""
    for record in _iter_records():
        iters = record.get("iterations", [])
        if not iters:
            continue
        b0 = iters[0]
        rejected0 = _batch_rejected(b0)
        if (
            int(record.get("batches_used", 0)) == 1
            and float(record.get("hit1", 0.0)) >= 1.0
            and record.get("stop_reason") != "path_exhausted"
            and 30 <= _question_len(record) <= 90
            and len(rejected0) >= 1
        ):
            return record
    return None


def find_case_b() -> dict | None:
    """Batch 0 is all_wrong (fully rejected); batch 1 rescues with correct answer."""
    for record in _iter_records():
        iters = record.get("iterations", [])
        if len(iters) < 2:
            continue
        b0, b1 = iters[0], iters[1]
        accepted1 = set(b1.get("accepted_path_indices") or [])
        ans0 = set(b0.get("answer_names") or [])
        ans1 = set(b1.get("answer_names") or [])
        new_answers = ans1 - ans0
        if (
            int(record.get("batches_used", 0)) == 2
            and float(record.get("hit1", 0.0)) >= 1.0
            and 30 <= _question_len(record) <= 120
            and b0.get("batch_status") == "all_wrong"
            and len(accepted1) > 0
            and len(new_answers) > 0
        ):
            return record
    return None


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record.get("sample_index"),
        "question": record.get("question_raw"),
        "topic_mid": record.get("topic_mid"),
        "gold_mids": record.get("gold_mids"),
        "named_mmr_reason_paths_top": record.get("named_mmr_reason_paths", [])[:8],
        "iterations_compact": [
            {
                "batch_index": item.get("batch_index"),
                "batch_status": item.get("batch_status"),
                "answer_names": item.get("answer_names"),
                "local_cited_path_indices": item.get("local_cited_path_indices"),
                "accepted_path_indices": item.get("accepted_path_indices"),
                "rejected_path_indices": sorted(
                    set(item.get("local_cited_path_indices") or [])
                    - set(item.get("accepted_path_indices") or [])
                ),
            }
            for item in record.get("iterations", [])
        ],
        "final_accepted_path_indices": record.get("final_accepted_path_indices"),
        "pred_answer_names": record.get("pred_answer_names"),
        "hit1": record.get("hit1"),
        "f1": record.get("f1"),
        "stop_reason": record.get("stop_reason"),
        "batches_used": record.get("batches_used"),
    }


def main() -> int:
    case_a = find_case_a()
    case_b = find_case_b()
    if case_a is None or case_b is None:
        raise RuntimeError(
            f"failed to find cases: a={case_a is not None}, b={case_b is not None}"
        )
    output = {
        "case_a_early_stop": _summary(case_a),
        "case_b_multi_batch": _summary(case_b),
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
