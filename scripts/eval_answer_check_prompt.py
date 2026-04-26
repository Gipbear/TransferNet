"""Evaluate the verify-only answer-check prompt on sampled or full KGQA records."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from oh_my_agent.llm_server.client import LLMClient
from oh_my_agent.tools import AnswerCheckTool
from oh_my_agent.tools.answer_check import build_paths_text, is_placeholder_answer

JSONL_PATH = "data/output/WebQSP/simple_agent_eval_debug.jsonl"
SERVER_URL = "http://localhost:8788"
ANSWER_CHECK_MODE = "verify"

QUALIFIER_HINTS = (
    "first",
    "last",
    "current",
    "currently",
    "today",
    "real",
    "full name",
    "original",
    "originally",
    "before",
    "after",
    "when did",
)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_qualifier(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in QUALIFIER_HINTS) or bool(re.search(r"\b(19|20)\d{2}\b", q))


def classify_record(result: dict) -> list[str]:
    tags: list[str] = []
    question = result["question"]
    pred_answers = result.get("pred_answers", [])

    if result["verdict"] == "PARSE_ERROR":
        tags.append("parse_error")
    if not result.get("gold_mids") and not result.get("gold_names"):
        tags.append("gold_missing")
    if 0 < float(result.get("f1") or 0) < 1:
        tags.append("partial_overlap")
    if any(is_placeholder_answer(a) for a in pred_answers):
        tags.append("placeholder_answer")
    if has_qualifier(question):
        tags.append("qualified_question")
    if result["verdict"] == "CORRECT" and result.get("hit_any") == 0:
        tags.append("verdict_hitany_mismatch")
    if result["verdict"] == "INCORRECT" and result.get("hit_any") == 1:
        tags.append("verdict_fn_against_hitany")
    if (
        result["verdict"] == "CORRECT"
        and result.get("hit_any") == 0
        and "placeholder_answer" not in tags
    ):
        tags.append("likely_label_issue_or_alias")
    if (
        "what year" in question.lower()
        and pred_answers
        and all(re.match(r"^\d{4}\b", answer.strip()) for answer in pred_answers)
    ):
        tags.append("wrapped_year_answer")
    return tags


def disputable(tags: list[str]) -> bool:
    dispute_tags = {
        "gold_missing",
        "likely_label_issue_or_alias",
        "partial_overlap",
        "qualified_question",
        "wrapped_year_answer",
    }
    return any(tag in dispute_tags for tag in tags)


def build_result_row(rec: dict, question: str, pred_answers: list[str], result, hit1: bool, hit_any: bool) -> dict:
    row = {
        "sample_index": rec.get("sample_index"),
        "question": question,
        "pred_answers": pred_answers,
        "gold_names": rec.get("gold_names", []),
        "gold_mids": rec.get("gold_mids", []),
        "hit1": rec.get("hit1"),
        "hit_any": rec.get("hit_any"),
        "f1": rec.get("f1"),
        "verdict": result.verdict,
        "llm_raw_output": result.raw_output,
        "agree_with_hit1": (result.verdict == "CORRECT") == hit1,
        "agree_with_hit_any": (result.verdict == "CORRECT") == hit_any,
        "path_verdicts": result.path_verdicts,
        "path_reasons": result.path_reasons,
        "any_valid_path": result.any_valid_path,
        "match": result.match,
        "match_detail": result.match_detail,
    }
    row["tags"] = classify_record(row)
    row["disputable"] = disputable(row["tags"])
    return row


def build_summary(*, seed: int, full: bool, results: list[dict]) -> dict:
    total = len(results)
    llm_correct_n = sum(r["verdict"] == "CORRECT" for r in results)
    hit1_n = sum(r["hit1"] == 1 for r in results)
    hit_any_n = sum(r["hit_any"] == 1 for r in results)
    agree_hit1_n = sum(r["agree_with_hit1"] for r in results)
    agree_hitany_n = sum(r["agree_with_hit_any"] for r in results)

    undisputed = [r for r in results if not r["disputable"]]
    undisputed_total = len(undisputed)
    undisputed_hit1_agree = sum(r["agree_with_hit1"] for r in undisputed)
    undisputed_hitany_agree = sum(r["agree_with_hit_any"] for r in undisputed)
    tag_counter = Counter(tag for r in results for tag in r["tags"])

    return {
        "seed": seed,
        "mode": ANSWER_CHECK_MODE,
        "sample_n": total,
        "full": full,
        "llm_correct": llm_correct_n,
        "llm_correct_rate": round(llm_correct_n / total, 4) if total else 0,
        "hit1_correct": hit1_n,
        "hit1_rate": round(hit1_n / total, 4) if total else 0,
        "hit_any_correct": hit_any_n,
        "hit_any_rate": round(hit_any_n / total, 4) if total else 0,
        "llm_hit1_agree": agree_hit1_n,
        "llm_hit1_agree_rate": round(agree_hit1_n / total, 4) if total else 0,
        "llm_hit_any_agree": agree_hitany_n,
        "llm_hit_any_agree_rate": round(agree_hitany_n / total, 4) if total else 0,
        "undisputed_n": undisputed_total,
        "undisputed_hit1_agree": undisputed_hit1_agree,
        "undisputed_hit1_agree_rate": round(undisputed_hit1_agree / undisputed_total, 4) if undisputed_total else 0,
        "undisputed_hit_any_agree": undisputed_hitany_agree,
        "undisputed_hit_any_agree_rate": round(undisputed_hitany_agree / undisputed_total, 4) if undisputed_total else 0,
        "tag_counts": dict(tag_counter.most_common()),
    }


def run_eval(sample_size: int, seed: int, output: str | None, full: bool) -> dict:
    records = load_records(JSONL_PATH)
    if full:
        sampled = records
    else:
        random.seed(seed)
        sampled = random.sample(records, sample_size)

    client = LLMClient(SERVER_URL)
    checker = AnswerCheckTool(
        client=client,
        default_use_adapter=False,
    )
    print("health:", client.health(), flush=True)
    print(
        f"run_mode={ANSWER_CHECK_MODE} sample_mode={'full' if full else 'sample'} "
        f"seed={seed} sample_n={len(sampled)} total_records={len(records)}",
        flush=True,
    )

    results = []
    for idx, rec in enumerate(sampled, start=1):
        question = rec.get("question", "")
        pred_answers = rec.get("pred_answer_names", [])
        paths_text = build_paths_text(rec)
        result = checker(question, pred_answers, paths_text)
        hit1 = rec.get("hit1") == 1
        hit_any = rec.get("hit_any") == 1
        row = build_result_row(rec, question, pred_answers, result, hit1, hit_any)
        results.append(row)
        marker = "✓" if result.verdict == "CORRECT" else "✗"
        print(
            f"[{idx:04d}/{len(sampled)}] {marker} hit1={rec.get('hit1')} hit_any={rec.get('hit_any')} "
            f"f1={rec.get('f1')} match={result.match} :: {question[:80]}",
            flush=True,
        )

    mismatch_examples = [r for r in results if not r["agree_with_hit1"]][:20]
    summary = build_summary(seed=seed, full=full, results=results)

    payload = {
        "summary": summary,
        "mismatch_examples": mismatch_examples,
        "results": results,
    }

    print("===SUMMARY===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {output_path}", flush=True)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    run_eval(
        sample_size=args.sample_size,
        seed=args.seed,
        output=args.output or None,
        full=args.full,
    )


if __name__ == "__main__":
    main()
