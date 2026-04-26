"""Evaluate the verify-only answer-check prompt on sampled or full KGQA records."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import LLMClient
from oh_my_agent.tools import AnswerCheckTool
from oh_my_agent.tools.answer_check import (
    AnswerCheckToolResult,
    LEGACY_VERIFY_ANSWER_CHECK_SYSTEM,
    VERIFY_ANSWER_CHECK_SYSTEM,
    build_paths_text,
    is_placeholder_answer,
)

JSONL_PATH = "data/output/WebQSP/simple_agent_eval_debug.jsonl"
SERVER_URL = "http://localhost:8788"
ANSWER_CHECK_MODE = "verify"
MAX_RETRIES = 2
RETRY_DELAY_S = 1.0
PROMPT_VARIANTS = {
    "legacy": LEGACY_VERIFY_ANSWER_CHECK_SYSTEM,
    "compact": VERIFY_ANSWER_CHECK_SYSTEM,
}

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

    if result.get("tool_error"):
        tags.append("tool_error")
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


def build_result_row(
    rec: dict,
    question: str,
    pred_answers: list[str],
    result: AnswerCheckToolResult,
    hit1: bool,
    hit_any: bool,
    *,
    tool_error: bool = False,
) -> dict:
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
        "tokens_generated": None if tool_error else result.tokens_generated,
        "elapsed_ms": None if tool_error else result.elapsed_ms,
        "tool_error": tool_error,
    }
    row["tags"] = classify_record(row)
    row["disputable"] = disputable(row["tags"])
    return row


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * q) - 1))
    return float(ordered[idx])


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "count": 0,
        }
    return {
        "avg": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "count": len(values),
    }


def build_summary(
    *,
    seed: int,
    full: bool,
    prompt_variant: str,
    client_timeout: int,
    results: list[dict],
) -> dict:
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
    token_values = [float(r["tokens_generated"]) for r in results if r.get("tokens_generated") is not None]
    elapsed_values = [float(r["elapsed_ms"]) for r in results if r.get("elapsed_ms") is not None]

    return {
        "seed": seed,
        "mode": ANSWER_CHECK_MODE,
        "prompt_variant": prompt_variant,
        "client_timeout": client_timeout,
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
        "tool_error_n": sum(r.get("tool_error", False) for r in results),
        "tokens_generated_stats": summarize_metric(token_values),
        "elapsed_ms_stats": summarize_metric(elapsed_values),
        "tag_counts": dict(tag_counter.most_common()),
    }


def build_error_result(question: str, pred_answers: list[str], error_text: str) -> AnswerCheckToolResult:
    return AnswerCheckToolResult(
        mode=ANSWER_CHECK_MODE,
        question=question,
        pred_answers=pred_answers,
        prompt="",
        raw_output=f"ERROR: {error_text}",
        verdict="PARSE_ERROR",
        tokens_generated=0,
        elapsed_ms=0.0,
    )


def run_checker_with_retry(
    checker: AnswerCheckTool,
    question: str,
    pred_answers: list[str],
    paths_text: str,
) -> tuple[AnswerCheckToolResult, bool]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return checker(question, pred_answers, paths_text), False
        except Exception as exc:  # pragma: no cover - exercised via integration runs
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            print(
                f"retry={attempt + 1}/{MAX_RETRIES} sample_question={question[:60]!r} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(RETRY_DELAY_S)
    assert last_error is not None
    return build_error_result(question, pred_answers, f"{type(last_error).__name__}: {last_error}"), True


def run_eval(
    sample_size: int,
    seed: int,
    output: str | None,
    full: bool,
    *,
    prompt_variant: str,
    client_timeout: int,
    max_new_tokens: int | None = None,
) -> dict:
    records = load_records(JSONL_PATH)
    if full:
        sampled = records
    else:
        random.seed(seed)
        sampled = random.sample(records, sample_size)

    client = LLMClient(SERVER_URL, timeout=client_timeout)
    checker = AnswerCheckTool(
        client=client,
        default_use_adapter=False,
        default_max_new_tokens=max_new_tokens,
        system_prompt=PROMPT_VARIANTS[prompt_variant],
    )
    print("health:", client.health(), flush=True)
    print(
        f"run_mode={ANSWER_CHECK_MODE} sample_mode={'full' if full else 'sample'} "
        f"prompt_variant={prompt_variant} timeout={client_timeout} "
        f"seed={seed} sample_n={len(sampled)} total_records={len(records)} "
        f"max_new_tokens={checker.default_max_new_tokens}",
        flush=True,
    )

    results = []
    for idx, rec in enumerate(sampled, start=1):
        question = rec.get("question", "")
        pred_answers = rec.get("pred_answer_names", [])
        paths_text = build_paths_text(rec)
        result, tool_error = run_checker_with_retry(checker, question, pred_answers, paths_text)
        hit1 = rec.get("hit1") == 1
        hit_any = rec.get("hit_any") == 1
        row = build_result_row(
            rec,
            question,
            pred_answers,
            result,
            hit1,
            hit_any,
            tool_error=tool_error,
        )
        results.append(row)
        marker = "✓" if result.verdict == "CORRECT" else "✗"
        print(
            f"[{idx:04d}/{len(sampled)}] {marker} hit1={rec.get('hit1')} hit_any={rec.get('hit_any')} "
            f"f1={rec.get('f1')} match={result.match} tok={row['tokens_generated']} "
            f"ms={row['elapsed_ms']} tool_error={tool_error} :: {question[:80]}",
            flush=True,
        )

    mismatch_examples = [r for r in results if not r["agree_with_hit1"]][:20]
    summary = build_summary(
        seed=seed,
        full=full,
        prompt_variant=prompt_variant,
        client_timeout=client_timeout,
        results=results,
    )

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
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="compact")
    parser.add_argument("--client-timeout", type=int, default=120)
    args = parser.parse_args()
    run_eval(
        sample_size=args.sample_size,
        seed=args.seed,
        output=args.output or None,
        full=args.full,
        max_new_tokens=args.max_new_tokens,
        prompt_variant=args.prompt_variant,
        client_timeout=args.client_timeout,
    )


if __name__ == "__main__":
    main()
