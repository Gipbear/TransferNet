"""Evaluate the verify-only answer-check prompt on sampled or full KGQA records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.llm_server.client import LLMClient, OpenAICompatibleLLMClient
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


@dataclass(frozen=True)
class EvalJob:
    index: int
    rec: dict
    question: str
    pred_answers: list[str]
    paths_text: str
    required_max_new_tokens: int


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


def format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return f"[{'-' * width}] 0/0 (0.0%)"
    ratio = min(1.0, max(0.0, completed / total))
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total} ({ratio * 100:.1f}%)"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


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


def infer_required_max_new_tokens(default_max_new_tokens: int, paths_text: str) -> int:
    n_paths = len(re.findall(r"^\s*P\d+:", paths_text, re.MULTILINE))
    if n_paths > 1:
        return max(default_max_new_tokens, n_paths * 30 + 80)
    return default_max_new_tokens


def build_eval_jobs(sampled: list[dict]) -> list[EvalJob]:
    jobs = []
    for idx, rec in enumerate(sampled, start=1):
        paths_text = build_paths_text(rec)
        jobs.append(EvalJob(
            index=idx,
            rec=rec,
            question=rec.get("question", ""),
            pred_answers=rec.get("pred_answer_names", []),
            paths_text=paths_text,
            required_max_new_tokens=0,
        ))
    return jobs


def run_checker_with_retry(
    checker: AnswerCheckTool,
    question: str,
    pred_answers: list[str],
    paths_text: str,
    *,
    max_new_tokens: int | None = None,
) -> tuple[AnswerCheckToolResult, bool]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return checker(
                question,
                pred_answers,
                paths_text,
                max_new_tokens=max_new_tokens,
            ), False
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


def run_jobs_with_concurrency(
    jobs: list[EvalJob],
    concurrent_requests: int,
    worker,
) -> list:
    if concurrent_requests <= 1:
        return [worker(job, None, [job]) for job in jobs]

    results = []
    for start in range(0, len(jobs), concurrent_requests):
        batch = jobs[start:start + concurrent_requests]
        barrier = threading.Barrier(len(batch)) if len(batch) > 1 else None
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(worker, job, barrier, batch)
                for job in batch
            ]
            results.extend(future.result() for future in futures)
    return results


def run_eval_job(
    checker: AnswerCheckTool,
    job: EvalJob,
    barrier: threading.Barrier | None,
) -> tuple[EvalJob, AnswerCheckToolResult, bool]:
    if barrier is not None:
        barrier.wait()
    result, tool_error = run_checker_with_retry(
        checker,
        job.question,
        job.pred_answers,
        job.paths_text,
        max_new_tokens=job.required_max_new_tokens,
    )
    return job, result, tool_error


def run_eval(
    sample_size: int,
    seed: int,
    output: str | None,
    full: bool,
    *,
    prompt_variant: str,
    client_timeout: int,
    concurrent_requests: int,
    llm_backend: str,
    server_url: str,
    model: str,
    adapter_model: str | None,
    api_key: str,
    max_new_tokens: int | None = None,
) -> dict:
    records = load_records(JSONL_PATH)
    if full:
        sampled = records
    else:
        random.seed(seed)
        sampled = random.sample(records, sample_size)
    jobs = build_eval_jobs(sampled)

    if llm_backend == "legacy":
        client = LLMClient(server_url, timeout=client_timeout)
    elif llm_backend == "openai":
        client = OpenAICompatibleLLMClient(
            server_url,
            model=model,
            timeout=client_timeout,
            api_key=api_key,
            adapter_model=adapter_model,
        )
    else:
        raise ValueError(f"Unsupported llm_backend: {llm_backend}")

    checker = AnswerCheckTool(
        client=client,
        default_use_adapter=False,
        default_max_new_tokens=max_new_tokens,
        system_prompt=PROMPT_VARIANTS[prompt_variant],
    )
    jobs = [
        EvalJob(
            index=job.index,
            rec=job.rec,
            question=job.question,
            pred_answers=job.pred_answers,
            paths_text=job.paths_text,
            required_max_new_tokens=infer_required_max_new_tokens(
                checker.default_max_new_tokens,
                job.paths_text,
            ),
        )
        for job in jobs
    ]
    print("health:", client.health(), flush=True)
    print(
        f"run_mode={ANSWER_CHECK_MODE} sample_mode={'full' if full else 'sample'} "
        f"prompt_variant={prompt_variant} timeout={client_timeout} "
        f"seed={seed} sample_n={len(sampled)} total_records={len(records)} "
        f"max_new_tokens={checker.default_max_new_tokens} concurrent_requests={concurrent_requests} "
        f"llm_backend={llm_backend} server_url={server_url} model={model or '(legacy)'}",
        flush=True,
    )

    results = []
    completed_jobs = 0
    started_at = time.perf_counter()
    batch_size = max(1, concurrent_requests)
    total_batches = (len(jobs) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(jobs), batch_size), start=1):
        batch = jobs[start:start + batch_size]
        max_required_tokens = max(item.required_max_new_tokens for item in batch)
        print(
            f"[batch {batch_idx:04d}/{total_batches:04d}] dispatch size={len(batch)} "
            f"max_required_tokens={max_required_tokens} sample_range="
            f"{batch[0].index:04d}-{batch[-1].index:04d}",
            flush=True,
        )
        batch_results = run_jobs_with_concurrency(
            batch,
            batch_size,
            lambda job, barrier, inner_batch: run_eval_job(
                checker,
                job,
                barrier,
            ),
        )
        for job, result, tool_error in batch_results:
            hit1 = job.rec.get("hit1") == 1
            hit_any = job.rec.get("hit_any") == 1
            row = build_result_row(
                job.rec,
                job.question,
                job.pred_answers,
                result,
                hit1,
                hit_any,
                tool_error=tool_error,
            )
            results.append(row)
            marker = "✓" if result.verdict == "CORRECT" else "✗"
            print(
                f"[{job.index:04d}/{len(sampled)}] {marker} hit1={job.rec.get('hit1')} hit_any={job.rec.get('hit_any')} "
                f"f1={job.rec.get('f1')} match={result.match} tok={row['tokens_generated']} "
                f"ms={row['elapsed_ms']} tool_error={tool_error} :: {job.question[:80]}",
                flush=True,
            )
            completed_jobs += 1
        elapsed_s = time.perf_counter() - started_at
        avg_s = elapsed_s / completed_jobs if completed_jobs else 0.0
        remaining_jobs = max(0, len(sampled) - completed_jobs)
        eta_s = avg_s * remaining_jobs
        print(
            f"[progress] {format_progress_bar(completed_jobs, len(sampled))} "
            f"elapsed={format_duration(elapsed_s)} avg={avg_s:.1f}s/sample eta={format_duration(eta_s)}",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--concurrent_requests", type=int, default=1)
    parser.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="compact")
    parser.add_argument("--client-timeout", type=int, default=120)
    parser.add_argument("--llm-backend", choices=["legacy", "openai"], default="legacy")
    parser.add_argument("--server-url", type=str, default=SERVER_URL)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--adapter-model", type=str, default="")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_eval(
        sample_size=args.sample_size,
        seed=args.seed,
        output=args.output or None,
        full=args.full,
        max_new_tokens=args.max_new_tokens,
        prompt_variant=args.prompt_variant,
        client_timeout=args.client_timeout,
        concurrent_requests=max(1, args.concurrent_requests),
        llm_backend=args.llm_backend,
        server_url=args.server_url,
        model=args.model,
        adapter_model=args.adapter_model or None,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
