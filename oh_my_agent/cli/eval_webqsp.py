"""Batch evaluation entrypoint for the simple WebQSP QA agent."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from oh_my_agent.agent import SimpleWebQAgent, SimpleWebQAgentV2
from oh_my_agent.common import (
    aggregate_metrics,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    load_webqsp_qa_samples,
)
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool, PathRetrieveTool


DEFAULT_INPUT_PATH = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
DEFAULT_OUTPUT_PATH = "data/output/WebQSP/simple_agent_eval.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the simple WebQSP QA agent")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--lambda_val", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--path_server_url", default="http://localhost:8787")
    parser.add_argument("--path_retrieve_url", default="http://localhost:8789",
                        help="Cached path-retrieve server URL (used with --use_cached)")
    parser.add_argument("--use_cached", action="store_true",
                        help="Use cached path-retrieve server (SimpleWebQAgentV2) instead of live path server")
    parser.add_argument(
        "--path_method",
        choices=["tail_blend", "baseline"],
        default="tail_blend",
        help="Cached path retrieval method (used with --use_cached)",
    )
    parser.add_argument(
        "--alpha_final",
        type=float,
        default=1.0,
        help="Final entity score blend weight for cached tail_blend retrieval",
    )
    parser.add_argument(
        "--path_threshold",
        type=float,
        default=0.01,
        help="Score threshold for cached path retrieval reconstruction",
    )
    parser.add_argument("--llm_server_url", default="http://localhost:8788")
    parser.add_argument(
        "--entity_map",
        default="data/resources/WebQSP/fbwq_full/mapped_entities.txt",
        help="MID->name mapping file",
    )
    parser.add_argument("--no_adapter", action="store_true", help="Use the base model instead of the adapter")
    return parser


def _build_record(sample_index: int, sample, result, answer_metrics, faith_metrics) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question_raw": sample.question_raw,
        "question": sample.question,
        "topic_mid": sample.topic_mid,
        "gold_mids": sample.gold_mids,
        "raw_topics": result.raw_topics,
        "named_topics": result.named_topics,
        "raw_mmr_reason_paths": result.raw_mmr_reason_paths,
        "named_mmr_reason_paths": result.named_mmr_reason_paths,
        "raw_prediction": result.raw_prediction,
        "named_prediction": result.named_prediction,
        "llm_prompt": result.llm_prompt,
        "raw_llm_output": result.raw_llm_output,
        "pred_answer_names": result.pred_answer_names,
        "pred_answer_expanded_mids": result.pred_answer_expanded_mids,
        "pred_answer_disambiguated_mids": result.pred_answer_disambiguated_mids,
        "cited_path_indices": result.cited_path_indices,
        "golden_path_indices": sorted(label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids)),
        "hop": result.hop,
        "format_ok": result.format_ok,
        "used_adapter": result.used_adapter,
        "tokens_generated": result.tokens_generated,
        "retrieval_elapsed_ms": result.retrieval_elapsed_ms,
        "llm_elapsed_ms": result.llm_elapsed_ms,
        **answer_metrics,
        **faith_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_webqsp_qa_samples(args.input, limit=args.limit)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    answer_tool = AnswerWithPathsTool(
        base_url=args.llm_server_url,
        default_use_adapter=not args.no_adapter,
        default_max_new_tokens=args.max_new_tokens,
    )
    if args.use_cached:
        path_tool = PathRetrieveTool(base_url=args.path_retrieve_url, entity_map_path=args.entity_map)
        agent = SimpleWebQAgentV2(path_tool=path_tool, answer_tool=answer_tool)
    else:
        path_tool = PathRetrievalTool(base_url=args.path_server_url, entity_map_path=args.entity_map)
        agent = SimpleWebQAgent(path_tool=path_tool, answer_tool=answer_tool)

    total = len(samples)
    records: list[dict[str, Any]] = []
    t_start = time.monotonic()
    with open(args.output, "w", encoding="utf-8") as output_handle:
        for sample_index, sample in enumerate(samples):
            if args.use_cached:
                result = agent.run(
                    sample.question,
                    sample.topic_mid,
                    beam_size=args.beam_size,
                    lambda_val=args.lambda_val,
                    method=args.path_method,
                    alpha_final=args.alpha_final,
                    threshold=args.path_threshold,
                )
            else:
                result = agent.run(
                    sample.question,
                    sample.topic_mid,
                    hop=args.hop,
                    beam_size=args.beam_size,
                    lambda_val=args.lambda_val,
                )
            answer_metrics = compute_answer_metrics(
                result.pred_answer_disambiguated_mids,
                sample.gold_mids,
            )
            faith_metrics = compute_faithfulness(
                cited_indices=set(result.cited_path_indices),
                golden_indices=label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids),
                pred_answers=result.pred_answer_names,
                path_entities=get_all_path_entities(result.named_mmr_reason_paths),
            )
            record = _build_record(sample_index, sample, result, answer_metrics, faith_metrics)
            records.append(record)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            if (sample_index + 1) % 10 == 0 or sample_index == 0:
                n = len(records)
                hit1     = sum(r.get("hit1", 0)    for r in records) / n
                hit_any  = sum(r.get("hit_any", 0) for r in records) / n
                macro_f1 = sum(r.get("f1", 0)      for r in records) / n
                elapsed = time.monotonic() - t_start
                eta_s = elapsed / n * (total - n)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                print(f"[{sample_index + 1}/{total}] "
                      f"hit1={hit1:.4f} hit_any={hit_any:.4f} macro_f1={macro_f1:.4f} "
                      f"ret={result.retrieval_elapsed_ms:.0f}ms "
                      f"llm={result.llm_elapsed_ms:.0f}ms "
                      f"ETA={eta_str}", flush=True)

    summary = aggregate_metrics(records)
    summary["input_path"] = args.input
    summary["output_path"] = args.output
    if args.use_cached:
        summary["path_method"] = args.path_method
        summary["alpha_final"] = args.alpha_final
        summary["path_threshold"] = args.path_threshold
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
