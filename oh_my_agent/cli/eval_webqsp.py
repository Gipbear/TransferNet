"""Batch evaluation entrypoint for the simple WebQSP QA agent."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from oh_my_agent.agent import SimpleWebQAgent
from oh_my_agent.common import (
    aggregate_metrics,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    load_webqsp_qa_samples,
)
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool


DEFAULT_INPUT_PATH = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt"
DEFAULT_OUTPUT_PATH = "data/output/WebQSP/simple_agent_eval.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the simple WebQSP QA agent")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--lambda_val", type=float, default=0.2)
    parser.add_argument("--prediction_threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--path_server_url", default="http://localhost:8787")
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

    path_tool = PathRetrievalTool(base_url=args.path_server_url, entity_map_path=args.entity_map)
    answer_tool = AnswerWithPathsTool(
        base_url=args.llm_server_url,
        default_use_adapter=not args.no_adapter,
        default_max_new_tokens=args.max_new_tokens,
    )
    agent = SimpleWebQAgent(path_tool=path_tool, answer_tool=answer_tool)

    records: list[dict[str, Any]] = []
    with open(args.output, "w", encoding="utf-8") as output_handle:
        for sample_index, sample in enumerate(samples):
            result = agent.run(
                sample.question,
                sample.topic_mid,
                hop=args.hop,
                beam_size=args.beam_size,
                lambda_val=args.lambda_val,
                prediction_threshold=args.prediction_threshold,
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

    summary = aggregate_metrics(records)
    summary["input_path"] = args.input
    summary["output_path"] = args.output
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
