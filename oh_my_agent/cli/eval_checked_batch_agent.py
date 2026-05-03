"""Batch evaluation entrypoint for the checked-batch WebQSP QA agent."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Any

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.common import (
    aggregate_metrics,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    load_webqsp_qa_samples,
)
from oh_my_agent.tools import AnswerWithPathsTool, CitedPathCheckTool, PathRetrieveTool


DEFAULT_INPUT_PATH = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
DEFAULT_OUTPUT_PATH = "data/output/WebQSP/checked_batch_agent/checked_batch_eval.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the checked-batch WebQSP QA agent")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--path_method", choices=["tail_blend", "baseline"], default="tail_blend")
    parser.add_argument("--alpha_final", type=float, default=1.0)
    parser.add_argument("--path_threshold", type=float, default=0.01)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--check_max_new_tokens", type=int, default=2)
    parser.add_argument("--path_retrieve_url", default="http://localhost:8789")
    parser.add_argument("--llm_server_url", default="http://localhost:8788")
    parser.add_argument(
        "--entity_map",
        default="data/resources/WebQSP/fbwq_full/mapped_entities.txt",
        help="MID->name mapping file",
    )
    parser.add_argument("--no_adapter", action="store_true", help="Use the base model for answering")
    parser.add_argument("--check_use_adapter", action="store_true", help="Use the adapter for path checks")
    parser.add_argument("--skip_server_check", action="store_true", help="Skip service health checks")
    parser.add_argument("--no_archive", action="store_true", help="Do not write data/analysis README")
    parser.add_argument(
        "--analysis_dir",
        default="",
        help="Optional explicit analysis archive directory",
    )
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
        "iterations": [item.to_dict() for item in result.iterations],
        "final_accepted_path_indices": result.final_accepted_path_indices,
        "cited_path_indices": result.cited_path_indices,
        "relation_expanded_path_indices": result.relation_expanded_path_indices,
        "golden_path_indices": sorted(label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids)),
        "pred_answer_names": result.pred_answer_names,
        "pred_answer_expanded_mids": result.pred_answer_expanded_mids,
        "pred_answer_disambiguated_mids": result.pred_answer_disambiguated_mids,
        "hop": result.hop,
        "batches_used": result.batches_used,
        "checked_paths_count": result.checked_paths_count,
        "accepted_paths_count": result.accepted_paths_count,
        "final_answer_count": result.final_answer_count,
        "stop_reason": result.stop_reason,
        "format_ok": result.format_ok,
        "used_adapter": result.used_adapter,
        "tokens_generated": result.tokens_generated,
        "answer_tokens_generated": result.answer_tokens_generated,
        "check_tokens_generated": result.check_tokens_generated,
        "retrieval_elapsed_ms": result.retrieval_elapsed_ms,
        "llm_elapsed_ms": result.llm_elapsed_ms,
        "check_elapsed_ms": result.check_elapsed_ms,
        **answer_metrics,
        **faith_metrics,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(aggregate_metrics(records))
    if not records:
        return summary

    stop_counts = Counter(str(record.get("stop_reason", "")) for record in records)
    summary.update(
        {
            "avg_batches_used": round(_mean(records, "batches_used"), 4),
            "stop_reason_counts": dict(sorted(stop_counts.items())),
            "avg_checked_paths": round(_mean(records, "checked_paths_count"), 4),
            "avg_accepted_paths": round(_mean(records, "accepted_paths_count"), 4),
            "avg_final_answer_count": round(_mean(records, "final_answer_count"), 4),
            "avg_retrieval_elapsed_ms": round(_mean(records, "retrieval_elapsed_ms"), 2),
            "avg_answer_elapsed_ms": round(_mean(records, "llm_elapsed_ms"), 2),
            "avg_check_elapsed_ms": round(_mean(records, "check_elapsed_ms"), 2),
        }
    )
    return summary


def _write_archive(args: argparse.Namespace, summary: dict[str, Any], summary_path: str) -> str:
    analysis_dir = args.analysis_dir
    if not analysis_dir:
        stamp = time.strftime("%Y%m%d_%H%M")
        analysis_dir = f"data/analysis/{stamp}__checked_batch_agent_eval"
    os.makedirs(analysis_dir, exist_ok=True)
    readme_path = os.path.join(analysis_dir, "README.md")
    lines = [
        "# checked_batch_agent_eval",
        "",
        "## Command",
        "",
        "```bash",
        "python -m oh_my_agent.cli.eval_checked_batch_agent "
        f"--input {args.input} --output {args.output}",
        "```",
        "",
        "## Config",
        "",
        f"- path_method: `{args.path_method}`",
        f"- alpha_final: `{args.alpha_final}`",
        f"- beam_size: `{args.beam_size}`",
        f"- lambda_val: `{args.lambda_val}`",
        f"- batch_size: `{args.batch_size}`",
        f"- output: `{args.output}`",
        f"- summary: `{summary_path}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    with open(readme_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return readme_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_webqsp_qa_samples(args.input, limit=args.limit)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    path_tool = PathRetrieveTool(
        base_url=args.path_retrieve_url,
        entity_map_path=args.entity_map,
    )
    answer_tool = AnswerWithPathsTool(
        base_url=args.llm_server_url,
        default_use_adapter=not args.no_adapter,
        default_max_new_tokens=args.max_new_tokens,
    )
    check_tool = CitedPathCheckTool(
        base_url=args.llm_server_url,
        default_use_adapter=args.check_use_adapter,
        default_max_new_tokens=args.check_max_new_tokens,
    )
    if not args.skip_server_check:
        print("path_retrieve:", path_tool.client.health(), flush=True)
        print("llm         :", answer_tool.client.health(), flush=True)

    agent = CheckedBatchWebQAgent(
        path_tool=path_tool,
        answer_tool=answer_tool,
        check_tool=check_tool,
    )

    total = len(samples)
    records: list[dict[str, Any]] = []
    t_start = time.monotonic()
    with open(args.output, "w", encoding="utf-8") as output_handle:
        for sample_index, sample in enumerate(samples):
            result = agent.run(
                sample.question,
                sample.topic_mid,
                method=args.path_method,
                alpha_final=args.alpha_final,
                threshold=args.path_threshold,
                beam_size=args.beam_size,
                lambda_val=args.lambda_val,
                batch_size=args.batch_size,
            )
            answer_metrics = compute_answer_metrics(
                result.pred_answer_disambiguated_mids,
                sample.gold_mids,
            )
            faith_metrics = compute_faithfulness(
                cited_indices=set(result.final_accepted_path_indices)
                | set(result.relation_expanded_path_indices),
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
                elapsed = time.monotonic() - t_start
                eta_s = elapsed / n * (total - n) if n else 0.0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                print(
                    f"[{sample_index + 1}/{total}] "
                    f"hit1={_mean(records, 'hit1'):.4f} "
                    f"hit_any={_mean(records, 'hit_any'):.4f} "
                    f"macro_f1={_mean(records, 'f1'):.4f} "
                    f"batches={_mean(records, 'batches_used'):.2f} "
                    f"accepted={_mean(records, 'accepted_paths_count'):.2f} "
                    f"ETA={eta_str}",
                    flush=True,
                )

    summary = _summarize(records)
    summary.update(
        {
            "input_path": args.input,
            "output_path": args.output,
            "path_method": args.path_method,
            "alpha_final": args.alpha_final,
            "path_threshold": args.path_threshold,
            "beam_size": args.beam_size,
            "lambda_val": args.lambda_val,
            "batch_size": args.batch_size,
        }
    )
    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    if not args.no_archive:
        summary["analysis_readme"] = _write_archive(args, summary, summary_path)
        with open(summary_path, "w", encoding="utf-8") as summary_handle:
            json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
