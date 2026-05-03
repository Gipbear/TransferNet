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
    build_reverse_entity_map,
    compute_answer_metrics,
    compute_faithfulness,
    expand_pred_answers_with_path_constraint,
    get_all_path_entities,
    label_golden_indices,
    load_webqsp_qa_samples,
)
from oh_my_agent.tools import AnswerWithPathsTool, CitedPathCheckTool, PathRetrieveTool


DEFAULT_INPUT_PATH = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
DEFAULT_OUTPUT_DIR_PREFIX = "data/output/WebQSP/checked_batch_agent/checked_batch_eval"
RESULT_FILENAME = "checked_batch_eval.jsonl"
SUMMARY_FILENAME = "checked_batch_eval_summary.json"
INITIAL_RETRIEVAL_FILENAME = "initial_retrieval.jsonl"
INITIAL_ANSWER_FILENAME = "initial_answer.jsonl"


def _default_output_dir() -> str:
    return f"{DEFAULT_OUTPUT_DIR_PREFIX}_{time.strftime('%Y%m%d_%H%M')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the checked-batch WebQSP QA agent")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=_default_output_dir(), help="Output directory")
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


def _resolve_output_paths(output: str) -> dict[str, str]:
    output_dir = os.path.splitext(output)[0] if output.endswith(".jsonl") else output
    return {
        "dir": output_dir,
        "records": os.path.join(output_dir, RESULT_FILENAME),
        "summary": os.path.join(output_dir, SUMMARY_FILENAME),
        "initial_retrieval": os.path.join(output_dir, INITIAL_RETRIEVAL_FILENAME),
        "initial_answer": os.path.join(output_dir, INITIAL_ANSWER_FILENAME),
    }


def _path_tail(path_dict: dict[str, Any]) -> str:
    edges = path_dict.get("path", [])
    return str(edges[-1][2]) if edges else ""


def _path_mid_entities(paths: list[dict[str, Any]]) -> set[str]:
    entities: set[str] = set()
    for path_dict in paths:
        for edge in path_dict.get("path", []):
            if len(edge) >= 3:
                entities.add(str(edge[0]))
                entities.add(str(edge[2]))
    return entities


def _path_metrics(paths: list[dict[str, Any]], gold_mids: list[str]) -> dict[str, float | bool]:
    gold_set = {str(mid).lower().strip() for mid in gold_mids}
    tail_set = {tail.lower().strip() for tail in (_path_tail(path) for path in paths) if tail}
    hit_count = len(tail_set & gold_set)
    precision = hit_count / len(tail_set) if tail_set else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    top1_tail = _path_tail(paths[0]).lower().strip() if paths else ""
    return {
        "mmr_answer_path_hit": bool(hit_count),
        "mmr_top1_hit": bool(top1_tail and top1_tail in gold_set),
        "mmr_answer_recall": round(recall, 4),
        "mmr_precision": round(precision, 4),
        "mmr_f1": round(f1, 4),
        "hit": bool(hit_count),
    }


def _path_diversity(paths: list[dict[str, Any]]) -> dict[str, float]:
    if len(paths) < 2:
        return {
            "jaccard_diversity": 0.0,
            "tail_diversity": 0.0,
            "edge_coverage": 0.0,
        }

    edge_sets = [
        {tuple(edge[:3]) for edge in path_dict.get("path", []) if len(edge) >= 3}
        for path_dict in paths
    ]
    pair_distances: list[float] = []
    for left_index, left_edges in enumerate(edge_sets):
        for right_edges in edge_sets[left_index + 1 :]:
            union = left_edges | right_edges
            similarity = len(left_edges & right_edges) / len(union) if union else 0.0
            pair_distances.append(1.0 - similarity)

    tails = [_path_tail(path) for path in paths if _path_tail(path)]
    all_edges = set().union(*edge_sets) if edge_sets else set()
    total_edges = sum(len(edge_set) for edge_set in edge_sets)
    return {
        "jaccard_diversity": round(sum(pair_distances) / len(pair_distances), 4),
        "tail_diversity": round(len(set(tails)) / len(paths), 4) if tails else 0.0,
        "edge_coverage": round(len(all_edges) / total_edges, 4) if total_edges else 0.0,
    }


def _build_record(sample_index: int, sample, result, answer_metrics, faith_metrics) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "question_raw": sample.question_raw,
        "question": sample.question,
        "topic_mid": sample.topic_mid,
        "gold_mids": sample.gold_mids,
        **answer_metrics,
        **faith_metrics,
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
    }


def _build_initial_retrieval_record(sample, result) -> dict[str, Any]:
    raw_paths = result.raw_mmr_reason_paths
    return {
        "question": sample.question_raw,
        "topics": result.raw_topics,
        "hop": result.hop,
        "mmr_reason_paths": raw_paths,
        "path_diversity": _path_diversity(raw_paths),
        "golden": sample.gold_mids,
        "prediction": result.raw_prediction,
        **_path_metrics(raw_paths, sample.gold_mids),
    }


def _build_initial_answer_record(
    sample_index: int,
    sample,
    result,
    batch_size: int,
    reverse_entity_map: dict[str, set[str]],
) -> dict[str, Any]:
    first_iteration = result.iterations[0] if result.iterations else None
    raw_paths = result.raw_mmr_reason_paths[:batch_size]
    named_paths = result.named_mmr_reason_paths[:batch_size]
    answer_names = first_iteration.answer_names if first_iteration else []
    cited_indices = first_iteration.local_cited_path_indices if first_iteration else []
    expanded_mids, disambiguated_mids = expand_pred_answers_with_path_constraint(
        pred_answers=answer_names,
        rev_entity_map=reverse_entity_map,
        path_mid_entities=_path_mid_entities(raw_paths),
    )
    answer_metrics = compute_answer_metrics(disambiguated_mids, sample.gold_mids)
    golden_indices = label_golden_indices(raw_paths, sample.gold_mids)
    faith_metrics = compute_faithfulness(
        cited_indices=set(cited_indices),
        golden_indices=golden_indices,
        pred_answers=answer_names,
        path_entities=get_all_path_entities(named_paths),
    )
    return {
        "sample_index": sample_index,
        "question": sample.question_raw,
        "topics": result.raw_topics,
        "hop": result.hop,
        "mmr_reason_paths": raw_paths,
        "named_mmr_reason_paths": named_paths,
        "path_diversity": _path_diversity(raw_paths),
        "golden": sample.gold_mids,
        "prediction": result.raw_prediction,
        "hit": bool(golden_indices),
        "llm_raw_input": first_iteration.answer_prompt if first_iteration else "",
        "llm_raw_output": first_iteration.raw_llm_output if first_iteration else "",
        "llm_pred": answer_names,
        "is_rejection": not answer_names,
        "llm_pred_expanded_mids": expanded_mids,
        "llm_pred_disambiguated_mids": disambiguated_mids,
        "cited_indices": cited_indices,
        "golden_path_indices": sorted(golden_indices),
        "format_ok": first_iteration.format_ok if first_iteration else False,
        "used_adapter": first_iteration.used_adapter if first_iteration else False,
        "tokens_generated": (
            first_iteration.answer_tokens_generated if first_iteration else 0
        ),
        "llm_elapsed_ms": first_iteration.answer_elapsed_ms if first_iteration else 0.0,
        **answer_metrics,
        **faith_metrics,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def _norm_value(value: Any) -> str:
    return str(value).lower().strip()


def _record_cited_answers(record: dict[str, Any]) -> set[str]:
    raw_paths = record.get("raw_mmr_reason_paths", [])
    cited_answers: set[str] = set()
    for index in record.get("cited_path_indices", []):
        if not isinstance(index, int):
            continue
        path_offset = index - 1
        if 0 <= path_offset < len(raw_paths):
            tail = _path_tail(raw_paths[path_offset])
            if tail:
                cited_answers.add(_norm_value(tail))
    return cited_answers


def _path_tail_mid_by_name(record: dict[str, Any]) -> dict[str, set[str]]:
    name_to_mids: dict[str, set[str]] = {}
    raw_paths = record.get("raw_mmr_reason_paths", [])
    named_paths = record.get("named_mmr_reason_paths", [])
    for raw_path, named_path in zip(raw_paths, named_paths):
        name_tail = _path_tail(named_path)
        raw_tail = _path_tail(raw_path)
        if name_tail and raw_tail:
            name_to_mids.setdefault(_norm_value(name_tail), set()).add(_norm_value(raw_tail))
    return name_to_mids


def _record_model_answers(record: dict[str, Any]) -> dict[str, set[str]]:
    name_to_mids = _path_tail_mid_by_name(record)
    answers: dict[str, set[str]] = {}
    for iteration in record.get("iterations", []):
        for answer in iteration.get("answer_names", []):
            answer_key = _norm_value(answer)
            if not answer_key:
                continue
            answers.setdefault(answer_key, set()).update(
                name_to_mids.get(answer_key, {answer_key})
            )
    return answers


def _answer_count_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "model_answer_count": 0,
        "model_correct_count": 0,
        "cited_answer_count": 0,
        "cited_correct_count": 0,
        "golden_answer_count": 0,
    }
    for record in records:
        gold = {
            _norm_value(mid)
            for mid in record.get("gold_mids", [])
            if str(mid).strip()
        }
        model_answers = _record_model_answers(record)
        cited_answers = _record_cited_answers(record)
        totals["model_answer_count"] += len(model_answers)
        totals["model_correct_count"] += sum(
            1 for mids in model_answers.values() if mids & gold
        )
        totals["cited_answer_count"] += len(cited_answers)
        totals["cited_correct_count"] += len(cited_answers & gold)
        totals["golden_answer_count"] += len(gold)
    return totals


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
        f"- output_dir: `{summary.get('output_dir', args.output)}`",
        f"- result_jsonl: `{summary.get('output_path', '')}`",
        f"- initial_retrieval_jsonl: `{summary.get('initial_retrieval_path', '')}`",
        f"- initial_answer_jsonl: `{summary.get('initial_answer_path', '')}`",
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
    output_paths = _resolve_output_paths(args.output)
    os.makedirs(output_paths["dir"], exist_ok=True)

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
    reverse_entity_map = build_reverse_entity_map(path_tool.entity_map)

    total = len(samples)
    records: list[dict[str, Any]] = []
    t_start = time.monotonic()
    with open(output_paths["records"], "w", encoding="utf-8") as output_handle, open(
        output_paths["initial_retrieval"], "w", encoding="utf-8"
    ) as retrieval_handle, open(
        output_paths["initial_answer"], "w", encoding="utf-8"
    ) as answer_handle:
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
            retrieval_record = _build_initial_retrieval_record(sample, result)
            retrieval_handle.write(json.dumps(retrieval_record, ensure_ascii=False) + "\n")
            retrieval_handle.flush()
            answer_record = _build_initial_answer_record(
                sample_index,
                sample,
                result,
                args.batch_size,
                reverse_entity_map,
            )
            answer_handle.write(json.dumps(answer_record, ensure_ascii=False) + "\n")
            answer_handle.flush()

            if (sample_index + 1) % 10 == 0 or sample_index == 0:
                n = len(records)
                elapsed = time.monotonic() - t_start
                eta_s = elapsed / n * (total - n) if n else 0.0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                answer_counts = _answer_count_totals(records)
                print(
                    f"[{sample_index + 1}/{total}] "
                    f"hit1={_mean(records, 'hit1'):.4f} "
                    f"hit_any={_mean(records, 'hit_any'):.4f} "
                    f"P={_mean(records, 'precision'):.4f} "
                    f"R={_mean(records, 'recall'):.4f} "
                    f"macro_f1={_mean(records, 'f1'):.4f} "
                    f"A/A_ok={answer_counts['model_answer_count']}/"
                    f"{answer_counts['model_correct_count']} "
                    f"B/B_ok={answer_counts['cited_answer_count']}/"
                    f"{answer_counts['cited_correct_count']} "
                    f"C={answer_counts['golden_answer_count']} "
                    f"batches={_mean(records, 'batches_used'):.2f} "
                    f"accepted={_mean(records, 'accepted_paths_count'):.2f} "
                    f"ETA={eta_str}",
                    flush=True,
                )

    summary = _summarize(records)
    summary.update(
        {
            "input_path": args.input,
            "output_dir": output_paths["dir"],
            "output_path": output_paths["records"],
            "initial_retrieval_path": output_paths["initial_retrieval"],
            "initial_answer_path": output_paths["initial_answer"],
            "path_method": args.path_method,
            "alpha_final": args.alpha_final,
            "path_threshold": args.path_threshold,
            "beam_size": args.beam_size,
            "lambda_val": args.lambda_val,
            "batch_size": args.batch_size,
        }
    )
    summary_path = output_paths["summary"]
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
