"""Batch evaluation entrypoint for the checked-batch WebQSP QA agent."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.common import (
    build_eval_record,
    build_initial_answer_record,
    build_initial_retrieval_record,
    build_reverse_entity_map,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    llm_produced_answers,
    load_webqsp_qa_samples,
    mean_metric,
    record_answer_counts,
    summarize_checked_batch_records,
)
from oh_my_agent.llm_server.client import LLMClient, SILICONFLOW_MODEL, SiliconFlowLLMClient
from oh_my_agent.tools import (
    AnswerWithPathsTool,
    PathRetrieveTool,
    RejectedAnswerCheckTool,
)


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
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Run only one 0-based sample index from the input file",
    )
    parser.add_argument(
        "--sample_indices",
        default="",
        help="Run a comma/space-separated list of 0-based sample indices",
    )
    parser.add_argument("--path_method", choices=["tail_blend", "baseline"], default="tail_blend")
    parser.add_argument("--alpha_final", type=float, default=1.0)
    parser.add_argument("--path_threshold", type=float, default=0.01)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--lambda_val", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--dedupe_tail_paths",
        action="store_true",
        help="Deduplicate retrieved paths by final raw tail entity before batching",
    )
    parser.add_argument(
        "--score_margin",
        type=float,
        default=None,
        help="Drop final answers whose best supporting-path log_score trails the top "
        "answer by more than this margin (relative post-filter; None disables it)",
    )
    parser.add_argument(
        "--no_relation_expansion",
        action="store_true",
        help="Disable relation expansion (do not re-add cited-but-not-accepted paths "
        "that share a relation sequence with accepted paths)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--check_mode",
        choices=["reject-answer-list", "hybrid-reject-list"],
        default="reject-answer-list",
        help=(
            "How to validate cited answers: reject-answer-list removes bad candidates; "
            "hybrid-reject-list uses loose reject-list on the first batch and strict "
            "reject-list on later batches."
        ),
    )
    parser.add_argument(
        "--check_max_new_tokens",
        type=int,
        default=0,
        help="Max new tokens for validation. Default 0 uses 48.",
    )
    parser.add_argument(
        "--check_use_adapter",
        action="store_true",
        help="Run validation calls with the LoRA adapter loaded on the check "
        "LLM server (e.g. a dedicated checker adapter served via "
        "--check_llm_server_url)",
    )
    parser.add_argument(
        "--check_constrained_decoding",
        action="store_true",
        help="Constrain check output tokens to valid candidate indices or NONE "
        "(local llm_server backend only)",
    )
    parser.add_argument(
        "--hop_filter",
        action="store_true",
        help="Drop final answers supported only by relation chains whose length "
        "differs from the retrieval-predicted hop count",
    )
    parser.add_argument(
        "--large_answer_expansion",
        action="store_true",
        help="For enumeration-type questions (many answers, no selective "
        "constraint words), expand final answers to all KG tails of the winning "
        "relation group gated by the TransferNet prediction "
        "(requires --kg_group_tails_file)",
    )
    parser.add_argument(
        "--kg_group_tails_file",
        default="",
        help="JSON file mapping 'topic|rel1[|rel2]' to full KG tail mid lists",
    )
    parser.add_argument(
        "--expansion_min_answers",
        type=int,
        default=8,
        help="Minimum current answer count before large-answer expansion applies",
    )
    parser.add_argument(
        "--expansion_top_groups",
        type=int,
        default=1,
        help="Number of top answer-supporting relation groups to expand",
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="Do not stop batching on low-accept mixed batches; check all "
        "retrieved paths (downstream margin/hop/expansion filters guard precision)",
    )
    parser.add_argument("--path_retrieve_url", default="http://localhost:8789")
    parser.add_argument("--llm_server_url", default="http://localhost:8788")
    parser.add_argument(
        "--check_backend",
        choices=["server", "siliconflow"],
        default="server",
        help="LLM backend for validation only. Answer generation still uses --llm_server_url.",
    )
    parser.add_argument(
        "--check_llm_server_url",
        default="",
        help="Optional /generate server URL for validation when --check_backend server. "
        "Defaults to --llm_server_url.",
    )
    parser.add_argument(
        "--check_siliconflow_model",
        default=SILICONFLOW_MODEL,
        help="SiliconFlow model used only for validation when --check_backend siliconflow.",
    )
    parser.add_argument(
        "--entity_map",
        default="data/resources/WebQSP/fbwq_full/mapped_entities.txt",
        help="MID->name mapping file",
    )
    parser.add_argument("--no_adapter", action="store_true", help="Use the base model for answering")
    parser.add_argument("--skip_server_check", action="store_true", help="Skip service health checks")
    return parser


def load_file_kg_group_tails(
    large_answer_expansion: bool, kg_group_tails_file: str
) -> dict[str, list[str]] | None:
    """加载文件版 KG sidecar(可选)。在线 group_tails(path server 实时算 + prediction
    过滤)可用后,文件不再必需:仅当显式传入文件时加载,作为旧 server 的回退/覆盖源。
    agent.run 内会让在线 group_tails 优先于此文件。"""
    if not (large_answer_expansion and kg_group_tails_file):
        return None
    with open(kg_group_tails_file, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_output_paths(output: str) -> dict[str, str]:
    output_dir = os.path.splitext(output)[0] if output.endswith(".jsonl") else output
    return {
        "dir": output_dir,
        "records": os.path.join(output_dir, RESULT_FILENAME),
        "summary": os.path.join(output_dir, SUMMARY_FILENAME),
        "initial_retrieval": os.path.join(output_dir, INITIAL_RETRIEVAL_FILENAME),
        "initial_answer": os.path.join(output_dir, INITIAL_ANSWER_FILENAME),
    }


def _parse_sample_indices(raw_value: str) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for item in raw_value.replace(",", " ").split():
        index = int(item)
        if index < 0:
            raise ValueError(f"sample index must be non-negative: {index}")
        if index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def _requested_sample_indices(args: argparse.Namespace) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    if args.sample_index is not None:
        if args.sample_index < 0:
            raise ValueError(f"sample_index must be non-negative: {args.sample_index}")
        indices.append(args.sample_index)
        seen.add(args.sample_index)
    for index in _parse_sample_indices(args.sample_indices):
        if index not in seen:
            indices.append(index)
            seen.add(index)
    return indices


def _resolve_check_max_new_tokens(args: argparse.Namespace) -> int:
    if args.check_max_new_tokens > 0:
        return args.check_max_new_tokens
    return 48


def _build_check_client(args: argparse.Namespace):
    if args.check_backend == "siliconflow":
        return SiliconFlowLLMClient(model=args.check_siliconflow_model)
    return LLMClient(args.check_llm_server_url or args.llm_server_url)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_sample_indices = _requested_sample_indices(args)
    if not selected_sample_indices:
        samples = load_webqsp_qa_samples(args.input, limit=args.limit)
        sample_items = list(enumerate(samples))
    else:
        samples = load_webqsp_qa_samples(args.input, limit=max(selected_sample_indices) + 1)
        missing_indices = [
            index for index in selected_sample_indices if index >= len(samples)
        ]
        if missing_indices:
            raise ValueError(f"sample_index out of range: {missing_indices[0]}")
        sample_items = [(index, samples[index]) for index in selected_sample_indices]
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
    check_client = _build_check_client(args)
    check_max_new_tokens = _resolve_check_max_new_tokens(args)

    def build_check_tool(reject_policy: str) -> RejectedAnswerCheckTool:
        return RejectedAnswerCheckTool(
            client=check_client,
            default_use_adapter=args.check_use_adapter,
            default_max_new_tokens=check_max_new_tokens,
            reject_policy=reject_policy,
            constrained_decoding=args.check_constrained_decoding,
        )

    check_tool = build_check_tool("loose")
    check_tool_after_first = (
        build_check_tool("strict")
        if args.check_mode == "hybrid-reject-list"
        else None
    )
    if not args.skip_server_check:
        print("path_retrieve:", path_tool.client.health(), flush=True)
        print("llm         :", answer_tool.client.health(), flush=True)
        if args.check_backend != "server" or (
            args.check_llm_server_url and args.check_llm_server_url != args.llm_server_url
        ):
            print("check_llm   :", check_client.health(), flush=True)

    kg_group_tails = load_file_kg_group_tails(
        args.large_answer_expansion, args.kg_group_tails_file
    )

    agent = CheckedBatchWebQAgent(
        path_tool=path_tool,
        answer_tool=answer_tool,
        check_tool=check_tool,
        check_tool_after_first=check_tool_after_first,
    )
    reverse_entity_map = build_reverse_entity_map(path_tool.entity_map)

    total = len(sample_items)
    records: list[dict[str, Any]] = []
    t_start = time.monotonic()
    with open(output_paths["records"], "w", encoding="utf-8") as output_handle, open(
        output_paths["initial_retrieval"], "w", encoding="utf-8"
    ) as retrieval_handle, open(
        output_paths["initial_answer"], "w", encoding="utf-8"
    ) as answer_handle:
        for progress_index, (sample_index, sample) in enumerate(sample_items):
            result = agent.run(
                sample.question,
                sample.topic_mid,
                method=args.path_method,
                alpha_final=args.alpha_final,
                threshold=args.path_threshold,
                beam_size=args.beam_size,
                lambda_val=args.lambda_val,
                batch_size=args.batch_size,
                sample_index=sample_index if selected_sample_indices else None,
                dedupe_tail_paths=args.dedupe_tail_paths,
                score_margin=args.score_margin,
                enable_relation_expansion=not args.no_relation_expansion,
                hop_filter=args.hop_filter,
                large_answer_expansion=args.large_answer_expansion,
                kg_group_tails=kg_group_tails,
                expansion_min_answers=args.expansion_min_answers,
                expansion_top_groups=args.expansion_top_groups,
                no_early_stop=args.no_early_stop,
            )
            answer_metrics = compute_answer_metrics(
                result.pred_answer_disambiguated_mids,
                sample.gold_mids,
            )
            faith_metrics = compute_faithfulness(
                cited_indices=set(result.final_accepted_path_indices)
                | set(result.relation_expanded_path_indices),
                golden_indices=label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids),
                # 忠实度只算 LLM 产出的答案,剔除 large_answer_expansion 补出的 KG 实体
                pred_answers=llm_produced_answers(
                    result.pred_answer_names,
                    result.pred_answer_disambiguated_mids,
                    result.large_answer_expanded_mids,
                ),
                path_entities=get_all_path_entities(result.named_mmr_reason_paths),
            )
            record = build_eval_record(sample_index, sample, result, answer_metrics, faith_metrics)
            records.append(record)
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            retrieval_record = build_initial_retrieval_record(sample, result)
            retrieval_handle.write(json.dumps(retrieval_record, ensure_ascii=False) + "\n")
            retrieval_handle.flush()
            answer_record = build_initial_answer_record(
                sample_index,
                sample,
                result,
                args.batch_size,
                reverse_entity_map,
            )
            answer_handle.write(json.dumps(answer_record, ensure_ascii=False) + "\n")
            answer_handle.flush()

            if (progress_index + 1) % 10 == 0 or progress_index == 0:
                n = len(records)
                elapsed = time.monotonic() - t_start
                eta_s = elapsed / n * (total - n) if n else 0.0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                answer_counts = record_answer_counts(record)
                print(
                    f"[{progress_index + 1}/{total}] "
                    f"sample={sample_index} "
                    f"hit1={mean_metric(records, 'hit1'):.4f} "
                    f"hit_any={mean_metric(records, 'hit_any'):.4f} "
                    f"P={mean_metric(records, 'precision'):.4f} "
                    f"R={mean_metric(records, 'recall'):.4f} "
                    f"macro_f1={mean_metric(records, 'f1'):.4f} "
                    f"A/A_ok={answer_counts['model_answer_count']}/"
                    f"{answer_counts['model_correct_count']} "
                    f"B/B_ok={answer_counts['cited_answer_count']}/"
                    f"{answer_counts['cited_correct_count']} "
                    f"C={answer_counts['golden_answer_count']} "
                    f"batches={mean_metric(records, 'batches_used'):.2f} "
                    f"accepted={mean_metric(records, 'accepted_paths_count'):.2f} "
                    f"ETA={eta_str}",
                    flush=True,
                )

    summary = summarize_checked_batch_records(records)
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
            "dedupe_tail_paths": args.dedupe_tail_paths,
            "score_margin": args.score_margin,
            "relation_expansion": not args.no_relation_expansion,
            "check_mode": args.check_mode,
            "check_backend": args.check_backend,
            "check_llm_server_url": args.check_llm_server_url or args.llm_server_url,
            "check_siliconflow_model": (
                args.check_siliconflow_model
                if args.check_backend == "siliconflow"
                else None
            ),
            "check_max_new_tokens": check_max_new_tokens,
            "check_constrained_decoding": args.check_constrained_decoding,
            "hop_filter": args.hop_filter,
            "large_answer_expansion": args.large_answer_expansion,
            "kg_group_tails_file": args.kg_group_tails_file or None,
            "expansion_min_answers": args.expansion_min_answers,
            "expansion_top_groups": args.expansion_top_groups,
            "no_early_stop": args.no_early_stop,
            "sample_index": args.sample_index,
            "sample_indices": selected_sample_indices,
        }
    )
    summary_path = output_paths["summary"]
    with open(summary_path, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
