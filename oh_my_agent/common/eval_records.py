"""Record builders and summary helpers for checked-batch agent evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .entity_mapping import (
    expand_pred_answers_with_path_constraint,
    get_all_path_entities,
)
from .metrics import (
    aggregate_metrics,
    compute_answer_metrics,
    compute_faithfulness,
    label_golden_indices,
)
from .paths import tail_from_path_dict


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record.get(key, 0.0)) for record in records) / len(records)


def path_mid_entities(paths: list[dict[str, Any]]) -> set[str]:
    entities: set[str] = set()
    for path_dict in paths:
        for edge in path_dict.get("path", []):
            if len(edge) >= 3:
                entities.add(str(edge[0]))
                entities.add(str(edge[2]))
    return entities


def path_answer_metrics(
    paths: list[dict[str, Any]], gold_mids: list[str]
) -> dict[str, float | bool]:
    gold_set = {str(mid).lower().strip() for mid in gold_mids}
    tail_set = {
        tail.lower().strip()
        for tail in (tail_from_path_dict(path) for path in paths)
        if tail
    }
    hit_count = len(tail_set & gold_set)
    precision = hit_count / len(tail_set) if tail_set else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    top1_tail = tail_from_path_dict(paths[0]).lower().strip() if paths else ""
    return {
        "mmr_answer_path_hit": bool(hit_count),
        "mmr_top1_hit": bool(top1_tail and top1_tail in gold_set),
        "mmr_answer_recall": round(recall, 4),
        "mmr_precision": round(precision, 4),
        "mmr_f1": round(f1, 4),
        "hit": bool(hit_count),
    }


def path_diversity(paths: list[dict[str, Any]]) -> dict[str, float]:
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
        for right_edges in edge_sets[left_index + 1:]:
            union = left_edges | right_edges
            similarity = len(left_edges & right_edges) / len(union) if union else 0.0
            pair_distances.append(1.0 - similarity)

    tails = [tail_from_path_dict(path) for path in paths if tail_from_path_dict(path)]
    all_edges = set().union(*edge_sets) if edge_sets else set()
    total_edges = sum(len(edge_set) for edge_set in edge_sets)
    return {
        "jaccard_diversity": round(sum(pair_distances) / len(pair_distances), 4),
        "tail_diversity": round(len(set(tails)) / len(paths), 4) if tails else 0.0,
        "edge_coverage": round(len(all_edges) / total_edges, 4) if total_edges else 0.0,
    }


def build_eval_record(
    sample_index: int,
    sample: Any,
    result: Any,
    answer_metrics: dict[str, Any],
    faith_metrics: dict[str, Any],
) -> dict[str, Any]:
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


def build_initial_retrieval_record(sample: Any, result: Any) -> dict[str, Any]:
    raw_paths = result.raw_mmr_reason_paths
    return {
        "question": sample.question_raw,
        "topics": result.raw_topics,
        "hop": result.hop,
        "mmr_reason_paths": raw_paths,
        "path_diversity": path_diversity(raw_paths),
        "golden": sample.gold_mids,
        "prediction": result.raw_prediction,
        **path_answer_metrics(raw_paths, sample.gold_mids),
    }


def build_initial_answer_record(
    sample_index: int,
    sample: Any,
    result: Any,
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
        path_mid_entities=path_mid_entities(raw_paths),
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
        "path_diversity": path_diversity(raw_paths),
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
            tail = tail_from_path_dict(raw_paths[path_offset])
            if tail:
                cited_answers.add(_norm_value(tail))
    return cited_answers


def _path_tail_mid_by_name(record: dict[str, Any]) -> dict[str, set[str]]:
    name_to_mids: dict[str, set[str]] = {}
    raw_paths = record.get("raw_mmr_reason_paths", [])
    named_paths = record.get("named_mmr_reason_paths", [])
    for raw_path, named_path in zip(raw_paths, named_paths):
        name_tail = tail_from_path_dict(named_path)
        raw_tail = tail_from_path_dict(raw_path)
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


def record_answer_counts(record: dict[str, Any]) -> dict[str, int]:
    gold = {
        _norm_value(mid)
        for mid in record.get("gold_mids", [])
        if str(mid).strip()
    }
    model_answers = _record_model_answers(record)
    cited_answers = _record_cited_answers(record)
    return {
        "model_answer_count": len(model_answers),
        "model_correct_count": sum(1 for mids in model_answers.values() if mids & gold),
        "cited_answer_count": len(cited_answers),
        "cited_correct_count": len(cited_answers & gold),
        "golden_answer_count": len(gold),
    }


def summarize_checked_batch_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(aggregate_metrics(records))
    if not records:
        return summary

    stop_counts = Counter(str(record.get("stop_reason", "")) for record in records)
    summary.update(
        {
            "avg_batches_used": round(mean_metric(records, "batches_used"), 4),
            "stop_reason_counts": dict(sorted(stop_counts.items())),
            "avg_checked_paths": round(mean_metric(records, "checked_paths_count"), 4),
            "avg_accepted_paths": round(mean_metric(records, "accepted_paths_count"), 4),
            "avg_final_answer_count": round(mean_metric(records, "final_answer_count"), 4),
            "avg_retrieval_elapsed_ms": round(mean_metric(records, "retrieval_elapsed_ms"), 2),
            "avg_answer_elapsed_ms": round(mean_metric(records, "llm_elapsed_ms"), 2),
            "avg_check_elapsed_ms": round(mean_metric(records, "check_elapsed_ms"), 2),
        }
    )
    return summary
