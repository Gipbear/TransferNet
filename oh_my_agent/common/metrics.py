"""Evaluation helpers for the simple QA agent."""

from __future__ import annotations

from .output_parser import REJECTION_SENTINEL


def norm_entity(value: str) -> str:
    return value.lower().strip()


def label_golden_indices(mmr_paths: list[dict], golden: list[str]) -> set[int]:
    """Return 1-based path indices whose tail entity is a gold answer."""
    golden_set = {norm_entity(answer) for answer in golden}
    indices: set[int] = set()
    for index, path_dict in enumerate(mmr_paths, start=1):
        edges = path_dict.get("path", [])
        tail = edges[-1][2] if edges else None
        if tail and norm_entity(tail) in golden_set:
            indices.add(index)
    return indices


def cited_indices_for_answers(
    cited_indices: set[int],
    mmr_paths: list[dict],
    final_answer_mids: list[str],
) -> set[int]:
    """Keep only cited path indices whose tail entity survives in the final
    (post-calibration) answer set.

    citation_accuracy must be measured over the paths that actually support the
    calibrated final answers. Paths whose tail entity was pruned by the精确性
    过滤(score_margin / hop_filter / topic 守卫)are no longer cited as evidence
    for any answer and must leave the denominator; otherwise these typically
    non-gold旁支 paths artificially depress citation_accuracy."""
    final_set = {norm_entity(mid) for mid in final_answer_mids if mid.strip()}
    kept: set[int] = set()
    for index in cited_indices:
        if 1 <= index <= len(mmr_paths):
            edges = mmr_paths[index - 1].get("path", [])
            tail = edges[-1][2] if edges else None
            if tail and norm_entity(tail) in final_set:
                kept.add(index)
    return kept


def compute_answer_metrics(pred: list[str], gold: list[str]) -> dict[str, float | int | bool]:
    """Compute answer accuracy metrics aligned with eval_faithfulness.py."""
    pred_set = {norm_entity(entity) for entity in pred if entity.strip()}
    gold_set = {norm_entity(entity) for entity in gold if entity.strip()}

    hit1 = int(bool(pred) and norm_entity(pred[0]) in gold_set)
    hit_any = int(bool(pred_set & gold_set))

    if not pred_set and not gold_set:
        return {
            "hit1": 1,
            "hit_any": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "exact_match": True,
            "tp": 0,
            "pred_n": 0,
            "gold_n": 0,
        }
    if not pred_set or not gold_set:
        return {
            "hit1": hit1,
            "hit_any": hit_any,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": False,
            "tp": 0,
            "pred_n": len(pred_set),
            "gold_n": len(gold_set),
        }

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "hit1": hit1,
        "hit_any": hit_any,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": pred_set == gold_set,
        "tp": tp,
        "pred_n": len(pred_set),
        "gold_n": len(gold_set),
    }


def compute_faithfulness(
    cited_indices: set[int],
    golden_indices: set[int],
    pred_answers: list[str],
    path_entities: set[str],
) -> dict[str, float | list[str]]:
    """Compute citation and hallucination metrics."""
    if cited_indices:
        citation_accuracy = len(cited_indices & golden_indices) / len(cited_indices)
    else:
        citation_accuracy = 0.0

    if golden_indices:
        citation_recall = len(cited_indices & golden_indices) / len(golden_indices)
    else:
        citation_recall = 0.0

    effective_answers = [
        answer for answer in pred_answers if norm_entity(answer) != norm_entity(REJECTION_SENTINEL)
    ]
    if effective_answers:
        hallucinated_entities = [
            answer for answer in effective_answers if norm_entity(answer) not in path_entities
        ]
        hallucination_rate = len(hallucinated_entities) / len(effective_answers)
    else:
        hallucinated_entities = []
        hallucination_rate = 0.0

    return {
        "citation_accuracy": round(citation_accuracy, 4),
        "citation_recall": round(citation_recall, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "hallucinated_entities": hallucinated_entities,
    }


def llm_produced_answers(
    pred_names: list[str], pred_mids: list[str], expanded_mids: list[str]
) -> list[str]:
    """剔除 large_answer_expansion 等确定性后处理补出的实体,只留 LLM 实际产出的
    答案名。LLM 忠实度指标(hallucination)应只在这部分上算——expansion 补的是"路径
    外、KG 内"的事实,把它们算进幻觉会让模型指标被 pipeline 机制污染。"""
    expanded = {norm_entity(mid) for mid in expanded_mids}
    return [
        name
        for name, mid in zip(pred_names, pred_mids)
        if norm_entity(mid) not in expanded
    ]


def aggregate_metrics(results: list[dict]) -> dict[str, float | int]:
    """Aggregate per-sample evaluation records into a summary."""
    if not results:
        return {}

    sample_count = len(results)

    def mean(key: str) -> float:
        return sum(float(record[key]) for record in results) / sample_count

    tp_sum = sum(int(record["tp"]) for record in results)
    pred_sum = sum(int(record["pred_n"]) for record in results)
    gold_sum = sum(int(record["gold_n"]) for record in results)
    micro_p = tp_sum / pred_sum if pred_sum > 0 else 0.0
    micro_r = tp_sum / gold_sum if gold_sum > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "n": sample_count,
        "hit1": round(mean("hit1"), 4),
        "hit_any": round(sum(1 for record in results if record["hit_any"]) / sample_count, 4),
        "macro_p": round(mean("precision"), 4),
        "macro_r": round(mean("recall"), 4),
        "macro_f1": round(mean("f1"), 4),
        "micro_p": round(micro_p, 4),
        "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match": round(sum(1 for record in results if record["exact_match"]) / sample_count, 4),
        "citation_accuracy": round(mean("citation_accuracy"), 4),
        "citation_recall": round(mean("citation_recall"), 4),
        "hallucination_rate": round(mean("hallucination_rate"), 4),
        "format_compliance": round(sum(1 for record in results if record["format_ok"]) / sample_count, 4),
    }
