"""第五章适用条件探查：无 gold 代理量的跨数据集区分度与增益相关性。

在 WebQSP(1581) 与 RoG-CWQ(601) 的既有轨迹上计算三个不含 gold 的可观测代理量：
  1. tail_ratio      模型答案落在支撑路径终点的比例
  2. mid_ratio       模型答案落在支撑路径中间节点的比例
  3. group_over_ans  支撑关系组尾实体数 / 模型答案数（组比答案大多少倍）
并检验它们与逐样本 F1 差（开补全 − 关补全）的关系，用于判断路径 A 是否可行。

复现：
    python -m scripts.ch5_applicability_probe --output <dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from kgqa.agent.checked_batch import _path_entity_sequence, _relation_sequence_from_path
from kgqa.agent.common import tail_from_path_dict
from kgqa.core.answer_metrics import norm_entity

WEBQSP_TRACE = (
    "data/output/kgqa/ch5_pv_gac/webqsp/transfernet_v2/reports/followup_20260911"
    "/corrected_trace/checked_batch_eval.jsonl"
)
# 开/关补全两侧必须同口径：均取 F0（max_batches=2 + 原生早停）的回放产物。
# corrected_trace 本身是完整轨迹口径（max3 + 关早停，micro-F1 73.04），与 F0 的 73.63 不可混用。
WEBQSP_OPEN = (
    "data/output/kgqa/ch5_pv_gac/webqsp/transfernet_v2/reports/followup_20260911"
    "/corrected_cpu_matrix/f0/predictions.jsonl"
)
WEBQSP_CLOSED = (
    "data/output/kgqa/ch5_pv_gac/webqsp/transfernet_v2/reports/followup_20260911"
    "/corrected_cpu_matrix/no_large_expansion/predictions.jsonl"
)
CWQ_OPEN = (
    "data/output/kgqa/ch5_pv_gac/cwq/rog_transfernet_v1/benchmark/补全放宽"
    "/checked_batch_eval.jsonl"
)
CWQ_CLOSED = (
    "data/output/kgqa/ch5_pv_gac/cwq/rog_transfernet_v1/benchmark/无补全基线"
    "/checked_batch_eval.jsonl"
)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sample_f1(pred: set[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def support_indices(record: dict[str, Any]) -> list[int]:
    """生产逻辑使用的支撑路径：接受路径 + 关系恢复路径。"""
    indices = list(record.get("final_accepted_path_indices", []))
    indices += list(record.get("relation_expanded_path_indices", []))
    return list(dict.fromkeys(indices))


def load_prediction_records(path: str | Path) -> dict[int, dict[str, Any]]:
    """从回放产物取答案、支撑路径与补全实体——三者必须同口径，才能与 ΔF1 配对。

    不可用完整轨迹（corrected_trace，max3/关早停）的补全字段去配 F0 口径的答案：
    两者门控触发结果不同，WebQSP 上有 6 条样本因此被误判为"未触发"。
    """
    records: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        records[row["sample_index"]] = {
            "answers": row.get("pred_answer_disambiguated_mids", []),
            "support": support_indices(row),
            "expanded": row.get("large_answer_expanded_mids", []),
        }
    return records


def position_ratios(
    paths: list[dict], indices: list[int], answer_keys: set[str]
) -> tuple[float, float]:
    """返回答案落在路径终点、中间节点的比例。"""
    if not answer_keys:
        return 0.0, 0.0
    tails: set[str] = set()
    middles: set[str] = set()
    for index in indices:
        if not 0 < index <= len(paths):
            continue
        entities = [norm_entity(entity) for entity in _path_entity_sequence(paths[index - 1])]
        if not entities:
            continue
        tails.add(entities[-1])
        middles.update(entities[1:-1])
    total = len(answer_keys)
    return len(answer_keys & tails) / total, len(answer_keys & middles) / total


def group_stats(
    record: dict[str, Any], paths: list[dict], indices: list[int], answer_keys: set[str]
) -> tuple[float, int, int]:
    """返回（组内一致率, 最大组尾实体数, 组内尾实体并集大小）。"""
    topic = record.get("topic_mid", "")
    if isinstance(topic, list):
        topic = topic[0] if topic else ""
    group_tails = record.get("group_tails", {}) or {}
    union_keys: set[str] = set()
    largest = 0
    for index in indices:
        if not 0 < index <= len(paths):
            continue
        sequence = _relation_sequence_from_path(paths[index - 1])
        if not sequence:
            continue
        tails = group_tails.get("|".join((str(topic), *sequence)), [])
        if not tails:
            continue
        largest = max(largest, len(tails))
        union_keys.update(norm_entity(str(mid)) for mid in tails)
    ratio = len(answer_keys & union_keys) / len(answer_keys) if answer_keys else 0.0
    return ratio, largest, len(union_keys)


def winning_sequences(
    record: dict[str, Any], paths: list[dict], answer_keys: set[str]
) -> list[tuple[str, ...]]:
    """复现 _apply_large_answer_expansion 的组选择：tail 落在答案内的支撑路径序列，按频次排序。"""
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for index in support_indices(record):
        if not 0 < index <= len(paths):
            continue
        path_dict = paths[index - 1]
        sequence = _relation_sequence_from_path(path_dict)
        if not sequence:
            continue
        if norm_entity(tail_from_path_dict(path_dict)) in answer_keys:
            counts[sequence] += 1
    return [sequence for sequence, _ in sorted(counts.items(), key=lambda item: -item[1])]


def relation_features(sequences: list[tuple[str, ...]]) -> dict[str, float]:
    """关系组的关系名特征：是否含反向关系、序列长度、是否单跳属性型。"""
    if not sequences:
        return {"has_reverse": 0.0, "seq_len": 0.0, "any_reverse": 0.0}
    top = sequences[0]
    joined = " ".join(relation.lower() for sequence in sequences for relation in sequence)
    return {
        "has_reverse": float(any("reverse" in relation.lower() or "_rev" in relation.lower()
                                 for relation in top)),
        "seq_len": float(len(top)),
        "any_reverse": float("reverse" in joined or "_rev" in joined),
    }


def retrieval_signals(
    record: dict[str, Any], expanded: list[str]
) -> tuple[int, float, float, float]:
    """返回（检索候选数, 补进实体的平均分/最低分, 补进实体在候选中的占比）。"""
    prediction = record.get("raw_prediction", {}) or {}
    scores = [float(prediction[mid]) for mid in expanded if mid in prediction]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    coverage = len(scores) / len(expanded) if expanded else 0.0
    return len(prediction), mean_score, min_score, coverage


def build_rows(
    trace_path: str | Path,
    closed_records: dict[int, dict[str, Any]],
    open_records: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """trace 只提供样本级结构（路径、关系组、gold）；答案与补全取同口径回放产物。"""
    rows: list[dict[str, Any]] = []
    for record in iter_jsonl(trace_path):
        index = record["sample_index"]
        paths = record.get("raw_mmr_reason_paths", [])
        closed_source = closed_records.get(index, {})
        open_source = open_records.get(index, {})
        indices = list(open_source.get("support", []))
        closed = {norm_entity(mid) for mid in closed_source.get("answers", [])}
        opened = {norm_entity(mid) for mid in open_source.get("answers", [])}
        gold = {norm_entity(mid) for mid in record.get("gold_mids", [])}
        expanded = [mid for mid in open_source.get("expanded", [])]
        expanded_keys = {norm_entity(mid) for mid in expanded}
        tail_ratio, mid_ratio = position_ratios(paths, indices, closed)
        in_ratio, largest, union_size = group_stats(record, paths, indices, closed)
        pred_size, score_mean, score_min, coverage = retrieval_signals(record, expanded)
        sequences = winning_sequences(record, paths, closed)
        features = relation_features(sequences)
        rows.append({
            "sample_index": index,
            "gold_n": len(gold),
            "closed_n": len(closed),
            "tp_closed": len(closed & gold),
            "tp_open": len(opened & gold),
            "pred_open": len(opened),
            "f1_closed": sample_f1(closed, gold),
            "f1_open": sample_f1(opened, gold),
            "delta_f1": sample_f1(opened, gold) - sample_f1(closed, gold),
            "tail_ratio": tail_ratio,
            "mid_ratio": mid_ratio,
            "group_in_ratio": in_ratio,
            "group_largest": largest,
            "group_union": union_size,
            "group_over_ans": largest / len(closed) if closed else 0.0,
            "triggered": int(bool(expanded)),
            "expanded_n": len(expanded),
            "expanded_gold": len(expanded_keys & gold),
            "expanded_noise": len(expanded_keys - gold),
            "pred_size": pred_size,
            "expanded_score_mean": score_mean,
            "expanded_score_min": score_min,
            "expanded_in_pred": coverage,
            "recall_closed": len(closed & gold) / len(gold) if gold else 0.0,
            "top_relation": "|".join(sequences[0]) if sequences else "",
            **features,
        })
    return rows


def shared_relation_report(
    webqsp_rows: list[dict[str, Any]], cwq_rows: list[dict[str, Any]], min_count: int = 3
) -> str:
    """同一关系组在两数据集上的补全精度对比，用于判断关系名本身是否决定成败。"""
    def collect(rows: list[dict[str, Any]]) -> dict[str, list[dict]]:
        table: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["triggered"] and row["expanded_n"] > 0 and row["top_relation"]:
                table[row["top_relation"]].append(row)
        return table

    left, right = collect(webqsp_rows), collect(cwq_rows)
    lines = ["### 同一关系组在两数据集上的补全精度", "",
             "| 主导关系组 | WebQSP 触发 | WebQSP 精度 | CWQ 触发 | CWQ 精度 |", "| --- | ---: | ---: | ---: | ---: |"]
    for relation in sorted(set(left) & set(right)):
        if len(left[relation]) < min_count or len(right[relation]) < min_count:
            continue
        lines.append(
            f"| `{relation}` | {len(left[relation])} | {expansion_precision(left[relation]) * 100:.1f}% | "
            f"{len(right[relation])} | {expansion_precision(right[relation]) * 100:.1f}% |"
        )
    if len(lines) == 4:
        lines.append("| （两数据集无共同关系组达到样本下限） | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def describe(rows: list[dict[str, Any]]) -> dict[str, float]:
    triggered = [row for row in rows if row["triggered"]]
    return {
        "n": len(rows),
        "n_triggered": len(triggered),
        "gold_n": mean([row["gold_n"] for row in rows]),
        "delta_f1_all": mean([row["delta_f1"] for row in rows]),
        "delta_f1_triggered": mean([row["delta_f1"] for row in triggered]) if triggered else 0.0,
        "tail_ratio": mean([row["tail_ratio"] for row in rows]),
        "mid_ratio": mean([row["mid_ratio"] for row in rows]),
        "group_in_ratio": mean([row["group_in_ratio"] for row in rows]),
        "group_over_ans": mean([row["group_over_ans"] for row in rows]),
        "expanded_n": mean([row["expanded_n"] for row in triggered]) if triggered else 0.0,
        "expansion_precision": (
            mean([row["expanded_gold"] / row["expanded_n"] for row in triggered
                  if row["expanded_n"] > 0]) if triggered else 0.0
        ),
    }


def bucket_report(
    rows: list[dict[str, Any]], key: str, edges: list[float], label: str
) -> list[dict[str, Any]]:
    """按代理量分桶，报告每桶的触发率与平均 delta_f1（仅触发样本）。"""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        slot = sum(1 for edge in edges if value >= edge)
        buckets[slot].append(row)
    report = []
    for slot in sorted(buckets):
        members = buckets[slot]
        triggered = [row for row in members if row["triggered"]]
        low = edges[slot - 1] if slot > 0 else float("-inf")
        high = edges[slot] if slot < len(edges) else float("inf")
        report.append({
            "metric": label,
            "range": f"[{low:.2f}, {high:.2f})",
            "n": len(members),
            "n_triggered": len(triggered),
            "delta_f1_all": mean([row["delta_f1"] for row in members]),
            "delta_f1_triggered": mean([row["delta_f1"] for row in triggered]) if triggered else 0.0,
            "gold_n": mean([row["gold_n"] for row in members]),
            "expansion_precision": (
                mean([row["expanded_gold"] / row["expanded_n"] for row in triggered
                      if row["expanded_n"] > 0]) if triggered else 0.0
            ),
        })
    return report


def render(title: str, rows: list[dict[str, Any]], edges: dict[str, list[float]]) -> str:
    labels = {"tail_ratio": "答案落终点比例", "mid_ratio": "答案落中间比例",
              "group_in_ratio": "组内一致率", "group_over_ans": "组规模/答案数",
              "expanded_in_pred": "补进实体在检索候选中的占比"}
    lines = [f"## {title}", ""]
    for key, boundary in edges.items():
        lines.append(f"### 按 {labels.get(key, key)} 分桶")
        lines.append("| 区间 | N | 触发 | ΔF1(全体) | ΔF1(触发) | 触发样本补全精度 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for item in bucket_report(rows, key, boundary, labels[key]):
            lines.append(
                f"| {item['range']} | {item['n']} | {item['n_triggered']} | "
                f"{item['delta_f1_all'] * 100:+.2f} | {item['delta_f1_triggered'] * 100:+.2f} | "
                f"{item['expansion_precision'] * 100:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines)


EDGES = {
    "tail_ratio": [0.5, 0.99],
    "mid_ratio": [0.01, 0.5],
    "group_in_ratio": [0.5, 0.99],
    "group_over_ans": [1.0, 2.0, 5.0, 20.0],
    "expanded_in_pred": [0.5, 0.99],
}

CONTRAST_KEYS = (
    "gold_n", "closed_n", "recall_closed", "group_largest", "group_union", "group_over_ans",
    "group_in_ratio", "expanded_n", "pred_size", "expanded_score_mean", "expanded_score_min",
    "expanded_in_pred", "dataset", "has_reverse", "seq_len", "any_reverse",
)


def relation_feature_report(rows: list[dict[str, Any]], title: str) -> str:
    """按关系组的关系名特征分组，报告触发率、补全精度与逐样本增益。"""
    groups = (
        ("含反向关系（组内任一关系名含 reverse/_rev）", lambda row: row["any_reverse"] > 0.5),
        ("不含反向关系", lambda row: row["any_reverse"] <= 0.5),
        ("主导关系组为反向关系", lambda row: row["has_reverse"] > 0.5),
        ("主导关系组非反向", lambda row: row["has_reverse"] <= 0.5),
        ("关系序列长度=1（单跳）", lambda row: row["seq_len"] == 1),
        ("关系序列长度≥2（多跳）", lambda row: row["seq_len"] >= 2),
    )
    lines = [f"### {title}", "",
             "| 分组 | N | 触发 | ΔF1(全体) | ΔF1(触发) | 补全精度 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, predicate in groups:
        members = [row for row in rows if predicate(row)]
        triggered = [row for row in members if row["triggered"] and row["expanded_n"] > 0]
        lines.append(
            f"| {label} | {len(members)} | {len(triggered)} | "
            f"{mean([row['delta_f1'] for row in members]) * 100:+.2f} | "
            f"{(mean([row['delta_f1'] for row in triggered]) * 100) if triggered else 0.0:+.2f} | "
            f"{expansion_precision(triggered) * 100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def auc(positive: list[float], negative: list[float]) -> float:
    """Mann-Whitney U 归一化：0.5 无判别力，1.0 完全分离。"""
    if not positive or not negative:
        return 0.5
    pairs = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = end + 1
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def expansion_precision(rows: list[dict[str, Any]]) -> float:
    usable = [row for row in rows if row["expanded_n"] > 0]
    if not usable:
        return 0.0
    return mean([row["expanded_gold"] / row["expanded_n"] for row in usable])


def contrast_report(rows: list[dict[str, Any]], title: str) -> str:
    """触发样本按是否有正收益分组，用 AUC 排序各可观测量的判别力。"""
    triggered = [row for row in rows if row["triggered"] and row["expanded_n"] > 0]
    positive = [row for row in triggered if row["delta_f1"] > 1e-9]
    negative = [row for row in triggered if row["delta_f1"] <= 1e-9]
    lines = [f"### {title}（触发样本按正/无收益分组，N={len(positive)}/{len(negative)}）", "",
             "| 观测量 | 正收益均值 | 无收益均值 | 差值 | AUC |", "| --- | ---: | ---: | ---: | ---: |"]
    scored = []
    for key in CONTRAST_KEYS:
        values_pos = [float(row.get(key, 0.0)) for row in positive]
        values_neg = [float(row.get(key, 0.0)) for row in negative]
        score = auc(values_pos, values_neg)
        scored.append((abs(score - 0.5), key, mean(values_pos), mean(values_neg), score))
    for _, key, pos_mean, neg_mean, score in sorted(scored, reverse=True):
        lines.append(f"| {key} | {pos_mean:.3f} | {neg_mean:.3f} | {pos_mean - neg_mean:+.3f} | {score:.3f} |")
    lines.append(f"| 补全精度 | {expansion_precision(positive) * 100:.1f}% | "
                 f"{expansion_precision(negative) * 100:.1f}% | "
                 f"{(expansion_precision(positive) - expansion_precision(negative)) * 100:+.1f}pt | — |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    webqsp_rows = build_rows(WEBQSP_TRACE, load_prediction_records(WEBQSP_CLOSED),
                             load_prediction_records(WEBQSP_OPEN))
    cwq_rows = build_rows(CWQ_OPEN, load_prediction_records(CWQ_CLOSED),
                          load_prediction_records(CWQ_OPEN))

    payload = {
        "webqsp": {"summary": describe(webqsp_rows), "rows": webqsp_rows},
        "cwq": {"summary": describe(cwq_rows), "rows": cwq_rows},
    }
    (args.output / "probe_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 第五章适用条件探查结果", "", "## 总体", "",
             "| 数据集 | N | 触发补全 | ΔF1(全体) | ΔF1(触发) | 落终点 | 落中间 | 组内一致率 | 组/答案 | 补全精度 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, summary in (("WebQSP", payload["webqsp"]["summary"]), ("CWQ", payload["cwq"]["summary"])):
        lines.append(
            f"| {name} | {summary['n']} | {summary['n_triggered']} | "
            f"{summary['delta_f1_all'] * 100:+.2f} | {summary['delta_f1_triggered'] * 100:+.2f} | "
            f"{summary['tail_ratio'] * 100:.1f}% | {summary['mid_ratio'] * 100:.1f}% | "
            f"{summary['group_in_ratio'] * 100:.1f}% | {summary['group_over_ans']:.2f} | "
            f"{summary['expansion_precision'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append(render("WebQSP", webqsp_rows, EDGES))
    lines.append(render("CWQ", cwq_rows, EDGES))
    lines.append("## 关系组特征分组（事前可算）")
    lines.append("")
    lines.append(relation_feature_report(webqsp_rows, "WebQSP"))
    lines.append(relation_feature_report(cwq_rows, "CWQ"))
    lines.append(shared_relation_report(webqsp_rows, cwq_rows))
    lines.append("## 判别分析（触发样本，正收益 vs 无收益）")
    lines.append("")
    lines.append(contrast_report(webqsp_rows, "WebQSP"))
    lines.append(contrast_report(cwq_rows, "CWQ"))
    for row in webqsp_rows:
        row["dataset"] = 1.0
    for row in cwq_rows:
        row["dataset"] = 0.0
    lines.append(contrast_report(webqsp_rows + cwq_rows, "合并（dataset: WebQSP=1, CWQ=0）"))
    report = "\n".join(lines)
    (args.output / "probe_report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
