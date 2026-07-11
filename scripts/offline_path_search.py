"""离线路径搜索实验脚本

从 WebQSP/predict.py 生成的得分缓存（.pt 文件）加载中间得分矩阵，
在 CPU 上快速重放统一路径搜索并汇总实验指标。

典型用法：

  # 运行离线路径检索
  python scripts/offline_path_search.py \\
      --cache output/score_cache/webqsp_val.pt \\
      --input_dir data/WebQSP \\
      --threshold 0.01 --beam_size 20 --lambda_val 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.kg.global_kg import GlobalKG
from kgqa.retrieve.engine import (
    LogNormStrategy,
    PathCandidate,
    candidate_hop_numbers,
    candidate_to_tuple,
    compute_candidate_score,
    path_to_triples,
    reconstruct_ent_dict,
    reconstruct_rel_dict,
    search_path_candidates,
    select_path_candidates,
)
from kgqa.scores.base import load_score_cache
from utils.path_utils import build_valid_edges_dict, compute_path_diversity, compute_path_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 缓存加载与稀疏重建
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_valid_edges_dict(input_dir: str) -> dict[int, list[tuple[int, int]]]:
    """兼容旧实验入口；实际 KG 加载由统一实现负责。"""
    return GlobalKG.from_input_dir(input_dir).valid_edges_dict


def sample_valid_edges_dict(
    sample: dict,
    global_valid_edges_dict: Optional[dict[int, list[tuple[int, int]]]],
) -> dict[int, list[tuple[int, int]]]:
    """返回当前样本可用的邻接表。

    WebQSP 使用全局 fbwq_full KG；CWQ 的 predict.py 按样本 subgraph 搜索，
    因此 CWQ dump cache 会在每个 sample 中保存 ``triples`` 并在这里重建。
    """
    if "triples" in sample:
        return build_valid_edges_dict(sample["triples"])
    if global_valid_edges_dict is None:
        raise ValueError("缓存未包含 sample triples，且未提供全局 valid_edges_dict")
    return global_valid_edges_dict


# ─────────────────────────────────────────────────────────────────────────────
# 实验主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def final_ent_score_dict(sample: dict) -> dict[int, float]:
    return {
        int(idx): float(val)
        for idx, val in zip(sample["e_score_indices"].tolist(), sample["e_score_values"].tolist())
    }


def run_experiment(
    cache: dict,
    valid_edges_dict: Optional[dict[int, list[tuple[int, int]]]],
    threshold: float,
    beam_size: int,
    output_path: Optional[str] = None,
    alpha_final: float = 2.0,
    lambda_val: float = 0.2,
) -> dict:
    """对缓存中所有样本运行离线路径搜索，返回聚合指标。

    若提供 output_path，则同时将每样本结果写入 JSONL 文件，
    格式与 data/output/WebQSP/grid_search/paths/beam*.jsonl 保持一致。
    """
    samples = cache["samples"]
    meta = cache["meta"]
    id2ent: dict = meta.get("id2ent", {})
    id2rel: dict = meta.get("id2rel", {})

    total = len(samples)
    agg = {
        "answer_hit": 0, "top1_hit": 0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "diversity_edge": 0.0,
        "relation_jaccard_diversity": 0.0,
        "tail_unique": 0.0,
        "relation_coverage": 0.0,
        "edge_coverage": 0.0,
        "empty_path": 0,
    }
    # 检查阈值截断风险：若最后一个保存的得分仍高于 threshold，可能丢失候选
    topk = meta.get("topk_entities", 500)
    truncation_warnings = 0

    out_file = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_file = open(output_path, "w", encoding="utf-8")

    try:
        scoring = LogNormStrategy()
        for sample in tqdm(samples, desc="search", unit="sample", dynamic_ncols=True):
            current_valid_edges_dict = sample_valid_edges_dict(sample, valid_edges_dict)
            hop_num = int(sample["hop_attn"].argmax().item()) + 1
            hop_nums = candidate_hop_numbers(len(sample["rel_probs"]))
            topic_ids = sample["topic_ids"]
            gold_ids = set(sample["gold_ids"])

            # 重建每跳的稀疏字典
            rel_dicts, ent_dicts = [], []
            for t in range(max(hop_nums)):
                rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], threshold))
                ed = reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], threshold)
                ent_dicts.append(ed)
                # 截断风险检测：若 top-K 末尾得分仍 >= threshold，可能有更多候选被丢弃
                scores_t = sample["ent_scores"][t]
                if len(scores_t) == topk and float(scores_t[-1]) >= threshold:
                    truncation_warnings += 1

            path_candidates: list[PathCandidate] = []
            final_scores = final_ent_score_dict(sample)
            for candidate_hop in hop_nums:
                path_candidates.extend(search_path_candidates(
                    topic_ids, rel_dicts, ent_dicts, candidate_hop,
                    current_valid_edges_dict, scoring, beam_size,
                    final_ent_scores=final_scores,
                    order_start=len(path_candidates),
                ))
            selected_candidates = select_path_candidates(
                path_candidates,
                beam_size,
                alpha_final=alpha_final,
                lambda_val=lambda_val,
            )
            candidates = [candidate_to_tuple(c) for c in selected_candidates]

            if not candidates:
                agg["empty_path"] += 1
                if out_file:
                    record = _build_empty_record(sample, hop_num, id2ent)
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            selected = candidates

            # 评估
            m = compute_path_metrics(selected, gold_ids, id2rel=id2rel)
            d = compute_path_diversity(selected)

            agg["answer_hit"] += int(m["answer_hit"])
            agg["top1_hit"] += int(m["top1_hit"])
            agg["precision"] += m["precision"]
            agg["recall"] += m["recall"]
            agg["f1"] += m["f1"]
            agg["diversity_edge"] += d.get("jaccard_diversity", 0.0)
            agg["relation_jaccard_diversity"] += d.get("relation_jaccard_diversity", 0.0)
            agg["tail_unique"] += d.get("tail_diversity", 0.0)
            agg["relation_coverage"] += d.get("relation_coverage", 0.0)
            agg["edge_coverage"] += d.get("edge_coverage", 0.0)

            # 写 JSONL
            if out_file:
                record = _build_record(
                    sample, hop_num, selected, m, d, id2ent, id2rel,
                )
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out_file:
            out_file.close()

    if truncation_warnings > 0:
        print(f"[WARN] {truncation_warnings} 次跳步检测到 top-K 截断风险（末位得分 >= threshold={threshold}）。"
              f" 建议降低 threshold 或在生成缓存时增大 --topk_entities。", flush=True)

    if output_path:
        print(f"[INFO] 路径结果已写入: {output_path}", flush=True)

    n = max(total - agg["empty_path"], 1)
    return {
        "total": total,
        "empty_path": agg["empty_path"],
        "answer_hit_rate": agg["answer_hit"] / total,
        "top1_hit_rate": agg["top1_hit"] / total,
        "precision": agg["precision"] / n,
        "recall": agg["recall"] / n,
        "f1": agg["f1"] / n,
        "diversity_edge": agg["diversity_edge"] / n,
        "relation_jaccard_diversity": agg["relation_jaccard_diversity"] / n,
        "tail_unique": agg["tail_unique"] / n,
        "relation_coverage": agg["relation_coverage"] / n,
        "edge_coverage": agg["edge_coverage"] / n,
    }


def _build_record(
    sample: dict,
    hop_num: int,
    selected: list[tuple[list[int], list[int], float]],
    metrics: dict,
    diversity: dict,
    id2ent: dict,
    id2rel: dict,
) -> dict:
    """将单样本搜索结果序列化为目标 JSONL 格式。"""
    # mmr_reason_paths: 路径三元组列表 + log_score
    mmr_reason_paths = []
    for nodes, rels, score in selected:
        triples = path_to_triples(nodes, rels, id2ent, id2rel)
        mmr_reason_paths.append({
            "path": triples,
            "log_score": round(score, 6),
        })

    # golden: MID 字符串列表
    golden = [id2ent.get(g, str(g)) for g in sample["gold_ids"]]

    # prediction: e_score > 0.5 的实体（与 predict.py 输出保持一致）
    prediction = {}
    for idx, val in zip(sample["e_score_indices"].tolist(),
                        sample["e_score_values"].tolist()):
        if val >= 0.5:
            mid = id2ent.get(idx, str(idx))
            prediction[mid] = round(val, 4)

    # topics: MID 字符串列表
    topics = [id2ent.get(t, str(t)) for t in sample["topic_ids"]]

    return {
        "question": sample["question"],
        "topics": topics,
        "hop": hop_num,
        "mmr_reason_paths": mmr_reason_paths,
        "mmr_answer_path_hit": bool(metrics["answer_hit"]),
        "mmr_top1_hit": bool(metrics["top1_hit"]),
        "path_diversity": {
            "jaccard_diversity": diversity.get("jaccard_diversity", 0.0),
            "relation_jaccard_diversity": diversity.get("relation_jaccard_diversity", 0.0),
            "tail_diversity": diversity.get("tail_diversity", 0.0),
            "relation_coverage": diversity.get("relation_coverage", 0.0),
            "edge_coverage": diversity.get("edge_coverage", 0.0),
        },
        "mmr_answer_recall": round(metrics["recall"], 4),
        "mmr_precision": round(metrics["precision"], 4),
        "mmr_f1": round(metrics["f1"], 4),
        "golden": golden,
        "prediction": prediction,
        "hit": bool(metrics["answer_hit"]),
    }


def _build_empty_record(sample: dict, hop_num: int, id2ent: dict) -> dict:
    """空路径样本的占位记录（路径检索失败）。"""
    golden = [id2ent.get(g, str(g)) for g in sample["gold_ids"]]
    topics = [id2ent.get(t, str(t)) for t in sample["topic_ids"]]
    prediction = {}
    for idx, val in zip(sample["e_score_indices"].tolist(),
                        sample["e_score_values"].tolist()):
        if val >= 0.5:
            mid = id2ent.get(idx, str(idx))
            prediction[mid] = round(val, 4)
    return {
        "question": sample["question"],
        "topics": topics,
        "hop": hop_num,
        "mmr_reason_paths": [],
        "mmr_answer_path_hit": False,
        "mmr_top1_hit": False,
        "path_diversity": {
            "jaccard_diversity": 0.0,
            "relation_jaccard_diversity": 0.0,
            "tail_diversity": 0.0,
            "relation_coverage": 0.0,
            "edge_coverage": 0.0,
        },
        "mmr_answer_recall": 0.0,
        "mmr_precision": 0.0,
        "mmr_f1": 0.0,
        "golden": golden,
        "prediction": prediction,
        "hit": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TransferNet 离线路径搜索实验")
    parser.add_argument("--cache", required=True,
                        help="predict.py 生成的得分缓存路径（.pt 文件）")
    parser.add_argument("--input_dir", default=None,
                        help="WebQSP 数据目录，用于重建全局 valid_edges_dict；"
                             "CWQ cache 含逐样本 triples 时不需要。"
                             "不提供时从缓存 meta.input_dir 自动读取。")
    parser.add_argument("--alpha_final", type=float, default=2.0,
                        help="最终实体得分融合权重（默认: 2.0）")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="实体/关系得分过滤阈值（默认: 0.01）")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="每个样本最终选取的路径数（默认: 3）")
    parser.add_argument("--lambda_val", type=float, default=0.2,
                        help="MMR 多样性惩罚系数（默认: 0.2）")
    parser.add_argument("--output", default=None,
                        help="逐样本结果输出路径（.jsonl），不指定则只打印聚合指标")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"[INFO] 加载得分缓存: {args.cache}", flush=True)
    cache = load_score_cache(args.cache)
    meta = cache["meta"]
    print(f"[INFO] 数据集={meta.get('dataset')}, split={meta.get('split')}, "
          f"样本数={meta.get('num_samples')}, topk_entities={meta.get('topk_entities')}", flush=True)

    samples_have_triples = any("triples" in sample for sample in cache.get("samples", []))
    if samples_have_triples:
        valid_edges_dict = None
        print("[INFO] 使用缓存中的逐样本 triples 重建 valid_edges_dict。", flush=True)
    else:
        input_dir = args.input_dir or meta.get("input_dir", "")
        if not input_dir or not os.path.isdir(input_dir):
            print(f"[ERROR] 无法找到数据目录: {input_dir!r}，请通过 --input_dir 指定。")
            sys.exit(1)

        print(f"[INFO] 重建 valid_edges_dict from: {input_dir}", flush=True)
        valid_edges_dict = rebuild_valid_edges_dict(input_dir)
        print(f"[INFO] 完成，共 {len(valid_edges_dict)} 个实体节点的出边。", flush=True)

    print(f"\n[RUN] selector=mmr, lambda={args.lambda_val}, "
          f"alpha_final={args.alpha_final}, threshold={args.threshold}, "
          f"beam_size={args.beam_size}", flush=True)

    metrics = run_experiment(
        cache, valid_edges_dict,
        threshold=args.threshold, beam_size=args.beam_size,
        output_path=args.output,
        alpha_final=args.alpha_final,
        lambda_val=args.lambda_val,
    )

    print("\n" + "=" * 60)
    print(f"  总样本数     : {metrics['total']}")
    print(f"  空路径数     : {metrics['empty_path']}")
    print(f"  Answer Hit   : {metrics['answer_hit_rate']:.4f}")
    print(f"  Top-1 Hit    : {metrics['top1_hit_rate']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  Edge Diversity: {metrics['diversity_edge']:.4f}")
    print(f"  Relation Diversity: {metrics['relation_jaccard_diversity']:.4f}")
    print(f"  Tail Unique  : {metrics['tail_unique']:.4f}")
    print(f"  Relation Coverage: {metrics['relation_coverage']:.4f}")
    print(f"  Edge Coverage: {metrics['edge_coverage']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
