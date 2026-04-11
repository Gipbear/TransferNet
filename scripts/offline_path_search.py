"""离线路径搜索实验脚本

从 WebQSP/predict.py 生成的得分缓存（.pt 文件）加载中间得分矩阵，
在 CPU 上快速重放路径搜索，支持可插拔的评分策略和多样性策略。

典型用法：

  # 复现在线 MMR 结果（用于验证一致性）
  python scripts/offline_path_search.py \\
      --cache output/score_cache/webqsp_val.pt \\
      --input_dir data/WebQSP \\
      --scoring log_norm --diversity mmr \\
      --threshold 0.01 --beam_size 3 --lambda_val 0.5

  # 网格搜索
  for thresh in 0.001 0.005 0.01 0.05; do
    for lam in 0.0 0.3 0.5 0.8; do
      python scripts/offline_path_search.py \\
          --cache output/score_cache/webqsp_val.pt \\
          --input_dir data/WebQSP \\
          --threshold $thresh --lambda_val $lam --beam_size 5
    done
  done
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.path_utils import (
    build_valid_edges_dict,
    compute_path_diversity,
    compute_path_metrics,
    path_to_rel_set,
)

EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 缓存加载与稀疏重建
# ─────────────────────────────────────────────────────────────────────────────

def load_score_cache(path: str) -> dict:
    """加载 predict.py 生成的得分缓存文件。"""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    ver = cache.get("version", 0)
    if ver < 1:
        raise ValueError(f"不支持的缓存版本: {ver}，请用最新 predict.py 重新生成。")
    return cache


def reconstruct_ent_dict(indices: torch.Tensor, scores: torch.Tensor,
                         threshold: float) -> dict[int, float]:
    """从稀疏 top-K 表示重建 {entity_id: score} 字典，过滤低于阈值的条目。"""
    mask = scores >= threshold
    return {int(i): float(s) for i, s in zip(indices[mask], scores[mask])}


def reconstruct_rel_dict(rel_probs: torch.Tensor,
                         threshold: float) -> dict[int, float]:
    """从密集关系得分向量重建 {rel_id: score} 字典。"""
    mask = rel_probs >= threshold
    idxs = mask.nonzero(as_tuple=True)[0]
    return {int(i): float(rel_probs[i]) for i in idxs}


def rebuild_valid_edges_dict(input_dir: str) -> dict[int, list[tuple[int, int]]]:
    """从 fbwq_full/train.txt 重建 KG 邻接表（与 predict.py 相同逻辑）。"""
    fb_dir = Path(input_dir) / "fbwq_full"

    ent2id: dict[str, int] = {}
    with (fb_dir / "entities.dict").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                ent2id[parts[0].strip()] = len(ent2id)

    rel2id: dict[str, int] = {}
    with (fb_dir / "relations.dict").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rel2id[parts[0].strip()] = int(parts[1])

    triples: list[list[int]] = []
    with (fb_dir / "train.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if s not in ent2id or r not in rel2id or o not in ent2id:
                continue
            sid, rid, oid = ent2id[s], rel2id[r], ent2id[o]
            triples.append([sid, rid, oid])
            rev = r + "_reverse"
            if rev in rel2id:
                triples.append([oid, rel2id[rev], sid])

    return build_valid_edges_dict(triples)


# ─────────────────────────────────────────────────────────────────────────────
# 可插拔评分策略
# ─────────────────────────────────────────────────────────────────────────────

class ScoringStrategy(ABC):
    """评分策略基类。

    给定单步的关系得分字典、实体得分字典和候选 (rel_id, tail_id)，
    返回该步的得分增量（越大越好）。
    """

    @abstractmethod
    def score_step(self, rel_dict: dict[int, float], ent_dict: dict[int, float],
                   rel_id: int, tail_id: int) -> float:
        ...

    def name(self) -> str:
        return self.__class__.__name__


class LogNormStrategy(ScoringStrategy):
    """当前默认策略：log(局部归一化关系得分) + log(全局归一化实体得分)。

    与 mmr_diversity_beam_search 的 Plan-A 逻辑完全一致，用于验证离线复现。
    """

    def score_step(self, rel_dict: dict[int, float], ent_dict: dict[int, float],
                   rel_id: int, tail_id: int) -> float:
        rel_score = rel_dict.get(rel_id, 0.0)
        ent_score = ent_dict.get(tail_id, 0.0)
        if rel_score <= 0 or ent_score <= 0:
            return -float("inf")
        sum_rel = sum(rel_dict.values()) or 1.0
        sum_ent = sum(ent_dict.values()) or 1.0
        local_rel = rel_score / sum_rel
        local_ent = ent_score / sum_ent
        return math.log(local_rel + EPS) + math.log(local_ent + EPS)

    def name(self) -> str:
        return "log_norm"


class RawProductStrategy(ScoringStrategy):
    """rel_score × ent_score 的对数，不做归一化。"""

    def score_step(self, rel_dict: dict[int, float], ent_dict: dict[int, float],
                   rel_id: int, tail_id: int) -> float:
        rel_score = rel_dict.get(rel_id, 0.0)
        ent_score = ent_dict.get(tail_id, 0.0)
        if rel_score <= 0 or ent_score <= 0:
            return -float("inf")
        return math.log(rel_score + EPS) + math.log(ent_score + EPS)

    def name(self) -> str:
        return "raw_product"


class SoftmaxRelStrategy(ScoringStrategy):
    """对关系得分做 softmax 归一化后取 log；实体得分保持全局归一化。"""

    def score_step(self, rel_dict: dict[int, float], ent_dict: dict[int, float],
                   rel_id: int, tail_id: int) -> float:
        rel_score = rel_dict.get(rel_id, 0.0)
        ent_score = ent_dict.get(tail_id, 0.0)
        if rel_score <= 0 or ent_score <= 0:
            return -float("inf")
        # softmax over all relations in rel_dict
        max_r = max(rel_dict.values())
        exp_sum = sum(math.exp(v - max_r) for v in rel_dict.values()) or 1.0
        softmax_rel = math.exp(rel_score - max_r) / exp_sum
        sum_ent = sum(ent_dict.values()) or 1.0
        local_ent = ent_score / sum_ent
        return math.log(softmax_rel + EPS) + math.log(local_ent + EPS)

    def name(self) -> str:
        return "softmax_rel"


SCORING_STRATEGIES: dict[str, type[ScoringStrategy]] = {
    "log_norm": LogNormStrategy,
    "raw_product": RawProductStrategy,
    "softmax_rel": SoftmaxRelStrategy,
}


# ─────────────────────────────────────────────────────────────────────────────
# 可插拔多样性策略
# ─────────────────────────────────────────────────────────────────────────────

class DiversityStrategy(ABC):
    """多样性选择策略基类。

    输入有序候选列表（已按得分降序排列），选出 k 条路径。
    每条路径格式：(nodes: list[int], rels: list[int], score: float)
    """

    @abstractmethod
    def select(self, candidates: list[tuple[list[int], list[int], float]],
               k: int) -> list[tuple[list[int], list[int], float]]:
        ...

    def name(self) -> str:
        return self.__class__.__name__


class NoDiversity(DiversityStrategy):
    """纯 top-K，无多样性惩罚。"""

    def select(self, candidates, k):
        return candidates[:k]

    def name(self) -> str:
        return "none"


class MMRDiversity(DiversityStrategy):
    """基于关系集合 Jaccard 相似度的 MMR 多样性选择。

    与 mmr_diversity_beam_search 的 MMR 部分逻辑完全一致。
    """

    def __init__(self, lambda_val: float = 0.5):
        self.lambda_val = lambda_val

    def select(self, candidates, k):
        if not candidates or k <= 0:
            return []
        rel_sets = [path_to_rel_set(rels) for (_, rels, _) in candidates]
        selected: list[tuple[list[int], list[int], float]] = []
        remaining = list(range(len(candidates)))
        max_sims = [0.0] * len(candidates)

        while len(selected) < k and remaining:
            best_idx: Optional[int] = None
            best_mmr = -float("inf")
            for idx in remaining:
                score = candidates[idx][2]
                mmr_score = score - self.lambda_val * max_sims[idx] * abs(score)
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected.append(candidates[best_idx])
            remaining.remove(best_idx)
            sel_rel_set = rel_sets[best_idx]
            for idx in remaining:
                union = rel_sets[idx] | sel_rel_set
                sim = len(rel_sets[idx] & sel_rel_set) / len(union) if union else 0.0
                if sim > max_sims[idx]:
                    max_sims[idx] = sim

        return selected

    def name(self) -> str:
        return f"mmr(λ={self.lambda_val})"


class MaxCoverDiversity(DiversityStrategy):
    """贪心最大化尾实体覆盖：每次选尾实体集合扩充最多的候选。"""

    def select(self, candidates, k):
        covered: set[int] = set()
        selected = []
        remaining = list(candidates)
        while len(selected) < k and remaining:
            best = max(remaining,
                       key=lambda c: (len(set([c[0][-1]]) - covered), c[2]))
            selected.append(best)
            covered.add(best[0][-1])
            remaining.remove(best)
        return selected

    def name(self) -> str:
        return "max_cover"


DIVERSITY_STRATEGIES: dict[str, type] = {
    "none": NoDiversity,
    "mmr": MMRDiversity,
    "max_cover": MaxCoverDiversity,
}


# ─────────────────────────────────────────────────────────────────────────────
# 路径搜索（beam 展开 + 策略打分）
# ─────────────────────────────────────────────────────────────────────────────

def search_paths(
    topic_ids: list[int],
    rel_dicts: list[dict[int, float]],
    ent_dicts: list[dict[int, float]],
    hop_num: int,
    valid_edges_dict: dict[int, list[tuple[int, int]]],
    scoring: ScoringStrategy,
    beam_size: int,
) -> list[tuple[list[int], list[int], float]]:
    """从 topic 出发，逐跳展开 beam，用 scoring 策略打分，返回所有候选路径。

    返回按得分降序排列的 (nodes, rels, score) 列表（未做多样性选择）。
    """
    # 初始 beam：每个 topic entity 作为起点，初始得分 0
    beam: list[tuple[list[int], list[int], float]] = [
        ([t_id], [], 0.0) for t_id in topic_ids
    ]

    for t in range(hop_num):
        rel_dict = rel_dicts[t]
        ent_dict = ent_dicts[t]
        next_candidates: list[tuple[list[int], list[int], float]] = []

        for nodes, rels, cur_score in beam:
            head = nodes[-1]
            edges = valid_edges_dict.get(head, [])
            for rel_id, tail_id in edges:
                if rel_id not in rel_dict or tail_id not in ent_dict:
                    continue
                step = scoring.score_step(rel_dict, ent_dict, rel_id, tail_id)
                if step == -float("inf"):
                    continue
                next_candidates.append((
                    nodes + [tail_id],
                    rels + [rel_id],
                    cur_score + step,
                ))

        if not next_candidates:
            break
        # 按得分降序，保留 beam 窗口（宽松：保留 beam_size * 10 以供多样性选择）
        next_candidates.sort(key=lambda x: x[2], reverse=True)
        beam = next_candidates[: beam_size * 10]

    # 最终按得分降序
    beam.sort(key=lambda x: x[2], reverse=True)
    return beam


# ─────────────────────────────────────────────────────────────────────────────
# 实验主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _path_to_triples(
    nodes: list[int], rels: list[int],
    id2ent: dict, id2rel: dict,
) -> list[list[str]]:
    """将 (nodes, rels) 路径转换为 [[head_mid, rel_str, tail_mid], ...] 格式。"""
    return [
        [id2ent.get(nodes[i], str(nodes[i])),
         id2rel.get(rels[i], str(rels[i])),
         id2ent.get(nodes[i + 1], str(nodes[i + 1]))]
        for i in range(len(rels))
    ]


def run_experiment(
    cache: dict,
    valid_edges_dict: dict,
    scoring: ScoringStrategy,
    diversity: DiversityStrategy,
    threshold: float,
    beam_size: int,
    output_path: Optional[str] = None,
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
        "diversity_edge": 0.0, "tail_unique": 0.0,
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
        for sample in tqdm(samples, desc="search", unit="sample", dynamic_ncols=True):
            hop_num = int(sample["hop_attn"].argmax().item()) + 1
            topic_ids = sample["topic_ids"]
            gold_ids = set(sample["gold_ids"])

            # 重建每跳的稀疏字典
            rel_dicts, ent_dicts = [], []
            for t in range(hop_num):
                rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], threshold))
                ed = reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], threshold)
                ent_dicts.append(ed)
                # 截断风险检测：若 top-K 末尾得分仍 >= threshold，可能有更多候选被丢弃
                scores_t = sample["ent_scores"][t]
                if len(scores_t) == topk and float(scores_t[-1]) >= threshold:
                    truncation_warnings += 1

            # beam 展开
            candidates = search_paths(
                topic_ids, rel_dicts, ent_dicts, hop_num,
                valid_edges_dict, scoring, beam_size,
            )

            if not candidates:
                agg["empty_path"] += 1
                if out_file:
                    record = _build_empty_record(sample, hop_num, id2ent)
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # 多样性选择
            selected = diversity.select(candidates, beam_size)

            # 评估
            m = compute_path_metrics(selected, gold_ids)
            d = compute_path_diversity(selected)

            agg["answer_hit"] += int(m["answer_hit"])
            agg["top1_hit"] += int(m["top1_hit"])
            agg["precision"] += m["precision"]
            agg["recall"] += m["recall"]
            agg["f1"] += m["f1"]
            agg["diversity_edge"] += d.get("jaccard_diversity", 0.0)
            agg["tail_unique"] += d.get("tail_diversity", 0.0)

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
        "tail_unique": agg["tail_unique"] / n,
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
        triples = _path_to_triples(nodes, rels, id2ent, id2rel)
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
            "tail_diversity": diversity.get("tail_diversity", 0.0),
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
        "path_diversity": {"jaccard_diversity": 0.0, "tail_diversity": 0.0, "edge_coverage": 0.0},
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

def main():
    parser = argparse.ArgumentParser(description="TransferNet 离线路径搜索实验")
    parser.add_argument("--cache", required=True,
                        help="predict.py 生成的得分缓存路径（.pt 文件）")
    parser.add_argument("--input_dir", default=None,
                        help="WebQSP 数据目录，用于重建 valid_edges_dict。"
                             "不提供时从缓存 meta.input_dir 自动读取。")
    parser.add_argument("--scoring", default="log_norm",
                        choices=list(SCORING_STRATEGIES.keys()),
                        help="评分策略（默认: log_norm，与在线结果一致）")
    parser.add_argument("--diversity", default="mmr",
                        choices=list(DIVERSITY_STRATEGIES.keys()),
                        help="多样性策略（默认: mmr）")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="实体/关系得分过滤阈值（默认: 0.01）")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="每个样本最终选取的路径数（默认: 3）")
    parser.add_argument("--lambda_val", type=float, default=0.5,
                        help="MMR 多样性惩罚系数（仅对 --diversity mmr 有效，默认: 0.5）")
    parser.add_argument("--output", default=None,
                        help="逐样本结果输出路径（.jsonl），不指定则只打印聚合指标")
    args = parser.parse_args()

    print(f"[INFO] 加载得分缓存: {args.cache}", flush=True)
    cache = load_score_cache(args.cache)
    meta = cache["meta"]
    print(f"[INFO] 数据集={meta.get('dataset')}, split={meta.get('split')}, "
          f"样本数={meta.get('num_samples')}, topk_entities={meta.get('topk_entities')}", flush=True)

    input_dir = args.input_dir or meta.get("input_dir", "")
    if not input_dir or not os.path.isdir(input_dir):
        print(f"[ERROR] 无法找到数据目录: {input_dir!r}，请通过 --input_dir 指定。")
        sys.exit(1)

    print(f"[INFO] 重建 valid_edges_dict from: {input_dir}", flush=True)
    valid_edges_dict = rebuild_valid_edges_dict(input_dir)
    print(f"[INFO] 完成，共 {len(valid_edges_dict)} 个实体节点的出边。", flush=True)

    # 初始化策略
    scoring = SCORING_STRATEGIES[args.scoring]()
    if args.diversity == "mmr":
        diversity = MMRDiversity(lambda_val=args.lambda_val)
    else:
        diversity = DIVERSITY_STRATEGIES[args.diversity]()

    print(f"\n[RUN] scoring={scoring.name()}, diversity={diversity.name()}, "
          f"threshold={args.threshold}, beam_size={args.beam_size}", flush=True)

    metrics = run_experiment(
        cache, valid_edges_dict, scoring, diversity,
        threshold=args.threshold, beam_size=args.beam_size,
        output_path=args.output,
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
    print(f"  Tail Unique  : {metrics['tail_unique']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
