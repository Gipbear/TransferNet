"""统一路径检索内核与单样本编排。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Optional

from utils.path_utils import path_to_rel_set

from kgqa.types import RetrieveResult
from kgqa.kg.base import KGEdgeSource

EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 以下为逐字迁移内核（数值红线）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathCandidate:
    """A path plus cached features used by offline reranking."""

    nodes: list[int]
    rels: list[int]
    hop: int
    base_score: float
    final_tail_score: float = 0.0
    order: int = 0
    score: Optional[float] = None

    def __post_init__(self):
        if self.score is None:
            object.__setattr__(self, "score", self.base_score)


def compute_candidate_score(
    candidate: PathCandidate,
    alpha_final: float,
) -> float:
    """融合路径分数与最终实体分数，并按路径长度归一化。"""
    score = candidate.base_score + alpha_final * math.log(
        max(candidate.final_tail_score, 0.0) + EPS
    )
    return score / max(candidate.hop, 1)


def score_path_candidates(
    candidates: list[PathCandidate],
    alpha_final: float,
) -> list[PathCandidate]:
    """返回写入融合分数后的候选路径。"""
    return [
        replace(candidate, score=compute_candidate_score(candidate, alpha_final))
        for candidate in candidates
    ]


def _ranked_candidates(candidates: list[PathCandidate]) -> list[PathCandidate]:
    return sorted(
        candidates,
        key=lambda c: (-(c.score if c.score is not None else -float("inf")), c.order),
    )


def candidate_to_tuple(candidate: PathCandidate) -> tuple[list[int], list[int], float]:
    score = candidate.score if candidate.score is not None else candidate.base_score
    return (candidate.nodes, candidate.rels, float(score))


def select_path_candidates(
    candidates: list[PathCandidate],
    k: int,
    alpha_final: float,
    lambda_val: float,
) -> list[PathCandidate]:
    """Score and select candidates with fixed MMR and deterministic tie-breaking."""
    if not candidates or k <= 0:
        return []

    scored = _ranked_candidates(score_path_candidates(candidates, alpha_final))

    rel_sets = [path_to_rel_set(c.rels) for c in scored]
    selected = []
    remaining = list(range(len(scored)))
    max_sims = [0.0] * len(scored)
    while len(selected) < k and remaining:
        best_idx: Optional[int] = None
        best_mmr = -float("inf")
        for idx in remaining:
            candidate = scored[idx]
            base_score = float(candidate.score)
            mmr_score = base_score - lambda_val * max_sims[idx] * abs(base_score)
            current_best_order = (
                -(scored[best_idx].order) if best_idx is not None else -10**18
            )
            if (mmr_score, -candidate.order) > (best_mmr, current_best_order):
                best_mmr = mmr_score
                best_idx = idx

        selected.append(scored[best_idx])
        remaining.remove(best_idx)
        sel_rel_set = rel_sets[best_idx]
        for idx in remaining:
            union = rel_sets[idx] | sel_rel_set
            sim = len(rel_sets[idx] & sel_rel_set) / len(union) if union else 0.0
            if sim > max_sims[idx]:
                max_sims[idx] = sim
    return selected


def reconstruct_ent_dict(indices, scores, threshold: float) -> dict[int, float]:
    """从稀疏 top-K 表示重建 {entity_id: score} 字典，过滤低于阈值的条目。"""
    mask = scores >= threshold
    return {int(i): float(s) for i, s in zip(indices[mask], scores[mask])}


def reconstruct_rel_dict(rel_probs, threshold: float) -> dict[int, float]:
    """从密集关系得分向量重建 {rel_id: score} 字典。"""
    mask = rel_probs >= threshold
    idxs = mask.nonzero(as_tuple=True)[0]
    return {int(i): float(rel_probs[i]) for i in idxs}


def compute_step_score(
    rel_dict: dict[int, float],
    ent_dict: dict[int, float],
    rel_id: int,
    tail_id: int,
) -> float:
    """计算单跳的关系与实体对数归一化分数。"""
    rel_score = rel_dict.get(rel_id, 0.0)
    ent_score = ent_dict.get(tail_id, 0.0)
    if rel_score <= 0 or ent_score <= 0:
        return -float("inf")
    sum_rel = sum(rel_dict.values()) or 1.0
    sum_ent = sum(ent_dict.values()) or 1.0
    return math.log(rel_score / sum_rel + EPS) + math.log(ent_score / sum_ent + EPS)


def search_path_candidates(
    topic_ids: list[int],
    rel_dicts: list[dict[int, float]],
    ent_dicts: list[dict[int, float]],
    hop_num: int,
    valid_edges_dict: dict[int, list[tuple[int, int]]],
    beam_size: int,
    final_ent_scores: Optional[dict[int, float]] = None,
    order_start: int = 0,
) -> list[PathCandidate]:
    """Beam-expand candidate paths and attach cached features for reranking."""
    beam: list[tuple[list[int], list[int], float]] = [([t_id], [], 0.0) for t_id in topic_ids]
    order = order_start

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
                step = compute_step_score(rel_dict, ent_dict, rel_id, tail_id)
                if step == -float("inf"):
                    continue
                next_candidates.append((nodes + [tail_id], rels + [rel_id], cur_score + step))

        if not next_candidates:
            return []
        next_candidates.sort(key=lambda x: x[2], reverse=True)
        beam = next_candidates[: beam_size * 10]

    candidates = []
    final_scores = final_ent_scores or {}
    for nodes, rels, score in sorted(beam, key=lambda x: x[2], reverse=True):
        tail_id = nodes[-1]
        candidates.append(PathCandidate(
            nodes=nodes,
            rels=rels,
            hop=hop_num,
            base_score=score,
            final_tail_score=final_scores.get(tail_id, 0.0),
            order=order,
        ))
        order += 1
    return candidates


def path_to_triples(
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


def candidate_hop_numbers(num_steps: int) -> list[int]:
    """当前检索器在所有可用 hop 上生成候选路径。"""
    return list(range(1, num_steps + 1))


# ─────────────────────────────────────────────────────────────────────────────
# 新增编排（SampleScoreLike 属性访问口径）
# ─────────────────────────────────────────────────────────────────────────────

def drop_loopback_paths(paths):
    """剔除尾==topic 的自指路径（迁移自 path_retrieve_server/service.py）。"""
    return [
        (node_ids, rel_ids, score)
        for node_ids, rel_ids, score in paths
        if not node_ids or node_ids[-1] != node_ids[0]
    ]


def final_ent_score_dict(sample) -> dict[int, float]:
    return {
        int(idx): float(val)
        for idx, val in zip(sample.e_score_indices.tolist(), sample.e_score_values.tolist())
    }


PREDICTION_TIE_EPS = 1e-6


def build_prediction(sample, id2ent: dict) -> dict[str, float]:
    """预测答案 = e_score 达到最高分的所有实体（含并列）。

    源自原始 TransferNet WebQSP/predict.py 的 `torch.max(e_score)` argmax 口径，
    但 WebQSP 多答案样本常出现多个 gold 并列最高分（如 e_score 前两位完全相等），
    argmax 只取其一会漏掉并列 gold、压低 recall/F1。故取所有与最大值相等
    （容差 PREDICTION_TIE_EPS）的实体。首个（e_score 降序的 argmax）用于 hit1，
    与原始 acc 口径一致。"""
    vals = sample.e_score_values
    vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
    if not vals:
        return {}
    idxs = sample.e_score_indices
    idxs = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
    max_val = max(vals)
    prediction: dict[str, float] = {}
    for idx, val in zip(idxs, vals):
        if float(val) >= max_val - PREDICTION_TIE_EPS:
            prediction[id2ent.get(int(idx), str(int(idx)))] = round(float(val), 4)
    return prediction


def _serialize_paths(paths, id2ent: dict, id2rel: dict) -> list[dict]:
    return [
        {"path": path_to_triples(nodes, rels, id2ent, id2rel),
         "log_score": round(float(score), 6)}
        for nodes, rels, score in paths
    ]


def retrieve_one(sample, edge_source: KGEdgeSource, id2ent: dict, id2rel: dict, *,
                 alpha_final: float = 1.0,
                 threshold: float = 0.01, beam_size: int = 50,
                 lambda_val: float = 0.2, drop_loopback: bool = True) -> RetrieveResult:
    """单样本检索：稀疏重建 → 逐跳 beam 展开 → MMR 选择 → 序列化。

    与 offline_path_search.run_experiment 的单样本分支逻辑等价（Task 11 回归锁定）。"""
    t0 = time.perf_counter()
    valid_edges_dict = getattr(edge_source, "valid_edges_dict", None)

    hop_num = int(sample.hop_attn.argmax().item()) + 1
    hop_nums = candidate_hop_numbers(len(sample.rel_probs))

    rel_dicts, ent_dicts = [], []
    for t in range(max(hop_nums)):
        rel_dicts.append(reconstruct_rel_dict(sample.rel_probs[t], threshold))
        ent_dicts.append(reconstruct_ent_dict(sample.ent_indices[t], sample.ent_scores[t], threshold))

    final_scores = final_ent_score_dict(sample)
    path_candidates = []
    for candidate_hop in hop_nums:
        path_candidates.extend(search_path_candidates(
            sample.topic_ids, rel_dicts, ent_dicts, candidate_hop,
            valid_edges_dict, beam_size,
            final_ent_scores=final_scores, order_start=len(path_candidates),
        ))

    selected = select_path_candidates(
        path_candidates, beam_size, alpha_final=alpha_final, lambda_val=lambda_val,
    )
    candidates = [candidate_to_tuple(c) for c in selected]
    if drop_loopback:
        candidates = drop_loopback_paths(candidates)

    topics = [id2ent.get(int(t), str(int(t))) for t in sample.topic_ids]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return RetrieveResult(
        question=sample.question,
        topics=topics,
        hop=hop_num,
        paths=_serialize_paths(candidates, id2ent, id2rel),
        prediction=build_prediction(sample, id2ent),
        elapsed_ms=elapsed_ms,
        sample_index=getattr(sample, "sample_index", -1),
    )
