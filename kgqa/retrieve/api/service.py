"""数据集无关的缓存路径检索服务层。

上移自 ``oh_my_agent/path_retrieve_server/service.py``(stage3):检索管线不变
(稀疏重建 → 逐跳 beam 展开 → MMR 选择 → 序列化),数据集耦合改经 adapter
(score_loader 加载缓存、kg_edge_source 提供邻接表);θ 从模块常量提升为
构造参数(默认 0.9,行为与 legacy 一致)。

注意:prediction 是 e_score ≥ θ 的全集口径(Ch5 agent 的 group_tails 过滤与
expansion 门依赖),与 ``engine.build_prediction`` 的 argmax 并列口径不同,
不可混用。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kgqa.retrieve.engine import (
    candidate_hop_numbers,
    candidate_to_tuple,
    drop_loopback_paths,
    final_ent_score_dict,
    path_to_triples,
    reconstruct_ent_dict,
    reconstruct_rel_dict,
    search_path_candidates,
    select_path_candidates,
    validate_penalty_mode,
    validate_score_scheme,
)

# TransferNet 实体预测阈值:e_score ≥ 此值视为"预测答案"。group_tails 过滤、
# prediction、下游 expansion 门必须共用同一阈值,否则三者集合漂移破坏等价性。
DEFAULT_PREDICTION_SCORE_THRESHOLD = 0.9


def normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\[(cls|sep|pad)\]", " ", text)
    text = re.sub(r"\s+##", "", text)   # merge BERT WordPiece subword tokens
    return re.sub(r"\s+", " ", text).strip()


def group_tails_from_path(
    node_ids: list[int],
    rel_ids: list[int],
    valid_edges_dict: dict[int, list[tuple[int, int]]],
    id2ent: dict[int, str],
    id2rel: dict[int, str],
    prediction_ids: Optional[set[int]] = None,
) -> Optional[tuple[str, list[str]]]:
    """实时算"(topic, 关系序列) → 全 KG 尾实体",在线替代离线 sidecar。

    沿全局 KG 邻接表(已含 _reverse 边)从 topic 节点逐跳遍历,返回与离线
    sidecar 对齐的 (key, sorted_tail_mids):key = 'topic_mid|rel_name1[|rel_name2]'。
    关系序列为空返回 None;遍历到死路返回空尾列表(key 仍可拼出)。

    传入 prediction_ids(TransferNet e_score≥θ 的实体 id)时,在**最后一跳**只收
    属于 prediction 的尾——下游 expansion 本就只用这些,提前过滤可阻止 frontier 在
    hub 节点(国家/类型)处膨胀到几十万,语义等价且消除长尾。中间跳不过滤(中间节点
    不在 prediction 内)。
    """
    if not node_ids or not rel_ids:
        return None
    last_hop = len(rel_ids) - 1
    frontier = {node_ids[0]}
    for hop, rel_id in enumerate(rel_ids):
        filter_pred = prediction_ids is not None and hop == last_hop
        frontier = {
            obj
            for node in frontier
            for rel, obj in valid_edges_dict.get(node, [])
            if rel == rel_id and (not filter_pred or obj in prediction_ids)
        }
        if not frontier:
            break
    topic_mid = id2ent.get(node_ids[0], str(node_ids[0]))
    rel_names = [id2rel.get(rel_id, str(rel_id)) for rel_id in rel_ids]
    key = "|".join([topic_mid, *rel_names])
    tails = sorted(id2ent.get(tail_id, str(tail_id)) for tail_id in frontier)
    return key, tails


@dataclass
class CachedRetrievalResult:
    question: str
    sample_index: int
    topics: list[str]
    hop: int
    mmr_reason_paths: list[dict[str, Any]]
    prediction: dict[str, float]
    elapsed_ms: float
    eta: float
    step_score_mode: str
    threshold: float
    beam_size: int
    lambda_val: float
    penalty_mode: str
    cache_path: str
    group_tails: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "sample_index": self.sample_index,
            "topics": self.topics,
            "hop": self.hop,
            "mmr_reason_paths": self.mmr_reason_paths,
            "prediction": self.prediction,
            "elapsed_ms": self.elapsed_ms,
            "eta": self.eta,
            "step_score_mode": self.step_score_mode,
            "threshold": self.threshold,
            "beam_size": self.beam_size,
            "lambda_val": self.lambda_val,
            "penalty_mode": self.penalty_mode,
            "cache_path": self.cache_path,
            "group_tails": self.group_tails,
        }


class PathRetrieveService:
    def __init__(
        self,
        adapter,
        *,
        cache_path: str,
        prediction_threshold: float = DEFAULT_PREDICTION_SCORE_THRESHOLD,
    ):
        self.adapter = adapter
        self.cache_path = str(Path(cache_path))
        self.prediction_threshold = float(prediction_threshold)
        self.bundle = adapter.score_loader().load(self.cache_path)
        self.samples = self.bundle.samples
        self.id2ent = self.bundle.meta.id2ent
        self.id2rel = self.bundle.meta.id2rel
        self.question_index = self._build_question_index()

    def _build_question_index(self) -> dict[str, int]:
        index: dict[str, int] = {}
        for i, sample in enumerate(self.samples):
            key = normalize_question(sample.question)
            index.setdefault(key, i)
        return index

    def retrieve(
        self,
        *,
        question: Optional[str] = None,
        sample_index: Optional[int] = None,
        topic_entities: Optional[list[str]] = None,
        eta: float = 1.0,
        step_score_mode: str = "joint",
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.2,
        penalty_mode: str = "adaptive",
        drop_loopback: bool = True,
    ) -> CachedRetrievalResult:
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        validate_score_scheme(step_score_mode, eta)
        validate_penalty_mode(penalty_mode)

        t0 = time.perf_counter()
        idx = self._resolve_sample_index(question, sample_index)
        sample = self.samples[idx]
        topics = self._topics(sample)
        if topic_entities is not None and set(topic_entities) != set(topics):
            raise ValueError(f"topic_entities mismatch: expected {topics}, got {topic_entities}")

        valid_edges_dict = self.adapter.kg_edge_source(sample).valid_edges_dict

        hop_num = int(sample.hop_attn.argmax().item()) + 1
        hop_nums = candidate_hop_numbers(len(sample.rel_probs))
        rel_dicts, ent_dicts = self._reconstruct_scores(sample, threshold, max(hop_nums))

        path_candidates = []
        final_scores = final_ent_score_dict(sample)
        for candidate_hop in hop_nums:
            path_candidates.extend(search_path_candidates(
                sample.topic_ids,
                rel_dicts,
                ent_dicts,
                candidate_hop,
                valid_edges_dict,
                beam_size,
                final_ent_scores=final_scores,
                order_start=len(path_candidates),
                step_score_mode=step_score_mode,
            ))

        selected = select_path_candidates(
            path_candidates,
            beam_size,
            eta=eta,
            lambda_val=lambda_val,
            penalty_mode=penalty_mode,
        )
        candidates = [candidate_to_tuple(candidate) for candidate in selected]
        paths = drop_loopback_paths(candidates) if drop_loopback else candidates
        prediction_ids = {
            int(ent_idx)
            for ent_idx, val in zip(
                sample.e_score_indices.tolist(), sample.e_score_values.tolist()
            )
            if float(val) >= self.prediction_threshold
        }
        group_tails = self._build_group_tails(paths, prediction_ids, valid_edges_dict)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return CachedRetrievalResult(
            question=sample.question,
            sample_index=idx,
            topics=topics,
            hop=hop_num,
            mmr_reason_paths=self._serialize_paths(paths),
            prediction=self._prediction(sample),
            elapsed_ms=round(elapsed_ms, 1),
            eta=eta,
            step_score_mode=step_score_mode,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
            penalty_mode=penalty_mode,
            cache_path=self.cache_path,
            group_tails=group_tails,
        )

    def _build_group_tails(
        self,
        paths: list[tuple[list[int], list[int], float]],
        prediction_ids: set[int],
        valid_edges_dict: dict[int, list[tuple[int, int]]],
    ) -> dict[str, list[str]]:
        """对每条选中路径的关系组(≤2 跳),用全局 KG 邻接表实时算尾实体,key 与
        离线 sidecar 对齐。最后一跳按 prediction 过滤(下游 expansion 只用这些),
        阻止 hub 组膨胀。同组只算一次。"""
        group_tails: dict[str, list[str]] = {}
        for node_ids, rel_ids, _score in paths:
            if not 1 <= len(rel_ids) <= 2:
                continue
            entry = group_tails_from_path(
                node_ids, rel_ids, valid_edges_dict,
                self.id2ent, self.id2rel, prediction_ids,
            )
            if entry is None:
                continue
            key, tails = entry
            if key not in group_tails:
                group_tails[key] = tails
        return group_tails

    def _resolve_sample_index(self, question: Optional[str], sample_index: Optional[int]) -> int:
        if sample_index is not None:
            if sample_index >= len(self.samples):
                raise IndexError(f"sample_index out of range: {sample_index}")
            return sample_index
        if not question:
            raise ValueError("question or sample_index is required")
        key = normalize_question(question)
        if key not in self.question_index:
            raise KeyError(f"question not found in cache: {question}")
        return self.question_index[key]

    def _reconstruct_scores(self, sample, threshold: float, hop_count: int):
        rel_dicts, ent_dicts = [], []
        for t in range(hop_count):
            rel_dicts.append(reconstruct_rel_dict(sample.rel_probs[t], threshold))
            ent_dicts.append(
                reconstruct_ent_dict(sample.ent_indices[t], sample.ent_scores[t], threshold)
            )
        return rel_dicts, ent_dicts

    def _topics(self, sample) -> list[str]:
        return [self.id2ent.get(int(topic_id), str(topic_id)) for topic_id in sample.topic_ids]

    def _prediction(self, sample) -> dict[str, float]:
        prediction = {}
        for ent_idx, val in zip(sample.e_score_indices.tolist(), sample.e_score_values.tolist()):
            if float(val) >= self.prediction_threshold:
                prediction[self.id2ent.get(int(ent_idx), str(ent_idx))] = round(float(val), 4)
        return prediction

    def _serialize_paths(self, paths: list[tuple[list[int], list[int], float]]):
        return [
            {
                "path": path_to_triples(nodes, rels, self.id2ent, self.id2rel),
                "log_score": round(float(score), 6),
            }
            for nodes, rels, score in paths
        ]

    def info(self) -> dict[str, Any]:
        return {
            "cache_path": self.cache_path,
            "dataset": self.bundle.meta.dataset,
            "split": self.bundle.meta.split,
            "qa_file": self.bundle.meta.qa_file,
            "num_samples": len(self.samples),
            "topk_entities": self.bundle.meta.topk_entities,
            "entity_count": len(self.id2ent),
            "relation_count": len(self.id2rel),
            "prediction_threshold": self.prediction_threshold,
        }
