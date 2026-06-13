"""Cached offline path retrieval service.

This module intentionally keeps only the single-sample retrieval path used by
``scripts.offline_path_search``: reconstruct sparse scores, expand candidates,
apply tail_blend scoring, then select paths with MMR.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scripts.offline_path_search import (
    LogNormStrategy,
    _method_hop_numbers,
    _path_to_triples,
    candidate_to_tuple,
    final_ent_score_dict,
    load_score_cache,
    rebuild_valid_edges_dict,
    reconstruct_ent_dict,
    reconstruct_rel_dict,
    search_path_candidates,
    select_path_candidates,
)


# TransferNet 实体预测阈值:e_score ≥ 此值视为"预测答案"。group_tails 过滤、
# _prediction、下游 expansion 门必须共用同一阈值,否则三者集合漂移破坏等价性。
PREDICTION_SCORE_THRESHOLD = 0.9


def normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"\[(cls|sep|pad)\]", " ", text)
    text = re.sub(r"\s+##", "", text)   # merge BERT WordPiece subword tokens
    return re.sub(r"\s+", " ", text).strip()


def drop_loopback_paths(
    paths: list[tuple[list[int], list[int], float]],
) -> list[tuple[list[int], list[int], float]]:
    """剔除"绕回 topic"的无效路径——尾实体(node_ids[-1])等于 topic(node_ids[0])。

    答案=被问的实体本身在逻辑上不可能成立:WebQSP test 全集此类路径 9777 条
    (占 13.4%),**0 条尾是 gold**。源头剔除后 LLM 看不到这些路径,既不会引用、
    也不会被诱导产出自指答案,零损失(无 gold 反例)。
    """
    return [
        (node_ids, rel_ids, score)
        for node_ids, rel_ids, score in paths
        if not node_ids or node_ids[-1] != node_ids[0]
    ]


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

    传入 prediction_ids(TransferNet e_score≥0.9 的实体 id)时,在**最后一跳**只收
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
    method: str
    alpha_final: float
    threshold: float
    beam_size: int
    lambda_val: float
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
            "method": self.method,
            "alpha_final": self.alpha_final,
            "threshold": self.threshold,
            "beam_size": self.beam_size,
            "lambda_val": self.lambda_val,
            "cache_path": self.cache_path,
            "group_tails": self.group_tails,
        }


class CachedPathRetriever:
    def __init__(self, *, cache_path: str, input_dir: str):
        self.cache_path = str(Path(cache_path))
        self.input_dir = str(Path(input_dir))
        self.cache = load_score_cache(self.cache_path)
        self.samples = self.cache["samples"]
        self.meta = self.cache["meta"]
        self.id2ent = self.meta.get("id2ent", {})
        self.id2rel = self.meta.get("id2rel", {})
        self.valid_edges_dict = rebuild_valid_edges_dict(self.input_dir)
        self.question_index = self._build_question_index()

    def _build_question_index(self) -> dict[str, int]:
        index: dict[str, int] = {}
        for i, sample in enumerate(self.samples):
            key = normalize_question(sample["question"])
            index.setdefault(key, i)
        return index

    def retrieve(
        self,
        *,
        question: Optional[str] = None,
        sample_index: Optional[int] = None,
        topic_entities: Optional[list[str]] = None,
        method: str = "tail_blend",
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.2,
    ) -> CachedRetrievalResult:
        if method not in {"tail_blend", "baseline"}:
            raise ValueError(f"unknown method: {method}")
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")

        t0 = time.perf_counter()
        idx = self._resolve_sample_index(question, sample_index)
        sample = self.samples[idx]
        topics = self._topics(sample)
        if topic_entities is not None and set(topic_entities) != set(topics):
            raise ValueError(f"topic_entities mismatch: expected {topics}, got {topic_entities}")

        hop_num = int(sample["hop_attn"].argmax().item()) + 1
        hop_nums = _method_hop_numbers(method, hop_num, len(sample["rel_probs"]))
        rel_dicts, ent_dicts = self._reconstruct_scores(sample, threshold, max(hop_nums))

        scoring = LogNormStrategy()
        path_candidates = []
        final_scores = final_ent_score_dict(sample)
        for candidate_hop in hop_nums:
            path_candidates.extend(search_path_candidates(
                sample["topic_ids"],
                rel_dicts,
                ent_dicts,
                candidate_hop,
                self.valid_edges_dict,
                scoring,
                beam_size,
                final_ent_scores=final_scores,
                order_start=len(path_candidates),
            ))

        selected = select_path_candidates(
            path_candidates,
            beam_size,
            method=method,
            alpha_final=alpha_final,
            lambda_val=lambda_val,
        )
        paths = drop_loopback_paths([candidate_to_tuple(candidate) for candidate in selected])
        prediction_ids = {
            int(idx)
            for idx, val in zip(
                sample["e_score_indices"].tolist(), sample["e_score_values"].tolist()
            )
            if float(val) >= PREDICTION_SCORE_THRESHOLD
        }
        group_tails = self._build_group_tails(paths, prediction_ids)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return CachedRetrievalResult(
            question=sample["question"],
            sample_index=idx,
            topics=topics,
            hop=hop_num,
            mmr_reason_paths=self._serialize_paths(paths),
            prediction=self._prediction(sample),
            elapsed_ms=round(elapsed_ms, 1),
            method=method,
            alpha_final=alpha_final,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
            cache_path=self.cache_path,
            group_tails=group_tails,
        )

    def _build_group_tails(
        self,
        paths: list[tuple[list[int], list[int], float]],
        prediction_ids: set[int],
    ) -> dict[str, list[str]]:
        """对每条选中路径的关系组(≤2 跳),用全局 KG 邻接表实时算尾实体,key 与
        离线 sidecar 对齐。最后一跳按 prediction 过滤(下游 expansion 只用这些),
        阻止 hub 组膨胀。同组只算一次。"""
        group_tails: dict[str, list[str]] = {}
        for node_ids, rel_ids, _score in paths:
            if not 1 <= len(rel_ids) <= 2:
                continue
            entry = group_tails_from_path(
                node_ids, rel_ids, self.valid_edges_dict,
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

    def _reconstruct_scores(self, sample: dict, threshold: float, hop_count: int):
        rel_dicts, ent_dicts = [], []
        for t in range(hop_count):
            rel_dicts.append(reconstruct_rel_dict(sample["rel_probs"][t], threshold))
            ent_dicts.append(
                reconstruct_ent_dict(sample["ent_indices"][t], sample["ent_scores"][t], threshold)
            )
        return rel_dicts, ent_dicts

    def _topics(self, sample: dict) -> list[str]:
        return [self.id2ent.get(int(topic_id), str(topic_id)) for topic_id in sample["topic_ids"]]

    def _prediction(self, sample: dict) -> dict[str, float]:
        prediction = {}
        for idx, val in zip(sample["e_score_indices"].tolist(), sample["e_score_values"].tolist()):
            if float(val) >= PREDICTION_SCORE_THRESHOLD:
                prediction[self.id2ent.get(int(idx), str(idx))] = round(float(val), 4)
        return prediction

    def _serialize_paths(self, paths: list[tuple[list[int], list[int], float]]):
        return [
            {
                "path": _path_to_triples(nodes, rels, self.id2ent, self.id2rel),
                "log_score": round(float(score), 6),
            }
            for nodes, rels, score in paths
        ]

    def info(self) -> dict[str, Any]:
        return {
            "cache_path": self.cache_path,
            "input_dir": self.input_dir,
            "dataset": self.meta.get("dataset"),
            "split": self.meta.get("split"),
            "qa_file": self.meta.get("qa_file"),
            "num_samples": len(self.samples),
            "num_steps": self.meta.get("num_steps"),
            "topk_entities": self.meta.get("topk_entities"),
            "entity_count": self.meta.get("num_entities"),
            "relation_count": self.meta.get("num_relations"),
            "edge_source_count": len(self.valid_edges_dict),
        }
