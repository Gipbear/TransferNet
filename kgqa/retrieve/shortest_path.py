"""基于骨干候选答案的有界最短路径后处理基线。"""
from __future__ import annotations

import math
import time
import weakref
from collections import deque
from dataclasses import dataclass

from kgqa.core.contracts import RetrieveResult
from kgqa.retrieve.engine import EPS, build_prediction, path_to_triples
from kgqa.retrieve.graph.base import KGEdgeSource


@dataclass(frozen=True)
class ShortestPathParams:
    """最短路径后处理的固定预算。"""

    candidate_topk: int = 20
    max_paths_per_pair: int = 20
    path_budget: int = 20
    drop_loopback: bool = True

    def __post_init__(self) -> None:
        for field in ("candidate_topk", "max_paths_per_pair", "path_budget"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} 必须为正整数")


@dataclass(frozen=True)
class _Path:
    nodes: tuple[int, ...]
    rels: tuple[int, ...]
    candidate_id: int
    candidate_score: float
    topic_id: int


_INCOMING_EDGE_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _as_list(values) -> list:
    return values.tolist() if hasattr(values, "tolist") else list(values)


def _top_candidates(sample, topk: int) -> list[tuple[int, float]]:
    """由最终实体分数稳定取得 Top-N 候选答案。"""
    scores: dict[int, float] = {}
    for entity_id, score in zip(_as_list(sample.e_score_indices), _as_list(sample.e_score_values)):
        entity_id = int(entity_id)
        scores[entity_id] = max(scores.get(entity_id, -float("inf")), float(score))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:topk]


def _incoming_edges(edge_source: KGEdgeSource) -> dict[int, list[tuple[int, int]]]:
    """缓存 ``tail -> (relation, head)`` 索引，避免反复展开高出度二跳邻域。"""
    cached = _INCOMING_EDGE_CACHE.get(edge_source)
    if cached is not None:
        return cached
    index: dict[int, list[tuple[int, int]]] = {}
    for head_id, relation_id, tail_id in edge_source.all_edges():
        index.setdefault(int(tail_id), []).append((int(relation_id), int(head_id)))
    _INCOMING_EDGE_CACHE[edge_source] = index
    return index


def _forward_shortest_paths_from_topic(
    edge_source: KGEdgeSource,
    topic_id: int,
    candidate_ids: set[int],
    *,
    max_hop: int,
    max_paths: int,
) -> dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """一次 BFS 枚举一个主题实体到多个候选答案的等长最短路径。"""
    if max_hop <= 0:
        return {}

    queue = deque([((topic_id,), ())])
    shortest_depth: dict[int, int] = {}
    found: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    while queue:
        nodes, rels = queue.popleft()
        depth = len(rels)
        if depth >= max_hop:
            continue
        for rel_id, tail_id in sorted(edge_source.neighbors(nodes[-1]), key=lambda edge: (edge[0], edge[1])):
            if tail_id in nodes:
                continue
            next_nodes = nodes + (int(tail_id),)
            next_rels = rels + (int(rel_id),)
            next_depth = depth + 1
            if tail_id in candidate_ids:
                candidate_depth = shortest_depth.get(tail_id)
                if candidate_depth is None:
                    shortest_depth[tail_id] = next_depth
                    candidate_depth = next_depth
                candidate_paths = found.setdefault(tail_id, [])
                if next_depth == candidate_depth and len(candidate_paths) < max_paths:
                    candidate_paths.append((next_nodes, next_rels))
            if next_depth < max_hop:
                queue.append((next_nodes, next_rels))
    return found


def _shortest_paths_from_topic(
    edge_source: KGEdgeSource,
    topic_id: int,
    candidate_ids: set[int],
    *,
    max_hop: int,
    max_paths: int,
) -> dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """以主题实体一跳邻域与候选终点入边相接，枚举 WebQSP 的一、二跳最短路径。"""
    if max_hop > 2:
        return _forward_shortest_paths_from_topic(
            edge_source, topic_id, candidate_ids, max_hop=max_hop, max_paths=max_paths,
        )
    if max_hop <= 0 or not candidate_ids:
        return {}

    found: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    one_hop: dict[int, list[int]] = {}
    for relation_id, tail_id in sorted(edge_source.neighbors(topic_id), key=lambda edge: (edge[0], edge[1])):
        relation_id, tail_id = int(relation_id), int(tail_id)
        if tail_id == topic_id:
            continue
        one_hop.setdefault(tail_id, []).append(relation_id)
        if tail_id in candidate_ids:
            found.setdefault(tail_id, []).append(((topic_id, tail_id), (relation_id,)))

    if max_hop == 1:
        return found
    incoming = _incoming_edges(edge_source)
    for candidate_id in sorted(candidate_ids):
        if candidate_id in found:
            continue
        candidate_paths: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for relation_id, head_id in sorted(incoming.get(candidate_id, []), key=lambda edge: (edge[0], edge[1])):
            if head_id == topic_id or head_id == candidate_id:
                continue
            for first_relation_id in one_hop.get(head_id, []):
                candidate_paths.append(
                    ((topic_id, head_id, candidate_id), (first_relation_id, relation_id))
                )
                if len(candidate_paths) >= max_paths:
                    break
            if len(candidate_paths) >= max_paths:
                break
        if candidate_paths:
            found[candidate_id] = candidate_paths
    return found


def retrieve_shortest_paths_one(
    sample,
    edge_source: KGEdgeSource,
    id2ent: dict[int, str],
    id2rel: dict[int, str],
    *,
    params: ShortestPathParams,
) -> RetrieveResult:
    """只以最终实体分数作为候选答案，构造有界最短路径。"""
    started = time.perf_counter()
    max_hop = len(sample.rel_probs)
    paths: list[_Path] = []
    candidates = _top_candidates(sample, params.candidate_topk)
    candidate_ids = {candidate_id for candidate_id, _ in candidates}
    for topic_id in sorted({int(topic) for topic in sample.topic_ids}):
        pair_paths = _shortest_paths_from_topic(
            edge_source,
            topic_id,
            candidate_ids - ({topic_id} if params.drop_loopback else set()),
            max_hop=max_hop,
            max_paths=params.max_paths_per_pair,
        )
        for candidate_id, candidate_score in candidates:
            for nodes, rels in pair_paths.get(candidate_id, []):
                paths.append(_Path(nodes, rels, candidate_id, candidate_score, topic_id))

    deduplicated = {(path.nodes, path.rels): path for path in paths}
    selected = sorted(
        deduplicated.values(),
        key=lambda path: (
            -path.candidate_score,
            len(path.rels),
            path.candidate_id,
            path.topic_id,
            path.rels,
            path.nodes,
        ),
    )[:params.path_budget]
    serialized = [
        {
            "path": path_to_triples(list(path.nodes), list(path.rels), id2ent, id2rel),
            "log_score": round(math.log(max(path.candidate_score, EPS)), 6),
        }
        for path in selected
    ]
    hop = int(sample.hop_attn.argmax().item()) + 1
    return RetrieveResult(
        question=sample.question,
        topics=[id2ent.get(int(topic), str(int(topic))) for topic in sample.topic_ids],
        hop=hop,
        paths=serialized,
        prediction=build_prediction(sample, id2ent),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        sample_index=getattr(sample, "sample_index", -1),
        golden=[id2ent.get(int(gold), str(int(gold))) for gold in getattr(sample, "gold_ids", [])],
    )
