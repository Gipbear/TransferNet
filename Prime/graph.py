"""加载 Prime 原生张量图并提供三跳可达性计算。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix


@dataclass(frozen=True, slots=True)
class PrimeGraph:
    triples: torch.Tensor
    adjacency: csr_matrix
    relation_ids: dict[str, int]
    entity_names: list[str]


def build_adjacency(edge_index: torch.Tensor, entity_count: int) -> csr_matrix:
    """把已含方向的 Prime 边张量构造成 CSR 邻接矩阵。"""
    subjects = edge_index[0].numpy()
    objects = edge_index[1].numpy()
    values = np.ones(subjects.shape[0], dtype=np.bool_)
    return csr_matrix((values, (subjects, objects)), shape=(entity_count, entity_count))


def load_graph(input_dir: str | Path) -> PrimeGraph:
    """加载转换后的 Prime 图。"""
    root = Path(input_dir)
    edge_index = torch.load(root / "graph/edge_index.pt", map_location="cpu", weights_only=True).long()
    edge_types = torch.load(root / "graph/edge_types.pt", map_location="cpu", weights_only=True).long()
    entity_names: list[str] = json.loads((root / "entity_names.json").read_text(encoding="utf-8"))
    relation_ids: dict[str, int] = json.loads(
        (root / "graph/edge_type_dict.json").read_text(encoding="utf-8")
    )
    triples = torch.stack((edge_index[0], edge_types, edge_index[1]), dim=1)
    return PrimeGraph(
        triples=triples,
        adjacency=build_adjacency(edge_index, len(entity_names)),
        relation_ids=relation_ids,
        entity_names=entity_names,
    )


def reachable_entities(adjacency: csr_matrix, topic_id: int, max_hop: int) -> np.ndarray:
    """返回主题实体在指定跳数内可达的全部实体 ID。"""
    seen = np.zeros(adjacency.shape[0], dtype=np.bool_)
    seen[topic_id] = True
    frontier = np.asarray([topic_id], dtype=np.int64)
    for _ in range(max_hop):
        neighbors = np.unique(adjacency[frontier].indices)
        frontier = neighbors[~seen[neighbors]]
        seen[frontier] = True
        if frontier.size == 0:
            break
    seen[topic_id] = False
    return np.flatnonzero(seen).astype(np.int32, copy=False)


def target_distances(
    adjacency: csr_matrix,
    topic_id: int,
    target_ids: set[int],
    max_hop: int,
) -> dict[int, int]:
    """计算一组目标实体从主题实体出发的最短跳数。"""
    distances: dict[int, int] = {}
    unresolved = set(target_ids)
    if topic_id in unresolved:
        distances[topic_id] = 0
        unresolved.remove(topic_id)
    seen = np.zeros(adjacency.shape[0], dtype=np.bool_)
    seen[topic_id] = True
    frontier = np.asarray([topic_id], dtype=np.int64)
    for hop in range(1, max_hop + 1):
        neighbors = np.unique(adjacency[frontier].indices)
        frontier = neighbors[~seen[neighbors]]
        seen[frontier] = True
        reached = unresolved.intersection(frontier.tolist())
        distances.update({entity_id: hop for entity_id in reached})
        unresolved.difference_update(reached)
        if not unresolved or frontier.size == 0:
            break
    return distances
