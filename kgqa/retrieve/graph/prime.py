"""为 Prime 原生张量图提供低内存的 CSR 邻接访问。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from kgqa.retrieve.graph.base import KGEdgeSource


class CSRNeighborMap:
    def __init__(self, offsets: np.ndarray, relations: np.ndarray, objects: np.ndarray):
        self.offsets = offsets
        self.relations = relations
        self.objects = objects

    def get(self, node_id: int, default=None) -> list[tuple[int, int]]:
        if node_id < 0 or node_id + 1 >= self.offsets.shape[0]:
            return [] if default is None else default
        start = int(self.offsets[node_id])
        end = int(self.offsets[node_id + 1])
        return list(zip(self.relations[start:end].tolist(), self.objects[start:end].tolist()))


class PrimeKG(KGEdgeSource):
    def __init__(self, input_dir: str):
        graph_dir = Path(input_dir) / "graph"
        edge_index = torch.load(graph_dir / "edge_index.pt", map_location="cpu", weights_only=True).numpy()
        edge_types = torch.load(graph_dir / "edge_types.pt", map_location="cpu", weights_only=True).numpy()
        order = np.argsort(edge_index[0], kind="stable")
        subjects = edge_index[0, order]
        entity_count = int(edge_index.max()) + 1
        counts = np.bincount(subjects, minlength=entity_count)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
        self.valid_edges_dict = CSRNeighborMap(offsets, edge_types[order], edge_index[1, order])

    def neighbors(self, node_id: int) -> list[tuple[int, int]]:
        return self.valid_edges_dict.get(node_id, [])

    def all_edges(self) -> Iterable[tuple[int, int, int]]:
        for subject in range(self.valid_edges_dict.offsets.shape[0] - 1):
            for relation, obj in self.neighbors(subject):
                yield subject, relation, obj
