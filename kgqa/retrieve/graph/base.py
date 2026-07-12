"""KG 边来源策略口（方案 C 发散点之一）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class KGEdgeSource(ABC):
    @abstractmethod
    def neighbors(self, node_id: int) -> list[tuple[int, int]]:
        """返回从 node_id 出发的 (rel_id, tail_id) 列表。"""

    @abstractmethod
    def all_edges(self) -> Iterable[tuple[int, int, int]]:
        """遍历全部 (subj_id, rel_id, obj_id)。"""
