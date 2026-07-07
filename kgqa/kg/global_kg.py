"""全局邻接表（WebQSP / MetaQA 共用）。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.path_utils import build_valid_edges_dict
from kgqa.kg.base import KGEdgeSource


class GlobalKG(KGEdgeSource):
    def __init__(self, valid_edges_dict: dict[int, list[tuple[int, int]]]):
        self.valid_edges_dict = valid_edges_dict

    @classmethod
    def from_triples(cls, triples: list[list[int]]) -> "GlobalKG":
        return cls(build_valid_edges_dict(triples))

    @classmethod
    def from_input_dir(cls, input_dir: str) -> "GlobalKG":
        """从 fbwq_full/{entities.dict,relations.dict,train.txt} 重建（含 _reverse 边）。

        逻辑迁移自 scripts.offline_path_search.rebuild_valid_edges_dict，逐字保留。"""
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
        return cls.from_triples(triples)

    def neighbors(self, node_id: int) -> list[tuple[int, int]]:
        return self.valid_edges_dict.get(node_id, [])

    def all_edges(self) -> Iterable[tuple[int, int, int]]:
        for subj, edges in self.valid_edges_dict.items():
            for rel, obj in edges:
                yield (subj, rel, obj)
