"""KG path restoration helpers for relation-completion demo paths."""
from __future__ import annotations

from pathlib import Path


class KGPathResolver:
    """Restore concrete MID paths from a topic, relation sequence, and tails.

    The path server's ``group_tails`` records only ``topic|rel1|...`` and final
    tail MIDs. For the demo graph we need the intermediate KG entities too, so
    this resolver scans the WebQSP KG triples selectively and reconstructs one
    valid path per requested tail.
    """

    def __init__(self, fb_dir: str | Path) -> None:
        self.fb_dir = Path(fb_dir)
        self.train_path = self.fb_dir / "train.txt"
        self.relations = self._load_relations(self.fb_dir / "relations.dict")
        self._cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, list[list[str]]]] = {}

    @staticmethod
    def _load_relations(path: Path) -> set[str]:
        rels: set[str] = set()
        if not path.is_file():
            return rels
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0]:
                    rels.add(parts[0].strip())
        return rels

    def available(self) -> bool:
        return self.train_path.is_file()

    def resolve_many(
        self,
        topic_mid: str,
        relations: list[str],
        tail_mids: list[str],
    ) -> dict[str, list[list[str]]]:
        rels = tuple(relations)
        targets = tuple(sorted({str(mid) for mid in tail_mids if mid}))
        key = (str(topic_mid), rels, targets)
        if key not in self._cache:
            self._cache[key] = self._resolve_many_uncached(str(topic_mid), list(rels), set(targets))
        return dict(self._cache[key])

    def _resolve_many_uncached(
        self,
        topic_mid: str,
        relations: list[str],
        target_mids: set[str],
    ) -> dict[str, list[list[str]]]:
        if not topic_mid or not relations or not target_mids or not self.available():
            return {}

        frontier = {topic_mid}
        parents_by_hop: list[tuple[str, dict[str, str]]] = []
        for hop, rel in enumerate(relations):
            tail_filter = target_mids if hop == len(relations) - 1 else None
            edges = self._scan_edges(frontier, rel, tail_filter=tail_filter)
            parents: dict[str, str] = {}
            for head, tail in edges:
                parents.setdefault(tail, head)
            if not parents:
                return {}
            parents_by_hop.append((rel, parents))
            frontier = set(parents)

        restored: dict[str, list[list[str]]] = {}
        for target in sorted(target_mids & frontier):
            nodes = self._reconstruct_nodes(target, parents_by_hop)
            if not nodes:
                continue
            restored[target] = [
                [nodes[i], relations[i], nodes[i + 1]]
                for i in range(len(relations))
            ]
        return restored

    def _scan_edges(
        self,
        heads: set[str],
        rel: str,
        *,
        tail_filter: set[str] | None = None,
    ) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        reverse_base = rel.removesuffix("_reverse") if rel.endswith("_reverse") else ""
        allow_reverse = bool(reverse_base) and rel in self.relations

        with self.train_path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                subj, edge_rel, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if edge_rel == rel and subj in heads:
                    if tail_filter is None or obj in tail_filter:
                        edges.append((subj, obj))
                if allow_reverse and edge_rel == reverse_base and obj in heads:
                    if tail_filter is None or subj in tail_filter:
                        edges.append((obj, subj))
        return edges

    @staticmethod
    def _reconstruct_nodes(
        target_mid: str,
        parents_by_hop: list[tuple[str, dict[str, str]]],
    ) -> list[str]:
        nodes = [target_mid]
        cur = target_mid
        for _rel, parents in reversed(parents_by_hop):
            parent = parents.get(cur)
            if not parent:
                return []
            nodes.append(parent)
            cur = parent
        nodes.reverse()
        return nodes
