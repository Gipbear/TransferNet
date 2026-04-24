"""Entity MID/name mapping helpers."""

from __future__ import annotations

from typing import Iterable


def load_entity_map(path: str) -> dict[str, str]:
    """Load a tab-separated MID -> name file."""
    entity_map: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if "\t" not in line:
                continue
            mid, name = line.split("\t", 1)
            entity_map[mid] = name
    return entity_map


def apply_entity_map(path_edges: list[list[str]], entity_map: dict[str, str]) -> list[list[str]]:
    """Map head/tail entities in a path from MID to display name."""
    return [
        [entity_map.get(edge[0], edge[0]), edge[1], entity_map.get(edge[2], edge[2])]
        for edge in path_edges
    ]


def map_entities(entities: Iterable[str], entity_map: dict[str, str]) -> list[str]:
    """Map a sequence of entity ids to display names."""
    return [entity_map.get(entity, entity) for entity in entities]


def build_reverse_entity_map(entity_map: dict[str, str]) -> dict[str, set[str]]:
    """Build name -> set[MID] lookup for answer disambiguation."""
    reverse_map: dict[str, set[str]] = {}
    for mid, name in entity_map.items():
        reverse_map.setdefault(name.lower().strip(), set()).add(mid)
    return reverse_map


def get_all_path_entities(mmr_paths: list[dict]) -> set[str]:
    """Collect all head/tail entities that appear in the given paths."""
    entities: set[str] = set()
    for path_dict in mmr_paths:
        for head, _relation, tail in path_dict.get("path", []):
            entities.add(head.lower().strip())
            entities.add(tail.lower().strip())
    return entities


def expand_pred_answers_with_path_constraint(
    pred_answers: list[str],
    rev_entity_map: dict[str, set[str]] | None,
    path_mid_entities: set[str] | None,
) -> tuple[list[str], list[str]]:
    """Expand predicted names to MIDs and prefer candidates present in the paths."""
    expanded_pred: list[str] = []
    constrained_pred: list[str] = []
    path_mid_entities = {entity.lower().strip() for entity in (path_mid_entities or set())}

    for answer in pred_answers:
        key = answer.lower().strip()
        if rev_entity_map and key in rev_entity_map:
            expanded = sorted(rev_entity_map[key])
            constrained = [
                mid for mid in expanded if mid.lower().strip() in path_mid_entities
            ]
            expanded_pred.extend(expanded)
            constrained_pred.extend(constrained if constrained else expanded)
        else:
            expanded_pred.append(answer)
            constrained_pred.append(answer)

    return expanded_pred, constrained_pred
