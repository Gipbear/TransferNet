"""兼容路径：实体映射工具已迁至 :mod:`kgqa.core.entity_map`。"""
from kgqa.core.entity_map import (
    apply_entity_map,
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
    get_all_path_entities,
    load_entity_map,
    map_entities,
)

__all__ = [
    "apply_entity_map", "build_reverse_entity_map", "expand_pred_answers_with_path_constraint",
    "get_all_path_entities", "load_entity_map", "map_entities",
]
