"""展示页业务变换：检索路径→KG 子图、回放结果→轨迹面板数据。"""
from __future__ import annotations

from typing import Any

# 终版（gatev2）配置：检索三参数 + 校验后处理 run flags。
# 与 scripts/_sweep_ch5_thresholds.py 的 BASE 保持一致，改动需同步。
FINAL_RETRIEVAL = {"beam_size": 50, "lambda_val": 0.2, "alpha_final": 1.0}
FINAL_CHECK_FLAGS = dict(
    batch_size=20,
    expansion_min_answers=8,
    expansion_top_groups=3,
    score_margin=2.0,
    hop_filter=False,
    large_answer_expansion=True,
    enable_relation_expansion=True,
    drop_topic_self=True,
    mixed_stop_ratio=0.5,
    max_batches=2,
)


def _path_text(triples: list[list[str]]) -> str:
    if not triples:
        return ""
    parts = [triples[0][0]]
    for head, rel, tail in triples:
        parts.extend([rel, tail])
    return " -> ".join(parts)


def _group_relations(group_key: str) -> list[str]:
    return [part for part in str(group_key).split("|")[1:] if part]


def _group_source_label(group_key: str) -> str:
    rels = _group_relations(group_key)
    if not rels:
        return ""
    return " -> ".join(rel.split(".")[-1] for rel in rels)


def _name_triples(triples: list[list[str]], entity_map: dict[str, str]) -> list[list[str]]:
    return [
        [entity_map.get(str(head), str(head)), str(rel), entity_map.get(str(tail), str(tail))]
        for head, rel, tail in triples
    ]


def paths_to_graph(
    named_paths: list[dict[str, Any]],
    topics: list[str],
    extra_paths: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    paths_out: list[dict[str, Any]] = []

    def ensure_node(name: str, layer: int) -> None:
        node = nodes.get(name)
        if node is None:
            nodes[name] = {"id": name, "layer": layer}
        elif layer < node["layer"]:
            node["layer"] = layer

    for topic in topics:
        ensure_node(topic, 0)
    for pid, item in enumerate(named_paths, start=1):
        triples = item.get("path") or []
        paths_out.append({
            "id": pid,
            "score": item.get("log_score"),
            "tail": triples[-1][2] if triples else "",
            "triples": triples,
            "text": _path_text(triples),
        })
        for hop, (head, rel, tail) in enumerate(triples):
            ensure_node(head, hop)
            ensure_node(tail, hop + 1)
            edge = edges.setdefault(
                (head, rel, tail),
                {"source": head, "target": tail, "relation": rel, "path_ids": []},
            )
            edge["path_ids"].append(pid)
    for item in extra_paths or []:
        pid = item.get("id")
        triples = item.get("path") or []
        paths_out.append({
            "id": pid,
            "label": item.get("label"),
            "score": item.get("score"),
            "tail": item.get("tail") or (triples[-1][2] if triples else ""),
            "triples": triples,
            "text": item.get("text") or _path_text(triples),
            "synthetic": item.get("synthetic", False),
        })
        for hop, (head, rel, tail) in enumerate(triples):
            ensure_node(head, hop)
            ensure_node(tail, hop + 1)
            edge = edges.setdefault(
                (head, rel, tail),
                {"source": head, "target": tail, "relation": rel, "path_ids": []},
            )
            edge["path_ids"].append(pid)
    return {"nodes": list(nodes.values()), "edges": list(edges.values()), "paths": paths_out}


def shape_replay_result(
    result: dict[str, Any],
    entity_map: dict[str, str],
    kg_path_resolver: Any = None,
) -> dict[str, Any]:
    paths = result.get("named_mmr_reason_paths", [])

    def tail_of(pid: int) -> str:
        # 关系扩展可能引入超出检索列表的路径号,越界按"无尾实体"处理而非崩溃
        if pid < 1 or pid > len(paths):
            return ""
        triples = paths[pid - 1].get("path") or []
        return triples[-1][2] if triples else ""

    iterations: list[dict[str, Any]] = []
    batch_answer_names: list[str] = []
    for it in result.get("iterations", []):
        cited = list(it.get("global_cited_path_indices", []))
        accepted = list(it.get("accepted_path_indices", []))
        iterations.append({
            "batch_index": it.get("batch_index"),
            "batch_start_rank": it.get("batch_start_rank"),
            "batch_end_rank": it.get("batch_end_rank"),
            "batch_status": it.get("batch_status"),
            "answers": list(it.get("answer_names", [])),
            "cited_path_ids": cited,
            "accepted_path_ids": accepted,
            "rejected_path_ids": [pid for pid in cited if pid not in set(accepted)],
        })
        if it.get("batch_status") != "all_wrong":
            batch_answer_names.extend(it.get("answer_names", []))

    final_names = list(result.get("pred_answer_names", []))
    relation_pids = set(result.get("relation_expanded_path_indices", []))
    accepted_support = {
        pid: tail_of(pid) for pid in result.get("final_accepted_path_indices", [])
    }
    # 关系扩展路径通常被校验拒绝后按同关系序列收回,不在 final_accepted 中,
    # 需单独映射,否则扩展来源的答案会显示为"无引用"
    expansion_support = {pid: tail_of(pid) for pid in relation_pids}
    group_names = [
        entity_map.get(mid, mid)
        for mid in result.get("large_answer_expanded_mids", [])
    ]
    group_name_set = set(group_names)
    group_source_labels_by_mid: dict[str, list[str]] = {}
    group_source_keys_by_mid: dict[str, list[str]] = {}
    for group_key, tails in (result.get("group_tails", {}) or {}).items():
        label = _group_source_label(group_key)
        if not label:
            continue
        for mid in tails:
            keys = group_source_keys_by_mid.setdefault(str(mid), [])
            if group_key not in keys:
                keys.append(group_key)
            labels = group_source_labels_by_mid.setdefault(str(mid), [])
            if label not in labels:
                labels.append(label)
    group_items = []
    group_source_labels_by_name: dict[str, list[str]] = {}
    group_kg_path_ids_by_name: dict[str, list[str]] = {}
    kg_completion_paths = []
    topic_mid = str(result.get("topic_mid") or (result.get("raw_topics") or [""])[0])
    if not topic_mid:
        for group_key in (result.get("group_tails", {}) or {}):
            topic_mid = str(group_key).split("|", 1)[0]
            break
    topic_names = list(result.get("named_topics", []))
    topic_name = topic_names[0] if topic_names else entity_map.get(topic_mid, topic_mid)

    resolved_raw_paths: dict[tuple[str, str], list[list[str]]] = {}
    if kg_path_resolver is not None:
        mids_by_source_key: dict[str, list[str]] = {}
        for mid in result.get("large_answer_expanded_mids", []):
            source_keys = group_source_keys_by_mid.get(str(mid), [])
            if source_keys:
                mids_by_source_key.setdefault(source_keys[0], []).append(str(mid))
        for source_key, mids in mids_by_source_key.items():
            rels = _group_relations(source_key)
            try:
                if hasattr(kg_path_resolver, "resolve_many"):
                    restored = kg_path_resolver.resolve_many(topic_mid, rels, mids)
                else:
                    restored = {
                        mid: kg_path_resolver(topic_mid, rels, mid)
                        for mid in mids
                    }
            except Exception:
                restored = {}
            for mid, raw_path in (restored or {}).items():
                if raw_path:
                    resolved_raw_paths[(source_key, str(mid))] = raw_path

    for mid, name in zip(result.get("large_answer_expanded_mids", []), group_names):
        labels = group_source_labels_by_mid.get(str(mid), [])
        source_keys = group_source_keys_by_mid.get(str(mid), [])
        source_key = source_keys[0] if source_keys else ""
        source_label = _group_source_label(source_key) if source_key else ""
        kg_pid = f"kg{len(kg_completion_paths) + 1}"
        kg_label = f"P_kg{len(kg_completion_paths) + 1}"
        raw_kg_path = resolved_raw_paths.get((source_key, str(mid)), [])
        if raw_kg_path:
            kg_path = _name_triples(raw_kg_path, entity_map)
            restored = True
        else:
            relation = f"KG: {source_label}" if source_label else "KG: relation_completion"
            kg_path = [[topic_name, relation, name]]
            restored = False
        kg_completion_paths.append({
            "id": kg_pid,
            "label": kg_label,
            "path": kg_path,
            "raw_path": raw_kg_path,
            "tail": name,
            "text": f"{kg_label}: {_path_text(kg_path)}",
            "source_key": source_key,
            "source_label": source_label,
            "relations": _group_relations(source_key) if source_key else [],
            "restored": restored,
            "synthetic": True,
        })
        group_items.append({
            "name": name,
            "source_labels": labels,
            "kg_path_ids": [kg_pid],
        })
        by_name = group_source_labels_by_name.setdefault(name, [])
        for label in labels:
            if label not in by_name:
                by_name.append(label)
        group_kg_path_ids_by_name.setdefault(name, []).append(kg_pid)

    final_answers = []
    # 不同 MID 可能映射到同一实体名(如 Henry VII 的两个 Edward Tudor),
    # 展示层按名字去重,避免答案卡出现内容完全相同的重复行
    seen_names: set[str] = set()
    for name in final_names:
        if name in seen_names:
            continue
        seen_names.add(name)
        pids = sorted(pid for pid, tail in accepted_support.items() if tail == name)
        pid_set = set(pids)
        exp_pids = sorted(
            pid for pid, tail in expansion_support.items()
            if tail == name and pid not in pid_set
        )
        if pids:
            via = "llm"
        elif exp_pids:
            via = "relation_expansion"
        elif name in group_name_set:
            via = "group_expansion"
        else:
            via = "llm"
        final_answers.append({
            "name": name,
            "path_ids": pids,
            "expansion_path_ids": exp_pids,
            "kg_path_ids": group_kg_path_ids_by_name.get(name, []),
            "group_source_labels": group_source_labels_by_name.get(name, []),
            "via": via,
        })

    final_set = set(final_names)
    dropped = [n for n in dict.fromkeys(batch_answer_names) if n not in final_set]
    return {
        "iterations": iterations,
        "final_answers": final_answers,
        "calibration": {
            "dropped_answers": dropped,
            "relation_expanded_path_ids": sorted(relation_pids),
            "group_expanded_names": group_names,
            "group_expanded_items": group_items,
        },
        "kg_completion_paths": kg_completion_paths,
        "stop_reason": result.get("stop_reason", ""),
    }
