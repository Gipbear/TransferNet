"""路径级评测：复用 utils.path_utils，增加 group_by 分组视图。"""
from __future__ import annotations

from utils.path_utils import compute_path_metrics, compute_path_diversity
from kgqa.types import MetricSpec, RetrieveResult


def _paths_as_tuples(result: RetrieveResult) -> list[tuple[list[str], list[str], float]]:
    tuples = []
    for p in result.paths:
        edges = p["path"]
        nodes = ([edges[0][0]] + [e[2] for e in edges]) if edges else []
        rels = [e[1] for e in edges]
        tuples.append((nodes, rels, p.get("log_score", 0.0)))
    return tuples


def path_record(result: RetrieveResult, gold: set[str], id2rel=None) -> dict:
    selected = _paths_as_tuples(result)
    if not selected:
        return {"hop": result.hop, "answer_hit": 0, "top1_hit": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "jaccard_diversity": 0.0, "relation_jaccard_diversity": 0.0,
                "tail_diversity": 0.0, "relation_coverage": 0.0, "edge_coverage": 0.0}
    m = compute_path_metrics(selected, gold, id2rel=id2rel)
    d = compute_path_diversity(selected)
    return {
        "hop": result.hop,
        "answer_hit": int(m["answer_hit"]), "top1_hit": int(m["top1_hit"]),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "jaccard_diversity": d.get("jaccard_diversity", 0.0),
        "relation_jaccard_diversity": d.get("relation_jaccard_diversity", 0.0),
        "tail_diversity": d.get("tail_diversity", 0.0),
        "relation_coverage": d.get("relation_coverage", 0.0),
        "edge_coverage": d.get("edge_coverage", 0.0),
    }


def _mean(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ["answer_hit", "top1_hit", "precision", "recall", "f1",
            "jaccard_diversity", "relation_jaccard_diversity", "tail_diversity",
            "relation_coverage", "edge_coverage"]
    n = len(records)
    return {"n": n, **{k: round(sum(float(r[k]) for r in records) / n, 4) for k in keys}}


def path_summary(results: list[RetrieveResult], gold_by_index: dict, spec: MetricSpec,
                 id2rel=None) -> dict:
    records = [path_record(r, gold_by_index.get(r.sample_index, set()), id2rel=id2rel)
               for r in results]
    overall = _mean(records)
    by_hop: dict[str, dict] = {}
    if spec.group_by == "hop":
        groups: dict[str, list[dict]] = {}
        for rec in records:
            if rec.get("hop") is None:
                continue
            groups.setdefault(str(rec["hop"]), []).append(rec)
        by_hop = {hop: _mean(recs) for hop, recs in sorted(groups.items())}
    return {"overall": overall, "by_hop": by_hop}
