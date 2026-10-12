"""使用训练关系模板补全完整路径，再以冻结评分器执行 RSDPR。"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.evidence import EvidenceFormatter
from kgqa.agent.workflow.interfaces import GraphScorer, KnowledgeSource, WorkflowTool
from kgqa.agent.workflow.models import AnswerResult, RuntimeQuestion, ToolOutcome
from kgqa.agent.workflow.templates import TemplateLibrary
from kgqa.retrieve.engine import PathCandidate, select_path_candidates


@dataclass
class RescueSearchResult:
    paths: list[dict] = field(default_factory=list)
    edges: set[tuple[str, str, str]] = field(default_factory=set)
    query_statuses: list[str] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        return {"edge_n": len(self.edges), "adjacency_query_n": len(self.query_statuses),
                "complete_queries": self.query_statuses.count("ok"),
                "query_status_counts": dict(Counter(self.query_statuses)),
                "partial": bool(self.stop_reasons), "stop_reasons": list(dict.fromkeys(self.stop_reasons))}


class GraphRescuer(WorkflowTool):
    def __init__(self, config: WorkflowConfig, library: TemplateLibrary, kg: KnowledgeSource,
                 scorer: GraphScorer | None = None, names: dict | None = None) -> None:
        self.config, self.library, self.kg, self.scorer = config, library, kg, scorer
        self.names = {}
        self._remember_names(names or {})

    def run(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        if not initial.is_rejection:
            return ToolOutcome(False, "not_rejected")
        self._remember_names(EvidenceFormatter.candidate_names(question))
        patterns = self.library.extract_patterns(
            question.question, self.config.neighbor_k, self.config.patterns_per_band
        )
        graph = self._expand(question.topic_mids, patterns)
        stats = graph.stats()
        if graph.stop_reasons and not self.config.allow_partial_rescue:
            return ToolOutcome(False, graph.stop_reasons[0], stats=stats)
        if not graph.paths:
            reason = graph.stop_reasons[-1] if graph.stop_reasons else "no_complete_paths"
            return ToolOutcome(False, reason, stats=stats)
        paths, scored = self._score_paths(question, graph)
        return self._finish(paths, {**scored, **stats})

    def _expand(self, topics: tuple[str, ...], patterns: list[dict]) -> RescueSearchResult:
        graph = RescueSearchResult()
        for topic in topics[:self.config.max_topics]:
            for pattern in patterns:
                remaining = self.config.max_queries - len(graph.query_statuses)
                if remaining <= 0:
                    graph.stop_reasons.append("query_budget_exhausted")
                    return graph
                result = self._execute(topic, pattern, remaining)
                graph.query_statuses.extend(result.query_statuses)
                graph.stop_reasons.extend(result.stop_reasons)
                if result.stop_reasons and not self.config.allow_partial_rescue:
                    return graph
                for path in result.paths:
                    edges = self._edges(path["nodes"], path["relations"])
                    new_edges = set(map(tuple, edges))
                    if len(graph.edges | new_edges) > self.config.triple_budget:
                        graph.stop_reasons.append("triple_budget_truncated")
                        return graph
                    graph.edges.update(new_edges)
                    graph.paths.append(path)
        return graph

    def _score_paths(self, question: RuntimeQuestion, graph: RescueSearchResult) -> tuple[list[dict], dict]:
        if self.config.rescue_scorer == "transfernet":
            if self.scorer is None:
                raise ValueError("transfernet 模式要求新图评分服务")
            scored = self.scorer.score(question.question, question.topic_mids, sorted(graph.edges), self.config)
            scored.setdefault("score_source", "transfernet")
            paths = scored.get("paths", [])
            scored["gnn_path_n"] = len(paths)
            if not paths and self.config.allow_template_fallback:
                paths = self._template_select(graph.paths)
                scored["score_source"] = "template_fallback"
        else:
            paths, scored = self._template_select(graph.paths), {"score_source": "template"}
        return paths, scored

    def _execute(self, topic: str, pattern: dict, remaining_queries: int) -> RescueSearchResult:
        paths, result = [[topic]], RescueSearchResult()
        for step in pattern["steps"]:
            if len(result.query_statuses) >= remaining_queries:
                result.stop_reasons.append("query_budget_exhausted")
                return result
            event = self.kg.adjacency([path[-1] for path in paths], step)
            result.query_statuses.append(event.status)
            if event.status != "ok":
                if event.status != "empty":
                    result.stop_reasons.append("query_" + event.status)
                return result
            adjacent = defaultdict(list)
            for row in event.rows:
                adjacent[row["source"]].append(row["target"])
            following = []
            for path in paths:
                for tail in adjacent[path[-1]]:
                    if tail not in path:
                        following.append([*path, tail])
            if len(following) > self.config.step_limit:
                result.stop_reasons.append("step_limit_truncated")
                if not self.config.allow_partial_rescue:
                    return result
                following = sorted(following)[:self.config.step_limit]
            paths = sorted(following)
            if not paths:
                return result
        relations = [step["predicate"] + ("_rev" if step["reverse"] else "") for step in pattern["steps"]]
        result.paths = [{"nodes": path, "relations": relations, "score": pattern["score"]} for path in paths]
        return result

    def _template_select(self, records: list[dict]) -> list[dict]:
        if self.config.max_paths_per_relation > 0:
            chain_counts = defaultdict(int)
            filtered = []
            for r in records:
                chain = tuple(r["relations"])
                if chain_counts[chain] < self.config.max_paths_per_relation:
                    chain_counts[chain] += 1
                    filtered.append(r)
            records = filtered or records
        node_ids, rel_ids, candidates = {}, {}, []
        for order, record in enumerate(records):
            nodes, relations = [], []
            for node in record["nodes"]:
                nodes.append(node_ids.setdefault(node, len(node_ids)))
            for relation in record["relations"]:
                relations.append(rel_ids.setdefault(relation, len(rel_ids)))
            candidates.append(PathCandidate(nodes, relations, len(relations),
                                            math.log(max(record["score"], 1e-6)), order=order))
        depth = self.config.rescue_max_depth if self.config.rescue_sliding_enabled else 20
        selected = select_path_candidates(candidates, depth, eta=0, lambda_val=self.config.lambda_val,
                                          penalty_mode=self.config.penalty_mode)
        return [{"path": self._edges(records[item.order]["nodes"], records[item.order]["relations"]),
                 "log_score": item.score} for item in selected]

    def _finish(self, paths: list[dict], scored: dict) -> ToolOutcome:
        stats = {key: value for key, value in scored.items() if key != "paths"}
        if not paths:
            return ToolOutcome(False, "no_scored_paths", stats=stats)
        stats.update(self._resolve_names(paths))
        formatted = self._format_paths(paths)
        stats.update({"path_n": len(formatted), "unresolved_tail_n": len(paths) - len(formatted)})
        if not formatted:
            return ToolOutcome(False, "unresolved_answer_names", stats=stats)
        evidence = tuple({"mid_path": item["mid_path"], "tail_mid": item["tail_mid"]} for item in formatted)
        return ToolOutcome(True, "external_paths", paths=tuple(formatted), evidence=evidence, stats=stats)

    def _resolve_names(self, paths: list[dict]) -> dict:
        mids = sorted({node for item in paths for h, _, t in item["path"] for node in (h, t)})
        missing = [mid for mid in mids if mid not in self.names]
        if not missing:
            return {"label_status": "not_needed", "label_query_n": 0}
        result = self.kg.labels(missing)
        if result.status == "ok":
            self._remember_names({row["mid"]: row["name"] for row in result.rows})
        return {"label_status": result.status, "label_query_n": 1, "label_error": result.error}

    def _format_paths(self, paths: list[dict]) -> list[dict]:
        formatted = []
        for item in paths:
            named = [[self.names.get(h, h), relation, self.names.get(t, t)] for h, relation, t in item["path"]]
            if not named or not self._valid_name(named[-1][2]):
                continue
            formatted.append({"path": named, "mid_path": item["path"], "tail_mid": item["path"][-1][2],
                              "log_score": item["log_score"]})
        return formatted

    def _remember_names(self, names: dict[str, str]) -> None:
        for mid, name in names.items():
            if self._valid_name(name):
                self.names.setdefault(mid, name.strip())

    @staticmethod
    def _valid_name(name: str) -> bool:
        if not isinstance(name, str) or not name.strip():
            return False
        return re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", name.strip()) is None

    @staticmethod
    def _edges(nodes: list[str], relations: list[str]) -> list[list[str]]:
        return [[nodes[i], relation, nodes[i + 1]] for i, relation in enumerate(relations)]
