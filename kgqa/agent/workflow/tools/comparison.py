"""根据训练预测的属性链与比较方向执行符号重排，集成路径排噪与初答优先对齐。"""

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.evidence import EvidenceFormatter
from kgqa.agent.workflow.interfaces import KnowledgeSource, WorkflowTool
from kgqa.agent.workflow.models import AnswerResult, RuntimeQuestion, ToolOutcome
from kgqa.agent.workflow.templates import TemplateLibrary


class PathFilter:
    """根据候选实体到达路径的关系与置信度剔除底层元数据泛化噪声实体。"""

    @staticmethod
    def extract_metrics(question: RuntimeQuestion) -> dict[str, dict]:
        paths_by_tail = {}
        for p in question.paths:
            tail = p.get("tail_mid")
            if tail:
                paths_by_tail.setdefault(tail, []).append(p)
        metrics = {}
        for cand in question.candidates:
            name_key = cand["name"].strip().casefold()
            mids = cand.get("mids", [])
            matched = [p for m in mids if m in paths_by_tail for p in paths_by_tail[m]]
            if not matched:
                metrics[name_key] = {"max_score": -999.0, "has_noise": False}
                continue
            max_sc = max(p.get("log_score", -999.0) for p in matched)
            has_noise = any(
                any("rdf-schema" in str(e[1]) or "valuenotation" in str(e[1])
                    for e in p.get("path", []) if len(e) >= 2)
                for p in matched
            )
            metrics[name_key] = {"max_score": max_sc, "has_noise": has_noise}
        return metrics

    @classmethod
    def filter_values(cls, values: list[tuple], question: RuntimeQuestion) -> list[tuple]:
        metrics = cls.extract_metrics(question)
        valid_scores = [
            m["max_score"] for m in metrics.values() if m["max_score"] > -900.0 and not m["has_noise"]
        ]
        if not valid_scores:
            return values
        top_score = max(valid_scores)
        filtered = []
        for item in values:
            name_key = item[0].strip().casefold()
            meta = metrics.get(name_key)
            if meta and meta["has_noise"] and (top_score - meta["max_score"] > 2.5):
                continue
            filtered.append(item)
        return filtered or values


class ComparisonSelector:
    """执行符号极值筛选、不等式过滤与初答优先重排。"""

    @staticmethod
    def select(values: list[tuple], plan: dict, initial: AnswerResult) -> list:
        operator = plan["operator"]
        if operator in ("argmin", "argmax"):
            extreme = (min if operator == "argmin" else max)(item[2] for item in values)
            extreme_items = [item for item in values if item[2] == extreme]
            if initial.answers:
                init_cand = {a.strip().casefold() for a in initial.answers if a}
                in_init = [item for item in extreme_items if item[0].strip().casefold() in init_cand]
                not_in_init = [item for item in extreme_items if item[0].strip().casefold() not in init_cand]
                return in_init + not_in_init
            return extreme_items
        threshold = EvidenceFormatter.comparable(plan.get("threshold") or "")
        if not threshold or threshold[0] != values[0][1]:
            return []
        comparisons = {
            "lt": lambda x: x < threshold[1], "le": lambda x: x <= threshold[1],
            "gt": lambda x: x > threshold[1], "ge": lambda x: x >= threshold[1],
        }
        cmp_fn = comparisons.get(operator)
        if not cmp_fn:
            return []
        selected = [item for item in values if cmp_fn(item[2])]
        return ComparisonSelector._reorder_with_initial(selected, initial)

    @staticmethod
    def _reorder_with_initial(selected: list[tuple], initial: AnswerResult) -> list:
        if initial.answers:
            init_set = {a.strip().casefold() for a in initial.answers if a}
            in_init = [item for item in selected if item[0].strip().casefold() in init_set]
            not_in_init = [item for item in selected if item[0].strip().casefold() not in init_set]
            selected = in_init + not_in_init
        unique = {}
        for item in selected:
            unique.setdefault(item[0].strip().casefold(), item)
        return list(unique.values())[:max(len(initial.answers), 1)]


class NumericReranker(WorkflowTool):
    def __init__(self, config: WorkflowConfig, library: TemplateLibrary, kg: KnowledgeSource) -> None:
        self.config, self.library, self.kg = config, library, kg

    def run(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        route = self.library.classify(question.question)
        plan = self.library.comparison_plan(question.question, route.kind)
        if not plan:
            return ToolOutcome(False, "no_plan")
        if plan.get("confidence", 1.0) < self.config.operator_confidence:
            return ToolOutcome(False, "low_operator_confidence")
        chains = plan.get("candidate_chains") or ([plan["chain"]] if "chain" in plan else [])
        if not chains:
            return ToolOutcome(False, "no_chains")
        if len(chains) == 1 and plan.get("margin", 0.0) < self.config.chain_margin:
            return ToolOutcome(False, "low_attribute_margin")
        chains = self._align_chains(question.question, chains)
        names = EvidenceFormatter.candidate_names(question)
        for chain in chains:
            outcome = self._try_chain(question, initial, plan, chain, names)
            if outcome.applied:
                return outcome
        return ToolOutcome(False, "no_comparable_selection")

    @staticmethod
    def _align_chains(question: str, chains: list[list[str]]) -> list[list[str]]:
        q_lower = question.lower()
        def score(chain: list[str]) -> int:
            text = " ".join(chain).lower()
            bonus = 0
            if ("movie" in q_lower or "film" in q_lower) and "tv" in text:
                bonus -= 20
            if "tv" in q_lower and ("film" in text or "movie" in text):
                bonus -= 20
            for kw in ("postgraduate", "undergraduate", "founded", "death", "birth",
                       "population", "elevation", "height", "release", "calling_code", "area", "armed_forces"):
                if kw in q_lower and kw in text:
                    bonus += 10
                elif kw in q_lower and kw not in text:
                    bonus -= 10
            return bonus
        filtered = [c for c in chains if score(c) >= 0]
        return sorted(filtered or chains, key=score, reverse=True)

    def _try_chain(self, question: RuntimeQuestion, initial: AnswerResult,
                   plan: dict, chain: list[str], names: dict[str, str]) -> ToolOutcome:
        result = self.kg.attributes(list(names), chain)
        if result.status != "ok":
            return ToolOutcome(False, "attribute_" + result.status)
        values = self._parse_values(result.rows, names, chain, question.question)
        if not values or len({v[1] for v in values}) != 1:
            return ToolOutcome(False, "missing_or_mixed_values")
        values = PathFilter.filter_values(values, question)
        selected = ComparisonSelector.select(values, plan, initial)
        if not selected:
            return ToolOutcome(False, "no_comparable_selection")
        answers = tuple(dict.fromkeys(item[0] for item in selected))
        evidence = tuple({**item[3], "chain": chain, "query_hash": result.query_hash} for item in selected)
        return ToolOutcome(True, "symbolic_selection", answers=answers, evidence=evidence,
                           stats={"plan": plan, "chain": chain, "query_ms": result.elapsed_ms,
                                  "cache_hit": result.cache_hit})

    @staticmethod
    def _parse_values(
        rows: tuple[dict, ...], names: dict[str, str], chain: list[str], question: str
    ) -> list[tuple]:
        values = []
        q_lower = question.lower()
        for row in rows:
            parsed = EvidenceFormatter.comparable(row.get("value", ""), chain)
            mid = row.get("mid", "")
            if parsed and mid in names:
                name = names[mid]
                if ("college" in q_lower or "university" in q_lower) and "high school" in name.lower():
                    continue
                values.append((name, parsed[0], parsed[1], row))
        return values

    @staticmethod
    def _select(values: list, plan: dict, initial: AnswerResult) -> list:
        return ComparisonSelector.select(values, plan, initial)
