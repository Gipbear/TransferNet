"""严格按历史数值匹配与一至三个候选规则富化路径。"""

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.evidence import EvidenceFormatter
from kgqa.agent.workflow.interfaces import KnowledgeSource, WorkflowTool
from kgqa.agent.workflow.models import AnswerResult, RuntimeQuestion, ToolOutcome
from kgqa.agent.workflow.templates import TemplateLibrary


class AttributeTagger(WorkflowTool):
    def __init__(self, config: WorkflowConfig, library: TemplateLibrary, kg: KnowledgeSource) -> None:
        self.config, self.library, self.kg = config, library, kg

    def run(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        numbers = EvidenceFormatter.numbers(question.question)
        names = EvidenceFormatter.candidate_names(question)
        if not numbers or not names:
            return ToolOutcome(False, "no_numeric_constraint_or_candidates")
        for chain in self.library.numeric_chains(question.question):
            result = self.kg.attributes(list(names), chain)
            if result.status not in ("ok", "truncated"):
                continue
            for number in numbers:
                target = EvidenceFormatter.comparable(number)
                matched = {row["mid"] for row in result.rows
                           if row["mid"] in names and EvidenceFormatter.comparable(row["value"]) == target}
                if not 0 < len(matched) <= self.config.matched_candidate_limit:
                    continue
                selected_names = {names[mid] for mid in matched}
                paths = EvidenceFormatter.annotate(question.paths, selected_names, chain, number)
                if paths == question.paths:
                    continue
                evidence = tuple({"mid": mid, "name": names[mid], "chain": list(chain), "value": number,
                                  "query_hash": result.query_hash} for mid in sorted(matched))
                return ToolOutcome(True, "numeric_match", paths=paths, evidence=evidence,
                                   stats={"matched_n": len(matched), "query_ms": result.elapsed_ms})
        if question.topic_mids:
            rescue_outcome = self._rescue_from_topics(question, numbers)
            if rescue_outcome.applied:
                return rescue_outcome
        return ToolOutcome(False, "no_verified_numeric_match")

    def _rescue_from_topics(self, question: RuntimeQuestion, numbers: list[str]) -> ToolOutcome:
        for chain in self.library.numeric_chains(question.question):
            for number in numbers:
                res = self.kg.intersect_topic_attribute(question.topic_mids, chain, number)
                if res.status not in ("ok", "truncated") or not res.rows:
                    continue
                matched_mids = [row["cand"] for row in res.rows if "cand" in row]
                if not 0 < len(matched_mids) <= self.config.matched_candidate_limit:
                    continue
                labels_res = self.kg.labels(matched_mids)
                label_map = {r["mid"]: r["name"] for r in labels_res.rows if "mid" in r and "name" in r}
                matched_names = {label_map.get(m, m) for m in matched_mids}
                label = chain[-1].split(".")[-1].replace("_", " ")
                synth_paths = list(question.paths)
                for m in matched_mids:
                    name = label_map.get(m, m)
                    display = f"{name} ({label}: {number})"
                    synth_paths.append({
                        "path": [[question.topic_mids[0], "rescued_attribute", display]],
                        "canonical_tail": name, "display_to_canonical": {display: name},
                        "log_score": 0.0, "tail_mid": m
                    })
                evidence = tuple({"mid": m, "name": label_map.get(m, m), "chain": list(chain), "value": number,
                                  "query_hash": res.query_hash, "mode": "topic_reverse"}
                                 for m in sorted(matched_mids))
                return ToolOutcome(True, "topic_numeric_match", paths=tuple(synth_paths),
                                   answers=tuple(sorted(matched_names)), evidence=evidence,
                                   stats={"matched_n": len(matched_mids), "query_ms": res.elapsed_ms})
        return ToolOutcome(False, "no_topic_numeric_match")

