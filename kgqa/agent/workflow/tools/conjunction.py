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
            if result.status != "ok":
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
        return ToolOutcome(False, "no_verified_numeric_match")
