"""多答案关系组展开工具：基于分类器门控与路径拓扑支撑展开同组实体，杜绝正则词表。"""

from collections import Counter
from typing import Any

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import KnowledgeSource, WorkflowTool
from kgqa.agent.workflow.models import AnswerResult, RuntimeQuestion, ToolOutcome
from kgqa.agent.workflow.templates import TemplateLibrary
from kgqa.core.answer_metrics import norm_entity


class RelationExpander(WorkflowTool):
    """根据拓扑支撑关系统计展开候选尾实体，由题型分类器与结构门控守护。"""

    def __init__(self, config: WorkflowConfig, library: TemplateLibrary | None,
                 kg: KnowledgeSource, names: dict[str, str] | None = None) -> None:
        self.config = config
        self.library = library
        self.kg = kg
        self.names = dict(names or {})

    def run(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome:
        gate_outcome = self._check_guard(question, initial)
        if gate_outcome is not None:
            return gate_outcome

        winning_seqs = self._find_winning_sequences(question, initial.answers)
        if not winning_seqs:
            return ToolOutcome(False, "no_supporting_relation_sequence")

        expanded = self._collect_expanded_answers(question, winning_seqs, initial.answers)
        if len(expanded) <= len(initial.answers):
            return ToolOutcome(False, "no_new_entities_expanded")

        stats = {
            "initial_answer_n": len(initial.answers),
            "expanded_answer_n": len(expanded),
            "winning_sequences": [list(seq) for seq in winning_seqs],
        }
        return ToolOutcome(True, "relation_expanded", answers=expanded, stats=stats)

    def _check_guard(self, question: RuntimeQuestion, initial: AnswerResult) -> ToolOutcome | None:
        if not self.config.expansion_enabled:
            return ToolOutcome(False, "expansion_disabled")
        if initial.is_rejection or not initial.format_ok or not initial.answers:
            return ToolOutcome(False, "initial_rejected_or_invalid")
        if len(initial.answers) < self.config.expansion_min_answers:
            return ToolOutcome(False, "answers_below_threshold")
        if self.library is not None:
            route = self.library.classify(question.question)
            if (route.kind in ("comparative", "superlative")
                    and route.confidence >= self.config.type_confidence):
                return ToolOutcome(False, f"guarded_by_classifier_{route.kind}")
        return None

    def _find_winning_sequences(self, question: RuntimeQuestion,
                                answers: tuple[str, ...]) -> list[tuple[str, ...]]:
        answer_keys = {norm_entity(ans) for ans in answers if norm_entity(ans)}
        candidate_mid_to_name = {
            m: cand["name"]
            for cand in question.candidates
            for m in cand.get("mids", [])
        }
        seq_counts: Counter[tuple[str, ...]] = Counter()

        for path_item in question.paths:
            path_steps = path_item.get("path", [])
            if not path_steps:
                continue
            tail_mid = path_item.get("tail_mid") or ""
            canonical = path_item.get("canonical_tail") or ""
            step_tail = path_steps[-1][2] if path_steps else ""
            tail_name = candidate_mid_to_name.get(tail_mid, "")

            cand_keys = {norm_entity(t) for t in (tail_mid, canonical, step_tail, tail_name) if t}
            if cand_keys & answer_keys:
                seq = tuple(step[1] for step in path_steps if len(step) >= 2)
                if seq:
                    seq_counts[seq] += 1

        top_k = self.config.expansion_top_groups
        return [seq for seq, _ in seq_counts.most_common(top_k)]

    def _collect_expanded_answers(self, question: RuntimeQuestion,
                                  winning_seqs: list[tuple[str, ...]],
                                  initial_answers: tuple[str, ...]) -> tuple[str, ...]:
        topic_keys = {norm_entity(t) for t in question.topic_mids if norm_entity(t)}
        expanded_answers = list(initial_answers)
        seen_keys = {norm_entity(ans) for ans in initial_answers if norm_entity(ans)}

        for path_item in question.paths:
            path_steps = path_item.get("path", [])
            if not path_steps:
                continue
            seq = tuple(step[1] for step in path_steps if len(step) >= 2)
            if seq not in winning_seqs:
                continue
            tail_mid = path_item.get("tail_mid") or ""
            tail_name = path_item.get("canonical_tail") or path_steps[-1][2]
            if norm_entity(tail_mid) in topic_keys or norm_entity(tail_name) in topic_keys:
                continue
            key = norm_entity(tail_name)
            if key and key not in seen_keys:
                seen_keys.add(key)
                expanded_answers.append(tail_name)

        return tuple(expanded_answers)
