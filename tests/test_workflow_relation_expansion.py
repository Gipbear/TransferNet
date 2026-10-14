"""验证统一工作流中的多答案关系组展开工具（RelationExpander）及其门控与补全行为。"""

import unittest
from unittest.mock import MagicMock

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.interfaces import KnowledgeSource
from kgqa.agent.workflow.models import AnswerResult, RouteDecision, RuntimeQuestion, WorkflowRecord
from kgqa.agent.workflow.orchestrator import MasterQAWorkflow
from kgqa.agent.workflow.tools.expansion import RelationExpander


class DummyKnowledgeSource(KnowledgeSource):
    def __init__(self, group_tails: dict[str, list[str]] | None = None) -> None:
        self.group_tails = group_tails or {}

    def adjacency(self, mids: list[str], step: tuple) -> MagicMock:
        return MagicMock(status="ok")

    def attributes(self, mids: list[str], chain: tuple) -> MagicMock:
        return MagicMock(status="ok", rows=[])

    def labels(self, mids: list[str]) -> MagicMock:
        rows = [{"mid": m, "name": f"Name of {m}"} for m in mids]
        return MagicMock(status="ok", rows=rows)

    def intersect_topic_attribute(self, topics: tuple[str, ...], chain: tuple, value: str) -> MagicMock:
        return MagicMock(status="ok", rows=[])


class TestRelationExpander(unittest.TestCase):
    def setUp(self) -> None:
        self.config_disabled = WorkflowConfig(expansion_enabled=False)
        self.config_enabled = WorkflowConfig(
            expansion_enabled=True,
            expansion_min_answers=2,
            expansion_top_groups=1,
            type_confidence=0.5,
        )
        self.kg = DummyKnowledgeSource()

    def test_disabled_by_config(self) -> None:
        expander = RelationExpander(self.config_disabled, None, self.kg)
        question = RuntimeQuestion(0, "q0", "Who played?", ("m.topic",), (), ())
        initial = AnswerResult(("Alice", "Bob"), is_rejection=False, format_ok=True)
        outcome = expander.run(question, initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "expansion_disabled")

    def test_initial_rejection_skipped(self) -> None:
        expander = RelationExpander(self.config_enabled, None, self.kg)
        question = RuntimeQuestion(0, "q0", "Who played?", ("m.topic",), (), ())
        initial = AnswerResult((), is_rejection=True, format_ok=True)
        outcome = expander.run(question, initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "initial_rejected_or_invalid")

    def test_answers_below_threshold_skipped(self) -> None:
        expander = RelationExpander(self.config_enabled, None, self.kg)
        question = RuntimeQuestion(0, "q0", "Who played?", ("m.topic",), (), ())
        initial = AnswerResult(("Alice",), is_rejection=False, format_ok=True)
        outcome = expander.run(question, initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "answers_below_threshold")

    def test_classifier_single_answer_guard(self) -> None:
        library = MagicMock()
        library.classify.return_value = RouteDecision("comparative", 0.95)
        expander = RelationExpander(self.config_enabled, library, self.kg)
        question = RuntimeQuestion(0, "q0", "Who is taller?", ("m.topic",), (), ())
        initial = AnswerResult(("Alice", "Bob"), is_rejection=False, format_ok=True)
        outcome = expander.run(question, initial)
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "guarded_by_classifier_comparative")

    def test_expand_group_from_paths_and_drop_topic_self(self) -> None:
        library = MagicMock()
        library.classify.return_value = RouteDecision("composition", 0.9)
        paths = (
            {
                "path": [["m.topic", "film.film.starring", "Alice"]],
                "tail_mid": "m.alice",
                "canonical_tail": "Alice",
            },
            {
                "path": [["m.topic", "film.film.starring", "Bob"]],
                "tail_mid": "m.bob",
                "canonical_tail": "Bob",
            },
            {
                "path": [["m.topic", "film.film.starring", "Charlie"]],
                "tail_mid": "m.charlie",
                "canonical_tail": "Charlie",
            },
            {
                "path": [["m.topic", "film.film.starring", "Topic Self"]],
                "tail_mid": "m.topic",
                "canonical_tail": "Topic Self",
            },
        )
        candidates = (
            {"name": "Alice", "mids": ["m.alice"], "resolution": "resolved"},
            {"name": "Bob", "mids": ["m.bob"], "resolution": "resolved"},
            {"name": "Charlie", "mids": ["m.charlie"], "resolution": "resolved"},
            {"name": "Topic Self", "mids": ["m.topic"], "resolution": "resolved"},
        )
        question = RuntimeQuestion(0, "q0", "Who starred?", ("m.topic",), paths, candidates)
        initial = AnswerResult(("Alice", "Bob"), is_rejection=False, format_ok=True)
        expander = RelationExpander(self.config_enabled, library, self.kg)
        outcome = expander.run(question, initial)
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.reason, "relation_expanded")
        self.assertIn("Charlie", outcome.answers)
        self.assertNotIn("Topic Self", outcome.answers)
        self.assertEqual(len(outcome.answers), 3)

    def test_master_workflow_dispatches_expansion(self) -> None:
        answerer = MagicMock()
        answerer.answer.return_value = AnswerResult(("Alice", "Bob"), is_rejection=False, format_ok=True)
        router = MagicMock()
        router.classify.return_value = RouteDecision("composition", 0.95)
        paths = (
            {
                "path": [["m.topic", "film.film.starring", "Alice"]],
                "tail_mid": "m.alice",
                "canonical_tail": "Alice",
            },
            {
                "path": [["m.topic", "film.film.starring", "Bob"]],
                "tail_mid": "m.bob",
                "canonical_tail": "Bob",
            },
            {
                "path": [["m.topic", "film.film.starring", "Charlie"]],
                "tail_mid": "m.charlie",
                "canonical_tail": "Charlie",
            },
        )
        candidates = (
            {"name": "Alice", "mids": ["m.alice"], "resolution": "resolved"},
            {"name": "Bob", "mids": ["m.bob"], "resolution": "resolved"},
            {"name": "Charlie", "mids": ["m.charlie"], "resolution": "resolved"},
        )
        question = RuntimeQuestion(0, "q0", "Who starred?", ("m.topic",), paths, candidates)
        comparison, conjunction, rescue = MagicMock(), MagicMock(), MagicMock()
        expander = RelationExpander(self.config_enabled, None, self.kg)
        workflow = MasterQAWorkflow(
            self.config_enabled, answerer, router, comparison, conjunction, rescue, expander
        )
        record = workflow.answer(question)
        self.assertEqual(record.source, "expansion")
        self.assertEqual(record.reason, "relation_expanded")
        self.assertIn("Charlie", record.final.answers)
        self.assertEqual(len(record.final.answers), 3)


if __name__ == "__main__":
    unittest.main()

