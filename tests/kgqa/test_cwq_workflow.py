"""验证 CWQ 工作流的无标签输入、路由与回退契约。"""

import unittest
from unittest.mock import patch

import requests

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import AnswerResult, RouteDecision, RuntimeQuestion, ToolOutcome
from kgqa.agent.workflow.orchestrator import MasterQAWorkflow


class FakeAnswerer:
    def __init__(self, initial, followup=None):
        self.results = [initial, followup or initial]
        self.calls = []

    def answer(self, question, paths):
        self.calls.append((question, paths))
        return self.results[len(self.calls) - 1]


class FakeRouter:
    def __init__(self, kind):
        self.kind = kind

    def classify(self, question):
        return RouteDecision(self.kind, 1.0)


class FakeTool:
    def __init__(self, outcome=None):
        self.outcome = outcome or ToolOutcome(False, "unchanged")
        self.calls = []

    def run(self, question, initial):
        self.calls.append(question)
        return self.outcome


class TestRuntimeQuestion(unittest.TestCase):
    def test_gold_and_official_type_do_not_enter_runtime(self):
        source = {"sample_index": 7, "id": "q7", "question": "Q", "topic_mids": ["m.a"]}
        expected = RuntimeQuestion.from_record(source)
        changed = {**source, "golden": ["leak"], "sparql": "SELECT leak", "compositionality_type": "comparative"}
        self.assertEqual(expected, RuntimeQuestion.from_record(changed))
        self.assertNotIn("golden", expected.to_dict())

    def test_empty_paths_preserve_question(self):
        self.assertEqual(RuntimeQuestion.from_record({"sample_index": 0, "question": "Q"}).paths, ())

    def test_numeric_template_threshold_cannot_silently_override_fitted_library(self):
        with self.assertRaisesRegex(ValueError, "min_chain_score"):
            WorkflowConfig(min_chain_score=0.2)


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.question = RuntimeQuestion.from_record({"sample_index": 0, "question": "Q"})
        self.initial = AnswerResult(("original",), False, True)
        self.followup = AnswerResult(("rescued",), False, True)
        self.tool = FakeTool()

    def workflow(self, kind, initial=None, config=None):
        answerer = FakeAnswerer(initial or self.initial, self.followup)
        workflow = MasterQAWorkflow(
            config or WorkflowConfig(), answerer, FakeRouter(kind), self.tool, self.tool, self.tool
        )
        return workflow, answerer

    def workflow_with_tools(self, kind, comparison, conjunction, rescue, initial=None, config=None):
        answerer = FakeAnswerer(initial or self.initial, self.followup)
        workflow = MasterQAWorkflow(
            config or WorkflowConfig(), answerer, FakeRouter(kind), comparison, conjunction, rescue
        )
        return workflow, answerer

    def test_non_rejected_composition_never_calls_kg(self):
        workflow, answerer = self.workflow("composition")
        record = workflow.answer(self.question)
        self.assertEqual(record.final, self.initial)
        self.assertEqual(len(answerer.calls), 1)
        self.assertFalse(self.tool.calls)

    def test_rejected_composition_uses_new_paths_and_same_answerer(self):
        rejected = AnswerResult((), True, True)
        self.tool.outcome = ToolOutcome(True, "new_evidence", paths=({"path": [["a", "r", "rescued"]]},))
        workflow, answerer = self.workflow("composition", rejected)
        record = workflow.answer(self.question)
        self.assertEqual(record.final.answers, ("rescued",))
        self.assertEqual(len(answerer.calls), 2)

    def test_unmatched_conjunction_keeps_exact_initial_result(self):
        workflow, answerer = self.workflow("conjunction")
        self.assertEqual(workflow.answer(self.question).final, self.initial)
        self.assertEqual(len(answerer.calls), 1)

    def test_numeric_conjunction_calls_pfit_once_more(self):
        self.tool.outcome = ToolOutcome(True, "numeric_match", paths=({"path": [["a", "r", "b (rate: 1.8)"]]},))
        workflow, answerer = self.workflow("conjunction")
        self.assertEqual(workflow.answer(self.question).final, self.followup)
        self.assertEqual(len(answerer.calls), 2)

    def test_comparison_direct_answer_does_not_call_second_pfit(self):
        self.tool.outcome = ToolOutcome(True, "symbolic_answer", answers=("tool_answer",))
        workflow, answerer = self.workflow("superlative")
        record = workflow.answer(self.question)
        self.assertEqual(record.final.answers, ("tool_answer",))
        self.assertEqual(len(answerer.calls), 1)

    def test_budget_exhaustion_keeps_initial(self):
        self.tool.outcome = ToolOutcome(True, "numeric_match", paths=({"path": [["a", "r", "b"]]},))
        workflow, answerer = self.workflow("conjunction", config=WorkflowConfig(max_pfit_calls=1))
        self.assertEqual(workflow.answer(self.question).final, self.initial)
        self.assertEqual(len(answerer.calls), 1)

    def test_failed_followup_keeps_initial(self):
        self.tool.outcome = ToolOutcome(True, "numeric_match", paths=({"path": [["a", "r", "b"]]},))
        workflow, answerer = self.workflow("conjunction")
        answerer.results[1] = AnswerResult((), False, False)
        self.assertEqual(workflow.answer(self.question).final, self.initial)

    def test_invalid_initial_format_is_not_treated_as_rejection(self):
        workflow, _answerer = self.workflow("composition", AnswerResult((), False, False))
        self.assertFalse(workflow.answer(self.question).final.format_ok)
        self.assertFalse(self.tool.calls)

    def test_only_actually_rendered_paths_are_recorded(self):
        path = {"path": [["a", "r", "b"]]}
        self.question = RuntimeQuestion(0, "q", "Q", paths=(path, path))
        self.initial = AnswerResult(("original",), False, True, used_path_count=1)
        workflow, _answerer = self.workflow("composition")
        record = workflow.answer(self.question)
        self.assertEqual(len(record.initial_paths), 1)
        self.assertEqual(len(record.final_paths), 1)

    def test_failed_followup_remains_in_call_trace(self):
        self.tool.outcome = ToolOutcome(True, "numeric_match", paths=({"path": [["a", "r", "b"]]},))
        workflow, answerer = self.workflow("conjunction")
        failure = AnswerResult((), False, False, prompt_tokens=99, completion_tokens=13)
        answerer.results[1] = failure
        record = workflow.answer(self.question)
        self.assertEqual(record.final, self.initial)
        self.assertEqual(record.call_results, (self.initial, failure))

    def test_followup_transport_failure_keeps_initial_and_records_attempt(self):
        self.tool.outcome = ToolOutcome(True, "numeric_match", paths=({"path": [["a", "r", "b"]]},))
        workflow, answerer = self.workflow("conjunction")
        with patch.object(answerer, "answer", side_effect=[self.initial, requests.Timeout("offline")]):
            record = workflow.answer(self.question)
        self.assertEqual(record.final, self.initial)
        self.assertEqual(record.reason, "followup_transport_error")
        self.assertEqual(record.pfit_calls, 2)
        self.assertIn("Timeout", record.call_results[-1].generation_error)

    def test_tool_transport_failure_keeps_initial(self):
        workflow, _answerer = self.workflow("superlative")
        with patch.object(self.tool, "run", side_effect=requests.ConnectionError("offline")):
            record = workflow.answer(self.question)
        self.assertEqual(record.final, self.initial)
        self.assertEqual(record.reason, "tool_transport_error")

    def test_rejected_unmatched_conjunction_cascades_to_rescue(self):
        rejected = AnswerResult((), True, True)
        conjunction = FakeTool(ToolOutcome(False, "numeric_unmatched"))
        rescue = FakeTool(ToolOutcome(True, "rescued_paths", paths=({"path": [["a", "r", "rescued"]]},)))
        workflow, answerer = self.workflow_with_tools("conjunction", self.tool, conjunction, rescue, rejected)
        record = workflow.answer(self.question)
        self.assertEqual(record.final.answers, ("rescued",))
        self.assertEqual(record.source, "rescue")
        self.assertEqual(len(conjunction.calls), 1)
        self.assertEqual(len(rescue.calls), 1)
        self.assertEqual(len(answerer.calls), 2)

    def test_non_rejected_unmatched_conjunction_never_cascades_to_rescue(self):
        conjunction = FakeTool(ToolOutcome(False, "numeric_unmatched"))
        rescue = FakeTool(ToolOutcome(True, "rescued_paths", paths=({"path": [["a", "r", "rescued"]]},)))
        workflow, answerer = self.workflow_with_tools("conjunction", self.tool, conjunction, rescue, self.initial)
        record = workflow.answer(self.question)
        self.assertEqual(record.final, self.initial)
        self.assertEqual(record.source, "baseline")
        self.assertEqual(len(conjunction.calls), 1)
        self.assertEqual(len(rescue.calls), 0)
        self.assertEqual(len(answerer.calls), 1)

    def test_rejected_unmatched_comparison_cascades_to_rescue(self):
        rejected = AnswerResult((), True, True)
        comparison = FakeTool(ToolOutcome(False, "unsupported_comparison"))
        rescue = FakeTool(ToolOutcome(True, "rescued_paths", paths=({"path": [["a", "r", "rescued"]]},)))
        workflow, answerer = self.workflow_with_tools("comparative", comparison, self.tool, rescue, rejected)
        record = workflow.answer(self.question)
        self.assertEqual(record.final.answers, ("rescued",))
        self.assertEqual(record.source, "rescue")
        self.assertEqual(len(comparison.calls), 1)
        self.assertEqual(len(rescue.calls), 1)

    def test_disabled_cascade_rescue_keeps_rejected_when_unmatched(self):
        rejected = AnswerResult((), True, True)
        conjunction = FakeTool(ToolOutcome(False, "numeric_unmatched"))
        rescue = FakeTool(ToolOutcome(True, "rescued_paths", paths=({"path": [["a", "r", "rescued"]]},)))
        cfg = WorkflowConfig(allow_cascade_rescue=False)
        workflow, answerer = self.workflow_with_tools("conjunction", self.tool, conjunction, rescue, rejected, cfg)
        record = workflow.answer(self.question)
        self.assertEqual(record.final, rejected)
        self.assertEqual(len(rescue.calls), 0)


if __name__ == "__main__":
    unittest.main()
