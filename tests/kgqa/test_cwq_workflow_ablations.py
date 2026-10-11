"""验证回放只关闭指定分支且不改动源轨迹。"""

import unittest

from kgqa.agent.workflow.ablations import WorkflowAblations
from tests.kgqa.test_cwq_workflow_scoring import record


class TestWorkflowAblations(unittest.TestCase):
    def test_disabled_branch_restores_initial_others_stay_identical(self):
        source = [record(0, ["A"], ["B"], "comparison"), record(1, [], ["C"], "rescue")]
        source[0]["route"]["kind"] = "comparative"
        result = WorkflowAblations.replay(source, "without_comparison")
        self.assertEqual(result[0]["final"]["answers"], ["A"])
        self.assertEqual(result[1], source[1])
        self.assertEqual(source[0]["final"]["answers"], ["B"])

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkflowAblations.replay([], "unknown")
