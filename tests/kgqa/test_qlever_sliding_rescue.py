"""测试 QLever 在线拓扑救援的深度候选选择与多批滑窗救援机制。"""

import unittest
from unittest.mock import MagicMock

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import (
    AnswerResult,
    RouteDecision,
    RuntimeQuestion,
    ToolOutcome,
)
from kgqa.agent.workflow.orchestrator import MasterQAWorkflow
from kgqa.agent.workflow.tools.rescue import GraphRescuer


class TestQLeverSlidingRescue(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WorkflowConfig(
            max_pfit_calls=6,
            rescue_enabled=True,
            rescue_sliding_enabled=True,
            rescue_window_size=20,
            rescue_max_depth=100,
            rescue_max_windows=5,
        )

    def test_workflow_sliding_rescues_on_second_window(self) -> None:
        """测试：初答拒答，第1批(0-20)仍拒答，第2批(20-40)成功答对救回。"""
        question = RuntimeQuestion(
            sample_index=0,
            id="q0",
            question="Where was someone born?",
            topic_mids=("m.01",),
            paths=(),
        )
        # 准备 50 条 mock 路径
        mock_paths = tuple({"path": [["h", f"r_{i}", f"t_{i}"]], "log_score": -0.1 * i} for i in range(50))

        initial_res = AnswerResult(answers=(), is_rejection=True, format_ok=True, prompt_tokens=10)
        win0_res = AnswerResult(answers=(), is_rejection=True, format_ok=True, prompt_tokens=10)
        win1_res = AnswerResult(answers=("Target City",), is_rejection=False, format_ok=True, prompt_tokens=10)

        mock_answerer = MagicMock()
        mock_answerer.answer.side_effect = [initial_res, win0_res, win1_res]

        mock_router = MagicMock()
        mock_router.classify.return_value = RouteDecision("default", 0.9)

        mock_rescue = MagicMock()
        mock_rescue.run.return_value = ToolOutcome(
            applied=True,
            reason="external_paths",
            paths=mock_paths,
            stats={"path_n": len(mock_paths)},
        )

        wf = MasterQAWorkflow(
            self.config,
            mock_answerer,
            mock_router,
            comparison=MagicMock(),
            conjunction=MagicMock(),
            rescue=mock_rescue,
        )

        rec = wf.answer(question)

        # 验证结果
        self.assertFalse(rec.final.is_rejection)
        self.assertEqual(rec.final.answers, ("Target City",))
        self.assertEqual(rec.source, "rescue")
        self.assertEqual(rec.reason, "rescue_window_1")
        self.assertEqual(rec.pfit_calls, 3)  # 初答 1 次 + win0 1 次 + win1 1 次
        self.assertEqual(len(rec.call_results), 3)

        # 验证送入 answerer 的切片
        calls = mock_answerer.answer.call_args_list
        self.assertEqual(len(calls), 3)
        # calls[0] 是初答
        self.assertEqual(calls[0][0][1], ())
        # calls[1] 是 win0 (前 20 条)
        self.assertEqual(len(calls[1][0][1]), 20)
        self.assertEqual(calls[1][0][1][0]["path"][0][1], "r_0")
        # calls[2] 是 win1 (第 21-40 条)
        self.assertEqual(len(calls[2][0][1]), 20)
        self.assertEqual(calls[2][0][1][0]["path"][0][1], "r_20")

    def test_workflow_sliding_stops_early_on_first_window_success(self) -> None:
        """测试：零误伤防御，若第1批(0-20)已给出实质答案，绝不继续调用后续窗口。"""
        question = RuntimeQuestion(
            sample_index=1,
            id="q1",
            question="When was it founded?",
            topic_mids=("m.02",),
            paths=(),
        )
        mock_paths = tuple({"path": [["h", f"r_{i}", f"t_{i}"]], "log_score": -0.1 * i} for i in range(50))

        initial_res = AnswerResult(answers=(), is_rejection=True, format_ok=True)
        win0_res = AnswerResult(answers=("1990",), is_rejection=False, format_ok=True)

        mock_answerer = MagicMock()
        mock_answerer.answer.side_effect = [initial_res, win0_res]

        mock_router = MagicMock()
        mock_router.classify.return_value = RouteDecision("default", 0.9)

        mock_rescue = MagicMock()
        mock_rescue.run.return_value = ToolOutcome(
            applied=True,
            reason="external_paths",
            paths=mock_paths,
        )

        wf = MasterQAWorkflow(
            self.config,
            mock_answerer,
            mock_router,
            comparison=MagicMock(),
            conjunction=MagicMock(),
            rescue=mock_rescue,
        )

        rec = wf.answer(question)

        self.assertFalse(rec.final.is_rejection)
        self.assertEqual(rec.final.answers, ("1990",))
        self.assertEqual(rec.source, "rescue")
        self.assertEqual(rec.reason, "rescue_window_0")
        self.assertEqual(rec.pfit_calls, 2)  # 初答 1 次 + win0 1 次，无额外调用
        self.assertEqual(mock_answerer.answer.call_count, 2)

    def test_workflow_sliding_respects_max_pfit_calls(self) -> None:
        """测试：当达到 max_pfit_calls 上限时，滑窗停止。"""
        strict_config = WorkflowConfig(
            max_pfit_calls=2,  # 仅允许 2 次调用：初答 1 次 + win0 1 次
            rescue_enabled=True,
            rescue_sliding_enabled=True,
            rescue_window_size=20,
            rescue_max_depth=100,
        )
        question = RuntimeQuestion(
            sample_index=2,
            id="q2",
            question="Test question?",
            topic_mids=("m.03",),
            paths=(),
        )
        mock_paths = tuple({"path": [["h", f"r_{i}", f"t_{i}"]], "log_score": -0.1 * i} for i in range(50))

        initial_res = AnswerResult(answers=(), is_rejection=True, format_ok=True)
        win0_res = AnswerResult(answers=(), is_rejection=True, format_ok=True)

        mock_answerer = MagicMock()
        mock_answerer.answer.side_effect = [initial_res, win0_res]

        mock_router = MagicMock()
        mock_router.classify.return_value = RouteDecision("default", 0.9)

        mock_rescue = MagicMock()
        mock_rescue.run.return_value = ToolOutcome(
            applied=True,
            reason="external_paths",
            paths=mock_paths,
        )

        wf = MasterQAWorkflow(
            strict_config,
            mock_answerer,
            mock_router,
            comparison=MagicMock(),
            conjunction=MagicMock(),
            rescue=mock_rescue,
        )

        rec = wf.answer(question)

        self.assertTrue(rec.final.is_rejection)
        self.assertEqual(rec.reason, "pfit_budget_exhausted")
        self.assertEqual(rec.pfit_calls, 2)
        self.assertEqual(mock_answerer.answer.call_count, 2)

    def test_rescue_tool_selects_depth_up_to_100(self) -> None:
        """测试：GraphRescuer 在 rescue_sliding_enabled 时支持保留最多 100 条路径。"""
        rescuer = GraphRescuer(
            config=self.config,
            library=MagicMock(),
            kg=MagicMock(),
            names={"m.01": "Node1", "m.02": "Node2"},
        )
        # 准备 60 条 records
        mock_records = [
            {
                "nodes": ["m.01", "m.02"],
                "relations": [f"rel_{i}"],
                "score": 0.9 - 0.01 * i,
            }
            for i in range(60)
        ]
        selected = rescuer._template_select(mock_records)
        # 应保留全部 60 条，而不是被 20 截断
        self.assertEqual(len(selected), 60)


if __name__ == "__main__":
    unittest.main()
