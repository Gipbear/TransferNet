import unittest
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathfinder_agent.agent import PathfinderAgent
from pathfinder_agent.tools.question_utils import normalize_question
from run_agent_eval import resolve_scored_answers


class PathfinderAgentFlowTest(unittest.TestCase):
    def _agent(self):
        agent = PathfinderAgent.__new__(PathfinderAgent)
        agent.model = object()
        agent.tokenizer = object()
        agent.transfernet_wrapper = object()
        return agent

    def test_normalize_question_removes_special_tokens_and_wordpieces(self):
        self.assertEqual(
            normalize_question("[CLS] what did george or ##well write [SEP]"),
            "what did george orwell write",
        )

    def test_run_requires_transfernet_wrapper(self):
        agent = self._agent()
        agent.transfernet_wrapper = None

        with self.assertRaisesRegex(RuntimeError, "transfernet_wrapper"):
            agent.run("question", "m.topic")

    def test_agent_import_does_not_configure_root_logger(self):
        code = """
import logging
import pathfinder_agent.agent as agent_module

root_handlers_after_import = len(logging.getLogger().handlers)
agent_logger = logging.getLogger("pathfinder_agent")
file_handlers_before_setup = sum(isinstance(h, logging.FileHandler) for h in agent_logger.handlers)
agent_module._setup_agent_logging()
agent_module._setup_agent_logging()
file_handlers_after_setup = sum(isinstance(h, logging.FileHandler) for h in agent_logger.handlers)

print(root_handlers_after_import)
print(file_handlers_before_setup)
print(file_handlers_after_setup)
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.stdout.splitlines(), ["0", "0", "1"])

    def test_fallback_answer_is_reverified_before_aggregation(self):
        agent = self._agent()

        with (
            patch("pathfinder_agent.agent.rewrite_question", return_value=["raw question"]) as rewrite,
            patch(
                "pathfinder_agent.agent.retrieve_paths",
                side_effect=[["primary_path"], ["fallback_path"]],
            ) as retrieve,
            patch(
                "pathfinder_agent.agent.reason_with_paths",
                side_effect=[(["bad"], {1}), (["good"], {1})],
            ) as reason,
            patch(
                "pathfinder_agent.agent.verify_answer",
                side_effect=[(False, "slot mismatch"), (True, "OK")],
            ) as verify,
            patch(
                "pathfinder_agent.agent.aggregate_answers",
                return_value=["good"],
            ) as aggregate,
        ):
            result = agent.run("[CLS] raw question [SEP]", "m.topic")

        self.assertEqual(result, ["good"])
        rewrite.assert_called_once_with(agent.model, agent.tokenizer, "raw question", "m.topic")
        self.assertEqual(retrieve.call_args_list[0].kwargs["fallback"], False)
        self.assertEqual(retrieve.call_args_list[1].kwargs["fallback"], True)
        self.assertEqual(reason.call_args_list[1].args[3], ["fallback_path"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(agent.last_evidence_paths, ["fallback_path"])
        aggregate.assert_called_once_with(
            agent.model,
            agent.tokenizer,
            [["good"]],
            question="raw question",
        )

    def test_invalid_fallback_answer_is_not_aggregated(self):
        agent = self._agent()

        with (
            patch("pathfinder_agent.agent.rewrite_question", return_value=["raw question"]),
            patch(
                "pathfinder_agent.agent.retrieve_paths",
                side_effect=[["primary_path"], ["fallback_path"]],
            ),
            patch(
                "pathfinder_agent.agent.reason_with_paths",
                side_effect=[(["bad"], {1}), (["still_bad"], {1})],
            ),
            patch(
                "pathfinder_agent.agent.verify_answer",
                side_effect=[(False, "slot mismatch"), (False, "hallucination")],
            ),
            patch(
                "pathfinder_agent.agent.aggregate_answers",
                return_value=[],
            ) as aggregate,
        ):
            result = agent.run("raw question", "m.topic")

        self.assertEqual(result, [])
        self.assertEqual(agent.last_evidence_paths, [])
        aggregate.assert_called_once_with(
            agent.model,
            agent.tokenizer,
            [],
            question="raw question",
        )


class PathfinderEvalMetricTest(unittest.TestCase):
    def test_name_predictions_are_path_constrained_before_metrics(self):
        pred_answers, expanded_answers, metrics = resolve_scored_answers(
            pred_answers=["Salisbury"],
            golden=["m.0jgvy"],
            mmr_paths=[
                {"path": [["m.topic", "rel", "m.0jgvy"]], "log_score": 0.0},
            ],
            rev_entity_map={"salisbury": {"m.010hl7", "m.0jgvy"}},
        )

        self.assertEqual(expanded_answers, ["m.010hl7", "m.0jgvy"])
        self.assertEqual(pred_answers, ["m.0jgvy"])
        self.assertEqual(metrics["hit1"], 1)
        self.assertEqual(metrics["f1"], 1.0)

    def test_mid_predictions_keep_existing_metric_behavior_without_entity_map(self):
        pred_answers, expanded_answers, metrics = resolve_scored_answers(
            pred_answers=["m.answer"],
            golden=["m.answer"],
            mmr_paths=[],
            rev_entity_map=None,
        )

        self.assertIsNone(expanded_answers)
        self.assertEqual(pred_answers, ["m.answer"])
        self.assertEqual(metrics["hit1"], 1)


if __name__ == "__main__":
    unittest.main()
