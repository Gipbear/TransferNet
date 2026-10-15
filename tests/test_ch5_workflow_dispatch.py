"""验证正式第五章入口对三路工作流的独立分派。"""

import unittest
from pathlib import Path

from experiments.ch5.run import _agent_command, _workflow_replay, _workflow_scoring


class TestWorkflowDispatch(unittest.TestCase):
    def test_workflow_configuration_does_not_require_checker_arguments(self):
        config = {"kind": "ch5_workflow", "qa_file": "input.jsonl", "templates": "templates.joblib",
                  "cache_dir": "cache", "_config_path": "config.json"}
        command = _agent_command("cwq", {"limit": 20}, Path("out"), config)
        self.assertIn("kgqa.agent.cli.run_workflow", command)
        self.assertNotIn("--check_mode", command)

    def test_full_scoring_does_not_accept_incomplete_predictions(self):
        command = _workflow_scoring({"limit": 0}, Path("out"), {"labels_source": "labels.jsonl"})
        self.assertNotIn("--allow_partial", command)
        self.assertIn("kgqa.agent.cli.score_workflow", command)

    def test_smoke_scoring_is_explicitly_partial(self):
        command = _workflow_scoring({"limit": 20}, Path("out"), {"labels_source": "labels.jsonl"})
        self.assertIn("--allow_partial", command)

    def test_workflow_replay_uses_workflow_predictions(self):
        config = {"labels_source": "labels.jsonl", "runs": {"benchmark": [{"id": "full"}]}}
        command = _workflow_replay(Path("benchmark"), Path("out"), config)
        self.assertIn("kgqa.agent.cli.replay_workflow", command)
        self.assertIn("benchmark/full/predictions_no_gold.jsonl", command)


if __name__ == "__main__":
    unittest.main()
