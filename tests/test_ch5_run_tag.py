"""第五章任务隔离目录与回放源选择测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.ch5 import run


class Ch5RunTagTests(unittest.TestCase):
    def test_tag_isolates_outputs_and_replay_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "matrix.json"
            config.write_text(json.dumps({
                "dataset": "webqsp", "qa_file": "qa.txt", "runs": {"benchmark": [{
                    "id": "trace", "beam_size": 50, "batch_size": 20,
                    "lambda_val": 0.2, "check_mode": "hybrid-reject-list",
                }]},
            }), encoding="utf-8")
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "config_id": "transfernet_v2", "backbone": "transfernet", "topk": 500, "retrieve": {},
            }), encoding="utf-8")
            args = ["--dataset", "webqsp", "--config", str(config), "--profile", str(profile),
                    "--project_dir", str(root), "--run_tag", "followup_test", "--dry_run"]
            with patch.object(run, "run_command") as command:
                run.main([*args, "--phase", "benchmark"])
                output = command.call_args.args[1]
                self.assertEqual(output.parts[-2:], ("benchmark", "followup_test"))
                run.main([*args, "--phase", "replay_ablations"])
                replay_args = command.call_args.args[0]
                source = Path(replay_args[replay_args.index("--canonical_dir") + 1])
                self.assertEqual(source, output)

    def test_tag_rejects_parent_and_absolute_paths(self):
        for tag in ("../escape", "/tmp/escape", "nested/tag", ".."):
            with self.subTest(tag=tag), self.assertRaises(SystemExit):
                run.build_parser().parse_args([
                    "--dataset", "webqsp", "--config", "c", "--profile", "p", "--run_tag", tag,
                ])


if __name__ == "__main__":
    unittest.main()
