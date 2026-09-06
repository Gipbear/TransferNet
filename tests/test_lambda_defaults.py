"""MMR lambda 默认值全链路统一为 0.2(与 Ch3 grid search / pathfinder 配置一致)。"""

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def param_default(func, name):
    return inspect.signature(func).parameters[name].default


class LambdaDefaultTests(unittest.TestCase):
    def test_eval_cli_default(self):
        from kgqa.agent.cli.eval_checked_batch import build_parser

        self.assertEqual(build_parser().parse_args([]).lambda_val, 0.2)

    def test_run_cli_default(self):
        from kgqa.agent.cli.run_checked_batch import build_parser

        self.assertEqual(build_parser().get_default("lambda_val"), 0.2)

    def test_agent_run_default(self):
        from kgqa.agent.checked_batch import CheckedBatchAgent

        self.assertEqual(param_default(CheckedBatchAgent.run, "lambda_val"), 0.2)

    def test_path_retrieve_tool_default(self):
        from kgqa.agent.tools.path_retrieve import PathRetrieveTool

        self.assertEqual(param_default(PathRetrieveTool.__call__, "lambda_val"), 0.2)

    def test_server_client_default(self):
        from kgqa.retrieve.api.client import PathRetrieveClient

        self.assertEqual(
            param_default(PathRetrieveClient.retrieve, "lambda_val"), 0.2
        )

    def test_server_schema_default(self):
        from kgqa.retrieve.api.schema import RetrieveRequest

        self.assertEqual(RetrieveRequest(sample_index=0).lambda_val, 0.2)

    def test_service_default(self):
        from kgqa.retrieve.api.service import PathRetrieveService

        self.assertEqual(
            param_default(PathRetrieveService.retrieve, "lambda_val"), 0.2
        )

    def test_checker_dataset_cli_exposes_lambda(self):
        from kgqa.agent.cli.build_checker_dataset import build_parser

        self.assertEqual(build_parser().parse_args([]).lambda_val, 0.2)


if __name__ == "__main__":
    unittest.main()
