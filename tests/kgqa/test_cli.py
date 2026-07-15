import os
import json
import tempfile
import unittest
from types import SimpleNamespace

from kgqa.core.contracts import MetricSpec, QASample, RetrieveResult
from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestCLI(unittest.TestCase):
    def test_parsers_build(self):
        from kgqa.retrieve.cli import retrieve, eval as eval_cli, dump_scores, shortest_path
        retrieve_parser = retrieve.build_parser()
        self.assertIsNotNone(retrieve_parser)
        self.assertNotIn("--method", retrieve_parser.format_help())
        self.assertIn("--backbone", retrieve_parser.format_help())
        self.assertIn("--eta", retrieve_parser.format_help())
        self.assertNotIn("--alpha_final", retrieve_parser.format_help())
        self.assertIn("--run_dir", retrieve_parser.format_help())
        self.assertEqual(retrieve_parser.parse_args(["--dataset", "webqsp", "--input_dir", "x", "--eta", "0.7"]).eta, 0.7)
        with self.assertRaises(SystemExit):
            retrieve_parser.parse_args(["--dataset", "webqsp", "--input_dir", "x", "--alpha_final", "0.7"])
        self.assertIsNotNone(eval_cli.build_parser())
        self.assertIsNotNone(dump_scores.build_parser())
        shortest_path_parser = shortest_path.build_parser()
        self.assertIn("--candidate_topk", shortest_path_parser.format_help())
        self.assertNotIn("--lambda_val", shortest_path_parser.format_help())

    def test_active_pfit_and_agent_parsers_have_runtime_args(self):
        from kgqa.agent.cli.eval_checked_batch import build_parser as build_agent_parser
        from kgqa.pfit.build import build_parser as build_build_parser
        from kgqa.pfit.eval import build_parser as build_eval_parser
        from kgqa.pfit.train import build_parser as build_train_parser

        for parser in (build_agent_parser(), build_build_parser(), build_eval_parser(), build_train_parser()):
            help_text = parser.format_help()
            self.assertIn("--run_dir", help_text)
            self.assertIn("--no_progress", help_text)

    def test_summary_separates_backbone_prediction_from_path_endpoint(self):
        from kgqa.retrieve.cli import eval as eval_cli

        sample = QASample(question="q", topic_ids=[0], gold_ids=[1], sample_index=0, hop=1)
        result = RetrieveResult(
            question="q", topics=["m.topic"], hop=1,
            paths=[{"path": [["m.topic", "r", "m.gold"]], "log_score": -0.1}],
            prediction={"m.other": 0.9}, elapsed_ms=0.0, sample_index=0,
        )
        backend = SimpleNamespace(
            adapter=SimpleNamespace(metric_spec=lambda: MetricSpec()),
            bundle=SimpleNamespace(
                meta=SimpleNamespace(id2ent={1: "m.gold"}, id2rel={}),
                samples=[sample],
            ),
        )

        summary = eval_cli.evaluate_results(backend, [result], None)

        self.assertEqual(summary["backbone"]["overall"]["hit1"], 0.0)
        self.assertEqual(summary["path"]["overall"]["answer_hit"], 1.0)
        self.assertNotIn("answer", summary)

    @unittest.skipUnless(artifact_test_available(CACHE), ARTIFACT_TEST_SKIP_REASON)
    def test_eval_writes_summary(self):
        from kgqa.retrieve.cli import eval as eval_cli
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "summary.json")
            eval_cli.main([
                "--dataset", "webqsp", "--backend", "offline",
                "--cache", CACHE, "--input_dir", "data/input/WebQSP",
                "--limit", "20", "--beam_size", "50", "--summary", out,
            ])
            with open(out, encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertIn("backbone", summary)
            self.assertNotIn("answer", summary)
            self.assertIn("path", summary)
            self.assertIn("overall", summary["backbone"])


if __name__ == "__main__":
    unittest.main()
