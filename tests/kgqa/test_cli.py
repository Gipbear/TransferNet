import os
import json
import tempfile
import unittest

from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestCLI(unittest.TestCase):
    def test_parsers_build(self):
        from kgqa.cli import retrieve, eval as eval_cli, dump_scores
        retrieve_parser = retrieve.build_parser()
        self.assertIsNotNone(retrieve_parser)
        self.assertNotIn("--method", retrieve_parser.format_help())
        self.assertIsNotNone(eval_cli.build_parser())
        self.assertIsNotNone(dump_scores.build_parser())

    @unittest.skipUnless(artifact_test_available(CACHE), ARTIFACT_TEST_SKIP_REASON)
    def test_eval_writes_summary(self):
        from kgqa.cli import eval as eval_cli
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "summary.json")
            eval_cli.main([
                "--dataset", "webqsp", "--backend", "offline",
                "--cache", CACHE, "--input_dir", "data/input/WebQSP",
                "--limit", "20", "--beam_size", "50", "--summary", out,
            ])
            with open(out, encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertIn("answer", summary)
            self.assertIn("path", summary)
            self.assertIn("overall", summary["answer"])


if __name__ == "__main__":
    unittest.main()
