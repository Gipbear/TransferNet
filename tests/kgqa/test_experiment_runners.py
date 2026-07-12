import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments import run_ch3, run_ch4, run_ch5


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


class TestExperimentRunners(unittest.TestCase):
    def test_ch3_dry_run_writes_under_unified_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "kind": "ch3_retrieval_profile", "status": "draft", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "topk_candidates": [100],
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {"test": {"qa_file": "data/test.txt"}}},
                "retrieve": {"beam_size": 50, "lambda_val": 0.2, "threshold": 0.01, "alpha_final": 1.0},
                "parameter_scan": [{"id": "对照", "retrieve": {"lambda_val": 0.0}}],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main(["--dataset", "webqsp", "--config", str(config), "--project_dir", str(root), "--dry_run"])
            self.assertIn("data/output/kgqa/ch3_retrieval/webqsp/transfernet", stream.getvalue())

    def test_ch4_requires_confirmed_profile_before_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {"kind": "ch3_retrieval_profile", "status": "draft"})
            matrix = write_json(root / "ch4.json", {"dataset": "webqsp", "config_id": "v1", "experiments": []})
            with self.assertRaisesRegex(ValueError, "尚未人工确认"):
                run_ch4.main(["--dataset", "webqsp", "--config", str(matrix), "--profile", str(profile), "--project_dir", str(root), "--dry_run"])

    def test_ch5_replay_uses_benchmark_as_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
            })
            matrix = write_json(root / "ch5.json", {"dataset": "webqsp", "qa_file": "qa.txt", "runs": {}})
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch5.main(["--dataset", "webqsp", "--config", str(matrix), "--profile", str(profile), "--project_dir", str(root), "--phase", "replay_ablations", "--dry_run"])
            self.assertIn("ch5_pv_gac/webqsp/v1/benchmark", stream.getvalue())
