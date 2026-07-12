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
                "retrieve": {"beam_size": 50, "lambda_val": 0.2, "threshold": 0.01, "eta": 1.0},
                "parameter_scan": [{"id": "对照", "retrieve": {"lambda_val": 0.0}}],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main(["--dataset", "webqsp", "--config", str(config), "--project_dir", str(root), "--dry_run"])
            self.assertIn("data/output/kgqa/ch3_retrieval/webqsp/transfernet", stream.getvalue())
            self.assertIn("topk100_test/evaluation", stream.getvalue())
            self.assertIn('"--eta" "1.0"', stream.getvalue())

    def test_ch3_webqsp_config_defines_complete_beam_lambda_scan(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "experiments/configs/ch3/webqsp_transfernet_v1.json").read_text(encoding="utf-8"))
        pairs = {
            (item["retrieve"]["beam_size"], item["retrieve"]["lambda_val"])
            for item in config["parameter_scan"]
        }
        self.assertEqual(len(pairs), 15)
        self.assertEqual(pairs, {
            (beam_size, lambda_val)
            for beam_size in (20, 50, 100)
            for lambda_val in (0.0, 0.1, 0.2, 0.3, 0.5)
        })
        self.assertEqual(config["retrieve"]["eta"], 1.0)

    def test_ch4_requires_confirmed_profile_before_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {"kind": "ch3_retrieval_profile", "status": "draft"})
            matrix = write_json(root / "ch4.json", {"dataset": "webqsp", "config_id": "v1", "experiments": []})
            with self.assertRaisesRegex(ValueError, "尚未人工确认"):
                run_ch4.main(["--dataset", "webqsp", "--config", str(matrix), "--profile", str(profile), "--project_dir", str(root), "--dry_run"])

    def test_ch3_publish_copies_only_confirmed_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
                "selected_candidate": "人工选择", "score_source": {"splits": {"train": {}, "test": {}}},
            })
            base = root / "data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/v1/candidates/人工选择"
            base.mkdir(parents=True)
            (base / "train.jsonl").write_text('{"sample_index": 0}\n', encoding="utf-8")
            (base / "test.jsonl").write_text('{"sample_index": 1}\n', encoding="utf-8")
            run_ch3.main(["--dataset", "webqsp", "--config", str(config), "--project_dir", str(root), "--phase", "publish"])
            published = root / "data/output/kgqa/ch3_retrieval/webqsp/transfernet/confirmed_profiles/v1"
            self.assertTrue((published / "train.jsonl").is_file())
            self.assertEqual((published / "confirmed_config.json").read_text(encoding="utf-8"), config.read_text(encoding="utf-8"))

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
