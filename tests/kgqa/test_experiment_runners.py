import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments import run_ch3, run_ch4, run_ch5
from experiments.common import run_command


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


class TestExperimentRunners(unittest.TestCase):
    def test_run_command_realtime_output_is_copied_to_console_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_command(
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('标准输出'); sys.stderr.write('\\r进度条'); sys.stderr.flush()",
                    ],
                    run_dir,
                    dry_run=False,
                )
            self.assertIn("标准输出", stream.getvalue())
            self.assertIn("进度条", stream.getvalue())
            console = (run_dir / "logs/console.log").read_bytes()
            self.assertIn("标准输出".encode("utf-8"), console)
            self.assertIn(b"\r", console)

    def test_ch3_dry_run_writes_under_unified_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "kind": "ch3_retrieval_profile", "status": "draft", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "topk_candidates": [100],
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {"test": {"qa_file": "data/test.txt"}}},
                "retrieve": {"beam_size": 50, "lambda_val": 0.2, "threshold": 0.01, "eta": 1.0},
                "parameter_scan": {"beam_size": [50], "lambda_val": [0.0], "eta": [1.0]},
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--dry_run", "--no_progress", "--progress_interval", "7",
                ])
            self.assertIn("data/output/kgqa/ch3_retrieval/webqsp/transfernet", stream.getvalue())
            self.assertIn("topk100_test/evaluation", stream.getvalue())
            self.assertIn('"--eta" "1.0"', stream.getvalue())
            self.assertIn('"--no_progress"', stream.getvalue())
            self.assertIn('"--progress_interval" "7"', stream.getvalue())

    def test_ch3_webqsp_config_defines_complete_beam_lambda_eta_scan(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "experiments/configs/ch3/webqsp_transfernet_v1.json").read_text(encoding="utf-8"))
        scan = config["parameter_scan"]
        triples = {
            (beam_size, lambda_val, eta)
            for beam_size in scan["beam_size"]
            for lambda_val in scan["lambda_val"]
            for eta in scan["eta"]
        }
        self.assertEqual(len(triples), len(scan["beam_size"]) * len(scan["lambda_val"]) * len(scan["eta"]))
        self.assertIn((50, 0.2, 1.0), triples)
        self.assertEqual(scan["eta"], [0.5, 1.0, 1.5])
        self.assertEqual(config["retrieve"]["eta"], 1.0)

    def test_ch3_parameter_scan_generates_stable_candidate_ids(self):
        items = run_ch3._parameter_scan_items({
            "parameter_scan": {"beam_size": [20, 50], "lambda_val": [0.0, 0.2], "eta": [0.5, 1.0]},
        })
        self.assertEqual(
            [item["id"] for item in items],
            [
                "beam20_lambda0_eta05", "beam20_lambda0_eta1",
                "beam20_lambda02_eta05", "beam20_lambda02_eta1",
                "beam50_lambda0_eta05", "beam50_lambda0_eta1",
                "beam50_lambda02_eta05", "beam50_lambda02_eta1",
            ],
        )

    def test_ch3_parameter_scan_requires_eta_list(self):
        with self.assertRaisesRegex(ValueError, "parameter_scan.eta 必须是非空列表"):
            run_ch3._parameter_scan_items({
                "parameter_scan": {"beam_size": [50], "lambda_val": [0.2]},
            })

    def test_ch3_parameter_scan_uses_test_split_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "selection_split": "test", "retrieve": {"beam_size": 50, "lambda_val": 0.2, "threshold": 0.01, "eta": 1.0},
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {
                    "train": {"qa_file": "data/train.txt"}, "test": {"qa_file": "data/test.txt"},
                }},
                "parameter_scan": {"beam_size": [50], "lambda_val": [0.2], "eta": [1.0]},
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--phase", "scan", "--dry_run", "--no_progress",
                ])
            self.assertIn("topk500_test", stream.getvalue())
            self.assertNotIn("topk500_train", stream.getvalue())

    def test_ch3_rejects_deprecated_alpha_final(self):
        config = {
            "retrieve": {
                "beam_size": 50,
                "lambda_val": 0.2,
                "threshold": 0.01,
                "eta": 1.0,
                "alpha_final": 1.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "不接受字段: alpha_final"):
            run_ch3._retrieve_args(config, {})

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
            run_ch3.main([
                "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                "--phase", "publish", "--no_progress",
            ])
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
