import argparse
import json
import tempfile
import unittest
from pathlib import Path

from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.runtime import configure_runtime, emit_event, update_progress


class TestExperimentPaths(unittest.TestCase):
    def test_paths_are_under_unified_output_root(self):
        paths = ExperimentPaths(Path("/tmp/project"))
        self.assertEqual(
            paths.score_dir("webqsp", "transfernet", "topk500"),
            Path("/tmp/project/data/output/kgqa/shared/webqsp/backbones/transfernet/scores/topk500"),
        )
        self.assertEqual(
            paths.ch4_run_dir("metaqa", "v1", "main", 17),
            Path("/tmp/project/data/output/kgqa/ch4_pfit/metaqa/v1/main/seed_17"),
        )
        self.assertEqual(
            paths.ch3_shortest_path_dir("webqsp", "transfernet", "v1"),
            Path("/tmp/project/data/output/kgqa/ch3_retrieval/webqsp/transfernet/shortest_path_baselines/v1"),
        )
        self.assertEqual(
            paths.ch3_downstream_qa_dir("webqsp", "transfernet", "v1"),
            Path("/tmp/project/data/output/kgqa/ch3_retrieval/webqsp/transfernet/downstream_qa/v1"),
        )
        self.assertEqual(
            paths.ch5_dir("webqsp", "v1", "benchmark"),
            Path("/tmp/project/data/output/kgqa/ch5_pv_gac/webqsp/v1/benchmark"),
        )

    def test_unknown_ch5_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知第五章阶段"):
            ExperimentPaths(Path("/tmp/project")).ch5_dir("webqsp", "v1", "unknown")


class TestConfigLoading(unittest.TestCase):
    def test_confirmed_config_requires_manual_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"kind": "ch3_retrieval_profile", "status": "draft"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "尚未人工确认"):
                load_confirmed_config(path)

            path.write_text(json.dumps({
                "kind": "ch3_retrieval_profile", "status": "confirmed",
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1",
                "topk": 500, "retrieve": {},
            }), encoding="utf-8")
            self.assertEqual(load_confirmed_config(path)["config_id"], "v1")

    def test_invalid_json_has_chinese_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不是合法 JSON"):
                load_json_config(path)


class TestRuntimeFiles(unittest.TestCase):
    def test_runtime_writes_manifest_progress_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(run_dir=directory, log_level="INFO")
            run_dir = configure_runtime(args, command="测试命令", manifest={"dataset": "webqsp"})
            update_progress(run_dir, completed=2, total=5, phase="检索")
            emit_event(run_dir, "checkpoint", completed=2)
            self.assertEqual(json.loads((Path(directory) / "run_manifest.json").read_text(encoding="utf-8"))["dataset"], "webqsp")
            self.assertEqual(json.loads((Path(directory) / "progress.json").read_text(encoding="utf-8"))["completed"], 2)
            self.assertIn("checkpoint", (Path(directory) / "logs/events.jsonl").read_text(encoding="utf-8"))
            self.assertTrue((Path(directory) / "logs/run.log").is_file())
