import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.ch3 import run as run_ch3
from experiments.ch3 import run_downstream_qa
from experiments.ch4 import run as run_ch4
from experiments.ch5 import run as run_ch5
from experiments.common import run_command
from kgqa.retrieve.cli import eval as eval_cli
from kgqa.retrieve.cli.dump_scores import truncate_score_cache


def write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


class TestExperimentRunners(unittest.TestCase):
    def test_ch3_downstream_qa_dry_run_uses_unified_root_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
            })
            row = {"question": "q", "golden": ["a"], "hop": 1, "mmr_reason_paths": []}
            conditions = []
            methods = {
                "no_path": {"no_paths": True},
                "shortest_path": {"method": "shortest_path_postprocess"},
                "score_beam": {
                    "beam_size": 20, "lambda_val": 0.0, "eta": 0.0, "penalty_mode": "none",
                },
                "terminal_score_beam": {
                    "beam_size": 20, "lambda_val": 0.0, "eta": 1.0, "penalty_mode": "none",
                },
                "fixed": {
                    "beam_size": 20, "lambda_val": 0.2, "eta": 1.0, "penalty_mode": "fixed",
                },
                "tarrs": {
                    "beam_size": 20, "lambda_val": 0.2, "eta": 1.0, "penalty_mode": "adaptive",
                },
            }
            for condition_id, method in methods.items():
                input_path = root / f"{condition_id}.jsonl"
                input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                condition = {"id": condition_id, "label": condition_id, "input": str(input_path), "method": method}
                if condition_id == "no_path":
                    condition["no_paths"] = True
                conditions.append(condition)
            config = write_json(root / "downstream.json", {
                "kind": "ch3_downstream_qa", "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1",
                "profile": str(profile), "evaluation": {"model": "model", "format": "v2", "path_format": "chain", "entity_repr": "name", "max_new_tokens": 2, "batch_size": 1, "path_budget": 20},
                "conditions": conditions, "fixed_pfit_adapter": None,
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_downstream_qa.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--condition", "fixed", "--phase", "eval", "--dry_run", "--no_progress",
                ])
            text = stream.getvalue()
            self.assertIn("kgqa.pfit.eval_batch", text)
            self.assertIn(
                "data/output/kgqa/ch3_retrieval/webqsp/transfernet/downstream_qa/"
                "v1/base_zeroshot/full/batch_fixed/jobs.json",
                text,
            )
            self.assertFalse((root / "data/output/kgqa/ch3_retrieval/webqsp/transfernet/downstream_qa/v1").exists())

    def test_ch3_downstream_qa_accepts_multiple_conditions_in_one_batch(self):
        self.assertEqual(
            run_downstream_qa._selected_conditions("terminal_score_beam,fixed,tarrs"),
            ("terminal_score_beam", "fixed", "tarrs"),
        )
        with self.assertRaisesRegex(ValueError, "未知或重复条件"):
            run_downstream_qa._selected_conditions("tarrs,tarrs")

        metaqa_conditions = ("no_path", "shortest_path", "score_beam", "fixed", "tarrs")
        self.assertEqual(
            run_downstream_qa._selected_conditions("all", metaqa_conditions),
            metaqa_conditions,
        )
        with self.assertRaisesRegex(ValueError, "未知或重复条件"):
            run_downstream_qa._selected_conditions("terminal_score_beam", metaqa_conditions)

    def test_batch_evaluation_marks_batch_run_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_file = Path(directory) / "jobs.json"
            jobs_file.write_text(json.dumps([
                {"id": "first"}, {"id": "second"},
            ]), encoding="utf-8")
            args = Namespace(
                backend="offline", jobs_file=str(jobs_file), no_progress=True,
                progress_interval=50, dataset="webqsp", backbone="transfernet",
                cache="cache.pt", run_dir="", log_level="INFO",
            )
            batch_dir = Path(directory) / "batch"
            summary = {"backbone": {"overall": {}}}
            with patch.object(eval_cli, "configure_runtime", return_value=batch_dir), \
                 patch.object(eval_cli, "build_backend", return_value=object()), \
                 patch.object(eval_cli, "_run_job", return_value=([object()], summary, 1.0)), \
                 patch.object(eval_cli, "update_progress") as update_progress, \
                 patch.object(eval_cli, "emit_event") as emit_event:
                eval_cli.run_jobs(args)
            update_progress.assert_called_with(
                batch_dir, completed=2, total=2, status="completed", phase="路径检索批量评测")
            emit_event.assert_called_with(batch_dir, "phase_end", phase="路径检索批量评测", jobs=2)

    def test_truncate_score_cache_keeps_top_ranked_scores_and_metadata(self):
        cache = {
            "meta": {"topk_entities": 4},
            "samples": [{
                "ent_indices": [torch.tensor([4, 3, 2, 1])],
                "ent_scores": [torch.tensor([0.9, 0.8, 0.7, 0.6])],
                "e_score_indices": torch.tensor([8, 7, 6, 5]),
                "e_score_values": torch.tensor([0.8, 0.7, 0.6, 0.5]),
            }],
        }
        truncated = truncate_score_cache(cache, 2)
        self.assertEqual(truncated["meta"]["topk_entities"], 2)
        self.assertEqual(truncated["samples"][0]["ent_indices"][0].tolist(), [4, 3])
        self.assertEqual(truncated["samples"][0]["e_score_indices"].tolist(), [8, 7])
        self.assertEqual(cache["samples"][0]["e_score_indices"].tolist(), [8, 7, 6, 5])

    def test_run_command_does_not_write_progress_frames_to_console_log(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_command(
                    [
                        sys.executable,
                        "-c", "import sys; print('开始输出', flush=True); sys.stderr.write(chr(13) + "
                        "''.join(map(chr, [80, 53, 48]))); "
                        "sys.stderr.flush(); print('结束输出', flush=True)",
                    ],
                    run_dir,
                    dry_run=False,
                )
            self.assertIn("开始输出", stream.getvalue())
            self.assertIn("P50", stream.getvalue())
            console = (run_dir / "logs/console.log").read_bytes()
            self.assertIn("开始输出".encode("utf-8"), console)
            self.assertIn("结束输出".encode("utf-8"), console)
            self.assertIn("[开始]".encode("utf-8"), console)
            self.assertIn("[结束] 退出码=0，耗时=".encode("utf-8"), console)
            self.assertNotIn(b"P50", console)
            self.assertNotIn(b"\r", console)

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
            self.assertIn('"--relation_normalization" "global"', stream.getvalue())
            self.assertIn('"--no_progress"', stream.getvalue())
            self.assertIn('"--progress_interval" "7"', stream.getvalue())

    def test_ch3_webqsp_config_defines_one_factor_parameter_scan(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "experiments/configs/ch3/webqsp_transfernet_v2.json").read_text(encoding="utf-8"))
        scan = config["parameter_scan"]
        triples = {
            (item["beam_size"], item["lambda_val"], item["eta"])
            for item in scan["items"]
        }
        self.assertEqual(len(triples), 16)
        self.assertEqual(
            {beam for beam, lambda_val, eta in triples if lambda_val == 0.2 and eta == 1.0},
            {3, 5, 10, 20, 50, 100},
        )
        self.assertEqual(
            {lambda_val for beam, lambda_val, eta in triples if beam == 20 and eta == 1.0},
            {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0},
        )
        self.assertEqual(
            {eta for beam, lambda_val, eta in triples if beam == 20 and lambda_val == 0.2},
            {0.0, 0.5, 1.0, 1.5, 2.0},
        )
        self.assertIn((50, 0.2, 1.0), triples)
        self.assertEqual(config["retrieve"]["eta"], 1.0)
        self.assertEqual(config["retrieve"]["step_score_mode"], "joint")
        self.assertEqual(config["retrieve"]["penalty_mode"], "adaptive")
        self.assertEqual(config["retrieve"]["relation_normalization"], "global")
        self.assertEqual(
            [item["id"] for item in config["score_component_ablation"]],
            ["joint_eta1", "joint_eta0", "relation_only", "entity_only"],
        )
        self.assertEqual(
            [item["id"] for item in config["penalty_ablation"]],
            ["none", "fixed", "adaptive"],
        )

    def test_ch3_penalty_ablation_dry_run_uses_separate_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "selection_split": "test",
                "retrieve": {"beam_size": 20, "lambda_val": 0.2, "threshold": 0.01,
                             "eta": 1.0, "penalty_mode": "adaptive"},
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {
                    "test": {"qa_file": "data/test.txt"},
                }},
                "penalty_ablation": [
                    {"id": "none", "label": "无惩罚", "retrieve": {"penalty_mode": "none"}},
                    {"id": "fixed", "label": "固定", "retrieve": {"penalty_mode": "fixed"}},
                    {"id": "adaptive", "label": "自适应", "retrieve": {"penalty_mode": "adaptive"}},
                ],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--phase", "penalty_ablation", "--dry_run", "--no_progress",
                ])
            output = stream.getvalue()
            self.assertIn("penalty_ablations/v1/none/test", output)
            self.assertIn("penalty_ablations/v1/fixed/test", output)
            self.assertIn("penalty_ablations/v1/adaptive/test", output)
            self.assertEqual(output.count('"-m" "kgqa.retrieve.cli.eval"'), 1)

    def test_ch3_score_ablation_is_explicit_not_cartesian_scan(self):
        config = {
            "retrieve": {"beam_size": 20, "lambda_val": 0.5, "threshold": 0.01, "eta": 1.5},
            "score_component_ablation": [
                {"id": "joint", "label": "联合", "retrieve": {"step_score_mode": "joint", "eta": 1.5}},
                {"id": "relation", "label": "仅关系", "retrieve": {"step_score_mode": "relation_only", "eta": 0.0}},
            ],
        }
        items = run_ch3._score_ablation_items(config)
        self.assertEqual([item["id"] for item in items], ["joint", "relation"])
        self.assertEqual(items[1]["retrieve"]["beam_size"], 20)
        self.assertEqual(items[1]["retrieve"]["step_score_mode"], "relation_only")

    def test_ch3_score_ablation_dry_run_uses_separate_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "selection_split": "test",
                "retrieve": {"beam_size": 20, "lambda_val": 0.5, "threshold": 0.01, "eta": 1.5},
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {
                    "test": {"qa_file": "data/test.txt"},
                }},
                "score_component_ablation": [
                    {"id": "joint", "label": "联合", "retrieve": {"step_score_mode": "joint", "eta": 1.5}},
                    {"id": "relation", "label": "仅关系", "retrieve": {"step_score_mode": "relation_only", "eta": 0.0}},
                ],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--phase", "score_ablation", "--dry_run", "--no_progress",
                ])
            output = stream.getvalue()
            self.assertIn("score_component_ablations/v1/joint/test", output)
            self.assertIn("score_component_ablations/v1/relation/test", output)
            self.assertEqual(output.count('"-m" "kgqa.retrieve.cli.eval"'), 1)
            self.assertFalse((
                root / "data/output/kgqa/ch3_retrieval/webqsp/transfernet/"
                "score_component_ablations/v1/relation/test/run_manifest.json"
            ).exists())

    def test_ch3_shortest_path_dry_run_keeps_completed_progress_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "selection_split": "test",
                "retrieve": {"beam_size": 20, "lambda_val": 0.5, "threshold": 0.01, "eta": 1.5},
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {
                    "test": {"qa_file": "data/test.txt"},
                }},
                "shortest_path_baseline": {
                    "id": "top20", "label": "候选答案最短路径", "candidate_topk": 20,
                    "max_paths_per_pair": 20, "path_budget": 20,
                    "max_hop_source": "available_steps", "drop_loopback": True,
                },
            })
            progress_path = (
                root / "data/output/kgqa/ch3_retrieval/webqsp/transfernet/"
                "shortest_path_baselines/v1/top20/test/progress.json"
            )
            progress_path.parent.mkdir(parents=True)
            completed = {"status": "completed", "completed": 1581, "total": 1581}
            progress_path.write_text(json.dumps(completed), encoding="utf-8")
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--phase", "shortest_path", "--dry_run", "--no_progress",
                ])
            self.assertIn("kgqa.retrieve.cli.shortest_path", stream.getvalue())
            self.assertIn("shortest_path_baselines/v1/top20/test.jsonl", stream.getvalue())
            self.assertEqual(json.loads(progress_path.read_text(encoding="utf-8")), completed)

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

    def test_ch3_parameter_scan_accepts_explicit_one_factor_items(self):
        items = run_ch3._parameter_scan_items({
            "parameter_scan": {"items": [
                {"beam_size": 3, "lambda_val": 0.2, "eta": 1.0},
                {"beam_size": 20, "lambda_val": 0.0, "eta": 1.0},
                {"beam_size": 20, "lambda_val": 0.2, "eta": 0.0},
            ]},
        })
        self.assertEqual(
            [item["id"] for item in items],
            ["beam3_lambda02_eta1", "beam20_lambda0_eta1", "beam20_lambda02_eta0"],
        )
        self.assertEqual(
            [item["retrieve"] for item in items],
            [
                {"beam_size": 3, "lambda_val": 0.2, "eta": 1.0},
                {"beam_size": 20, "lambda_val": 0.0, "eta": 1.0},
                {"beam_size": 20, "lambda_val": 0.2, "eta": 0.0},
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

    def test_ch3_parameter_scan_runs_candidates_in_one_batch_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_json(root / "ch3.json", {
                "dataset": "webqsp", "backbone": "transfernet", "config_id": "v1", "topk": 500,
                "selection_split": "test", "retrieve": {"beam_size": 50, "lambda_val": 0.2, "threshold": 0.01, "eta": 1.0},
                "score_source": {"ckpt": "models/a.pt", "input_dir": "data/input/WebQSP", "splits": {
                    "test": {"qa_file": "data/test.txt"},
                }},
                "parameter_scan": {"beam_size": [20, 50], "lambda_val": [0.2], "eta": [1.0]},
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch3.main([
                    "--dataset", "webqsp", "--config", str(config), "--project_dir", str(root),
                    "--phase", "scan", "--dry_run", "--no_progress",
                ])
            output = stream.getvalue()
            self.assertEqual(output.count('"-m" "kgqa.retrieve.cli.eval"'), 1)
            self.assertIn('"--jobs_file"', output)

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

    def test_ch4_dry_run_keeps_matrix_level_inputs_and_default_train_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
            })
            matrix = write_json(root / "ch4.json", {
                "dataset": "webqsp", "config_id": "v1",
                "train_file": "matrix_train.jsonl", "test_file": "matrix_test.jsonl",
                "experiments": [{"id": "baseline", "mode": "train", "seeds": [17]}],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch4.main([
                    "--dataset", "webqsp", "--config", str(matrix), "--profile", str(profile),
                    "--project_dir", str(root), "--dry_run",
                ])
            output = stream.getvalue()
            self.assertIn("confirmed_profiles/v1/matrix_train.jsonl", output)
            self.assertIn("confirmed_profiles/v1/matrix_test.jsonl", output)
            self.assertNotIn('"--epochs"', output)

    def test_ch4_entry_overrides_inputs_and_forwards_train_args(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = write_json(root / "profile.json", {
                "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
                "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
            })
            matrix = write_json(root / "ch4.json", {
                "dataset": "webqsp", "config_id": "v1",
                "train_file": "matrix_train.jsonl", "test_file": "matrix_test.jsonl",
                "experiments": [{
                    "id": "override", "mode": "train", "seeds": [17],
                    "train_file": "candidates/custom/train.jsonl",
                    "test_file": "candidates/custom/test.jsonl",
                    "train_args": ["--epochs", "3"],
                }],
            })
            stream = io.StringIO()
            with redirect_stdout(stream):
                run_ch4.main([
                    "--dataset", "webqsp", "--config", str(matrix), "--profile", str(profile),
                    "--project_dir", str(root), "--dry_run",
                ])
            output = stream.getvalue()
            self.assertIn("confirmed_profiles/v1/candidates/custom/train.jsonl", output)
            self.assertIn("confirmed_profiles/v1/candidates/custom/test.jsonl", output)
            self.assertNotIn("confirmed_profiles/v1/matrix_train.jsonl", output)
            self.assertNotIn("confirmed_profiles/v1/matrix_test.jsonl", output)
            self.assertEqual(output.count('"--epochs" "3"'), 1)

    def test_ch4_qwen35_config_uses_16bit_text_only_for_train_and_eval(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch4/webqsp_qwen35_2b_v1.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))

        experiment = config["experiments"][0]
        expected_loading_args = [
            "--model", "unsloth/Qwen3.5-2B",
            "--model_precision", "16bit",
            "--text_only",
        ]
        self.assertEqual(experiment["train_args"][:5], expected_loading_args)
        self.assertEqual(experiment["eval_args"][-5:], expected_loading_args)

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
