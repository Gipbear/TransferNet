import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from experiments.ch3.metaqa_p4 import build_p4_report, filter_score_cache_by_hop
from experiments.ch3 import run as run_ch3


def _cache_sample(question: str, hop: int) -> dict:
    return {
        "question": question,
        "topic_ids": [0],
        "gold_ids": [1],
        "hop_attn": torch.tensor([0.0, 0.0, 1.0]),
        "rel_probs": [torch.tensor([1.0]) for _ in range(3)],
        "ent_indices": [torch.tensor([1]) for _ in range(3)],
        "ent_scores": [torch.tensor([1.0]) for _ in range(3)],
        "e_score_indices": torch.tensor([1]),
        "e_score_values": torch.tensor([1.0]),
        "hop": hop,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class TestMetaQAP4(unittest.TestCase):
    def test_p4_config_runs_only_three_hop_penalty_and_shortest_path_outputs(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "experiments/configs/ch3/metaqa_transfernet_v1_p4.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["selection_split"], "test_3hop")
        self.assertEqual([item["id"] for item in config["penalty_ablation"]], ["none", "fixed", "adaptive"])
        self.assertEqual(config["p4"]["expected_samples"], 14274)

        penalty_stream = io.StringIO()
        with redirect_stdout(penalty_stream):
            run_ch3.main([
                "--dataset", "metaqa", "--config", str(config_path),
                "--project_dir", str(root), "--phase", "penalty_ablation",
                "--dry_run", "--no_progress",
            ])
        penalty_output = penalty_stream.getvalue()
        self.assertIn("topk500_test_3hop/test_3hop.pt", penalty_output)
        self.assertIn("penalty_ablations/transfernet_v1_3hop/none/test_3hop", penalty_output)
        self.assertIn("penalty_ablations/transfernet_v1_3hop/fixed/test_3hop", penalty_output)
        self.assertIn("penalty_ablations/transfernet_v1_3hop/adaptive/test_3hop", penalty_output)

        shortest_stream = io.StringIO()
        with redirect_stdout(shortest_stream):
            run_ch3.main([
                "--dataset", "metaqa", "--config", str(config_path),
                "--project_dir", str(root), "--phase", "shortest_path",
                "--dry_run", "--no_progress",
            ])
        shortest_output = shortest_stream.getvalue()
        self.assertIn("topk500_test_3hop/test_3hop.pt", shortest_output)
        self.assertIn(
            "shortest_path_baselines/transfernet_v1_3hop/"
            "top20_hop_available/test_3hop.jsonl",
            shortest_output,
        )

    def test_filter_score_cache_selects_three_hop_samples_in_source_order(self):
        cache = {
            "version": 1,
            "meta": {"dataset": "MetaQA", "split": "test", "num_samples": 4},
            "samples": [
                _cache_sample("one", 1),
                _cache_sample("three-a", 3),
                _cache_sample("two", 2),
                _cache_sample("three-b", 3),
            ],
        }

        filtered, manifest = filter_score_cache_by_hop(cache, hop=3, split="test_3hop")

        self.assertEqual([sample["question"] for sample in filtered["samples"]], ["three-a", "three-b"])
        self.assertEqual(filtered["meta"]["split"], "test_3hop")
        self.assertEqual(filtered["meta"]["num_samples"], 2)
        self.assertEqual(manifest["source_hop_counts"], {"1": 1, "2": 1, "3": 2})
        self.assertEqual(manifest["selected_hop"], 3)
        self.assertEqual(manifest["selected_samples"], 2)
        self.assertEqual(cache["meta"]["split"], "test")

    def test_filter_score_cache_rejects_missing_hop_labels(self):
        cache = {
            "version": 1,
            "meta": {"dataset": "MetaQA"},
            "samples": [{**_cache_sample("missing", 3), "hop": None}],
        }

        with self.assertRaisesRegex(ValueError, "缺少 hop"):
            filter_score_cache_by_hop(cache, hop=3, split="test_3hop")

    def test_build_p4_report_checks_alignment_and_computes_average_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "sample_index": 0,
                "question": "who",
                "topics": ["topic"],
                "hop": 3,
                "golden": ["answer"],
                "mmr_reason_paths": [
                    {"path": [["topic", "rel", "answer"]], "log_score": -0.1},
                    {"path": [["topic", "rel2", "other"]], "log_score": -0.2},
                ],
                "prediction": {"answer": 0.9},
            }
            result_paths = {}
            summary_paths = {}
            for method in ("sp", "score", "fixed", "adaptive"):
                result_path = root / f"{method}.jsonl"
                summary_path = root / f"{method}_summary.json"
                _write_jsonl(result_path, [row])
                summary_path.write_text(json.dumps({
                    "n": 1,
                    "path": {"overall": {
                        "n": 1,
                        "answer_hit": 1.0,
                        "top1_hit": 1.0,
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                        "relation_jaccard_diversity": 1.0,
                    }},
                }), encoding="utf-8")
                result_paths[method] = result_path
                summary_paths[method] = summary_path

            report = build_p4_report(
                result_paths,
                summary_paths,
                expected_samples=1,
                dataset_hop=3,
            )

            self.assertTrue(report["alignment"]["passed"])
            self.assertEqual(report["alignment"]["fields"], ["sample_index", "question", "golden"])
            self.assertEqual(report["methods"]["adaptive"]["average_paths"], 2.0)
            self.assertEqual(report["methods"]["adaptive"]["n"], 1)

            changed = dict(row)
            changed["question"] = "different"
            _write_jsonl(result_paths["fixed"], [changed])
            with self.assertRaisesRegex(ValueError, "题目顺序或 golden 不一致"):
                build_p4_report(
                    result_paths,
                    summary_paths,
                    expected_samples=1,
                    dataset_hop=3,
                )


if __name__ == "__main__":
    unittest.main()
