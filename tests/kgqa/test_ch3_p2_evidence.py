import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestCh3P2Evidence(unittest.TestCase):
    def test_p2_config_includes_completed_fixed_qa_comparison(self):
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "experiments/configs/ch3/webqsp_transfernet_v1_p2.json").read_text(
                encoding="utf-8"
            )
        )
        statistics = config["statistics"]

        self.assertIn("fixed", statistics["qa_inputs"])
        comparison = next(
            item for item in statistics["comparisons"]
            if item["id"] == "qa_adaptive_vs_fixed"
        )
        self.assertEqual(comparison["family"], "qa")
        self.assertEqual(comparison["left"], "tarrs")
        self.assertEqual(comparison["right"], "fixed")
        self.assertEqual(comparison["metrics"], ["macro_f1", "hit1"])
        self.assertEqual(statistics.get("pending_comparisons", []), [])

    def test_paired_bootstrap_is_deterministic_and_keeps_pairing(self):
        from experiments.ch3.p2_evidence import paired_bootstrap_interval

        left = np.array([1.0, 1.0, 0.0, 1.0])
        right = np.array([0.0, 1.0, 0.0, 0.0])
        first = paired_bootstrap_interval(
            left, right, replicates=2000, confidence_level=0.95, seed=17,
        )
        second = paired_bootstrap_interval(
            left, right, replicates=2000, confidence_level=0.95, seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["n"], 4)
        self.assertAlmostEqual(first["left_mean"], 0.75)
        self.assertAlmostEqual(first["right_mean"], 0.25)
        self.assertAlmostEqual(first["difference"], 0.5)
        self.assertLessEqual(first["ci_low"], first["difference"])
        self.assertGreaterEqual(first["ci_high"], first["difference"])

    def test_path_records_are_aligned_by_sample_index_not_file_order(self):
        from experiments.ch3.p2_evidence import load_path_outcomes

        rows = [
            {
                "sample_index": 9,
                "question": "q9",
                "golden": ["gold"],
                "mmr_reason_paths": [
                    {"path": [["topic", "r", "other"]]},
                    {"path": [["topic", "r2", "gold"]]},
                ],
            },
            {
                "sample_index": 3,
                "question": "q3",
                "golden": ["gold"],
                "mmr_reason_paths": [{"path": [["topic", "r", "gold"]]}],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            outcomes = load_path_outcomes(path)

        self.assertEqual(list(outcomes), [3, 9])
        self.assertEqual(outcomes[3]["answer_hit"], 1.0)
        self.assertEqual(outcomes[3]["top1_hit"], 1.0)
        self.assertEqual(outcomes[9]["answer_hit"], 1.0)
        self.assertEqual(outcomes[9]["top1_hit"], 0.0)

    def test_path_outcomes_use_reverse_edge_sink_as_answer(self):
        from experiments.ch3.p2_evidence import load_path_outcomes

        row = {
            "sample_index": 0,
            "question": "sink",
            "golden": ["gold"],
            "mmr_reason_paths": [{
                "path": [
                    ["topic", "forward_relation", "gold"],
                    ["gold", "incoming_relation_reverse", "other"],
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            outcomes = load_path_outcomes(path)

        self.assertEqual(outcomes[0]["answer_hit"], 1.0)
        self.assertEqual(outcomes[0]["top1_hit"], 1.0)

    def test_alignment_rejects_question_mismatch(self):
        from experiments.ch3.p2_evidence import paired_metric_arrays

        left = {0: {"question": "left", "hit1": 1.0}}
        right = {0: {"question": "right", "hit1": 0.0}}

        with self.assertRaisesRegex(ValueError, "question"):
            paired_metric_arrays(left, right, "hit1")


if __name__ == "__main__":
    unittest.main()
