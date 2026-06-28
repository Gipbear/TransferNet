import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_stop_policy_results


class StopPolicyCollectTests(unittest.TestCase):
    def test_collects_best_complete_policy_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = root / "sweep_a"
            sweep.mkdir()
            rows = [
                {
                    "policy": "bad",
                    "complete_support": "True",
                    "n": "2",
                    "unsupported_n": "0",
                    "macro_f1": "0.1",
                    "exact_match": "0.0",
                    "hit1": "0.0",
                    "avg_batches_used": "1.0",
                    "avg_final_answer_count": "1.0",
                },
                {
                    "policy": "best",
                    "complete_support": "True",
                    "n": "2",
                    "unsupported_n": "0",
                    "macro_f1": "0.9",
                    "exact_match": "0.8",
                    "hit1": "0.7",
                    "avg_batches_used": "1.5",
                    "avg_final_answer_count": "2.0",
                },
            ]
            with (sweep / "stop_policy_sweep_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            out = root / "out"

            exit_code = collect_stop_policy_results.main(
                [str(sweep), "--output_dir", str(out)]
            )

            self.assertEqual(exit_code, 0)
            text = (out / "stop_policy_compare.md").read_text(encoding="utf-8")
            self.assertIn("`best`", text)
            self.assertIn("0.9000", text)
            self.assertTrue((out / "stop_policy_compare.csv").exists())


if __name__ == "__main__":
    unittest.main()
