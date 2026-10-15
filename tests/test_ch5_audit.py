"""第五章CPU核对的覆盖增量和不一致拒绝测试。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.ch5.audit import BenchmarkAudit


class Ch5AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.qa = self.root / "qa.txt"
        self.qa.write_text("question [topic]\tanswer\n", encoding="utf-8")
        self.record = {
            "sample_index": 0, "question": "question", "gold_mids": ["answer"],
            "pred_answer_disambiguated_mids": ["answer"], "citation_accuracy": 1.0,
            "citation_recall": 1.0, "hallucination_rate": 0.0, "format_ok": True,
            "raw_mmr_reason_paths": [{"path": [["topic", "relation", "noise"]]}] * 20
            + [{"path": [["topic", "relation", "answer"]]}],
        }
        self.trace = self.root / "checked_batch_eval.jsonl"
        self.trace.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        summary = dict.fromkeys([
            "hit1", "hit_any", "macro_p", "macro_r", "macro_f1", "micro_p", "micro_r", "micro_f1",
            "exact_match", "citation_accuracy", "citation_recall", "format_compliance",
        ], 1.0)
        summary.update(n=1, hallucination_rate=0.0)
        (self.root / "checked_batch_eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.output = self.root / "result.json"

    def test_later_path_rescues_gold_outside_first_twenty(self):
        with contextlib.redirect_stdout(io.StringIO()):
            BenchmarkAudit(self.root, self.qa).run(self.output)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["coverage"]["20"]["hit_n"], 0)
        self.assertEqual(result["coverage"]["50"]["hit_n"], 1)
        self.assertEqual(result["rescued_sample_indices"], [0])

    def test_duplicate_sample_is_rejected_before_writing_summary(self):
        with self.trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.record) + "\n")
        with self.assertRaisesRegex(ValueError, "重复sample_index"):
            BenchmarkAudit(self.root, self.qa).run(self.output)
        self.assertFalse(self.output.exists())

    def test_changed_prediction_rejects_stale_summary(self):
        self.record["pred_answer_disambiguated_mids"] = ["wrong"]
        self.trace.write_text(json.dumps(self.record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "指标重算与原summary不一致"):
            BenchmarkAudit(self.root, self.qa).run(self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
