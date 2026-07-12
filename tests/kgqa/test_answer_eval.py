import unittest
from kgqa.retrieve.eval.answer_eval import answer_record, answer_summary
from kgqa.core.contracts import MetricSpec


class TestAnswerEval(unittest.TestCase):
    def _records(self):
        return [
            answer_record(pred=["a"], gold=["a"], hop=1, format_ok=True),
            answer_record(pred=["x"], gold=["b"], hop=1, format_ok=True),
            answer_record(pred=["c", "d"], gold=["c"], hop=2, format_ok=True),
        ]

    def test_overall_hit1(self):
        out = answer_summary(self._records(), MetricSpec())
        self.assertIn("overall", out)
        self.assertEqual(out["by_hop"], {})
        self.assertAlmostEqual(out["overall"]["hit1"], 2 / 3, places=4)

    def test_group_by_hop(self):
        out = answer_summary(self._records(), MetricSpec(group_by="hop"))
        self.assertEqual(set(out["by_hop"]), {"1", "2"})
        self.assertAlmostEqual(out["by_hop"]["1"]["hit1"], 0.5, places=4)
        self.assertAlmostEqual(out["by_hop"]["2"]["hit1"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
