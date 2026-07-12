import unittest
from kgqa.retrieve.eval.path_eval import path_record, path_summary
from kgqa.core.contracts import MetricSpec, RetrieveResult


def _result(idx, hop, tail):
    return RetrieveResult(
        question="q", topics=["m.t"], hop=hop,
        paths=[{"path": [["m.t", "r", tail]], "log_score": -0.1}],
        prediction={}, elapsed_ms=0.0, sample_index=idx,
    )


class TestPathEval(unittest.TestCase):
    def test_path_record_hit(self):
        rec = path_record(_result(0, 1, "m.gold"), {"m.gold"})
        self.assertEqual(rec["answer_hit"], 1)

    def test_path_summary_group_by_hop(self):
        results = [_result(0, 1, "m.gold"), _result(1, 2, "m.x")]
        gold = {0: {"m.gold"}, 1: {"m.gold"}}
        out = path_summary(results, gold, MetricSpec(group_by="hop"))
        self.assertEqual(set(out["by_hop"]), {"1", "2"})
        self.assertEqual(out["by_hop"]["1"]["answer_hit"], 1.0)
        self.assertEqual(out["by_hop"]["2"]["answer_hit"], 0.0)


if __name__ == "__main__":
    unittest.main()
