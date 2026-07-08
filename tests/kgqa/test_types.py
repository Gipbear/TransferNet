import unittest
from kgqa.types import QASample, ReasonPath, RetrieveResult, MetricSpec


class TestTypes(unittest.TestCase):
    def test_qasample_defaults(self):
        s = QASample(question="q", topic_ids=[1], gold_ids=[2, 3])
        self.assertEqual(s.sample_index, -1)
        self.assertIsNone(s.hop)
        self.assertEqual(s.extra, {})

    def test_reasonpath_to_triples(self):
        p = ReasonPath(nodes=[10, 11, 12], rels=[5, 6], score=-1.5)
        id2ent = {10: "m.a", 11: "m.b", 12: "m.c"}
        id2rel = {5: "r1", 6: "r2"}
        self.assertEqual(
            p.to_triples(id2ent, id2rel),
            [["m.a", "r1", "m.b"], ["m.b", "r2", "m.c"]],
        )

    def test_reasonpath_to_triples_missing_id_falls_back_to_str(self):
        p = ReasonPath(nodes=[10, 99], rels=[5], score=0.0)
        self.assertEqual(p.to_triples({10: "m.a"}, {}), [["m.a", "5", "99"]])

    def test_metricspec_defaults(self):
        spec = MetricSpec()
        self.assertEqual(spec.gold_key, "mid")
        self.assertIsNone(spec.group_by)

    def test_retrieve_result_holds_paths(self):
        r = RetrieveResult(question="q", topics=["m.a"], hop=1,
                           paths=[{"path": [["m.a", "r", "m.b"]], "log_score": -0.1}],
                           prediction={"m.b": 0.9}, elapsed_ms=1.2)
        self.assertEqual(r.paths[0]["log_score"], -0.1)


if __name__ == "__main__":
    unittest.main()
