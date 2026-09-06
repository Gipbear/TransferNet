"""pfit.eval 指标纯函数:与 llm_infer.eval_faithfulness 对拍 + by_hop 分组。"""
import random
import sys
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT / "llm_infer"))
import eval_faithfulness as legacy  # noqa: E402


_RAW_OUTPUTS = {
    "v1": ["Answer: m.01 | m.02", "garbage no answer line", "Answer: entity1"],
    "v2": ["Supporting Paths: 1, 3\nAnswer: Foo | Bar",
           "Answer: Foo"],
    "v3": ['{"reasoning": ["Path 1", "Path 3"], "answer": ["Foo", "Foo", "Bar"]}',
           "not json at all\nAnswer: Baz"],
    "v4": ["Reasoning: paths 2 lead there.\nSupporting Paths: 2\nAnswer: Qux",
           "Supporting Paths: 2\nAnswer: Qux"],
    "v11": ["[Reasoning]\n1 → Foo via [r]\n\n[Answer]\nSupporting Paths: 1\nAnswer: Foo"],
}


def _fake_results(n=20, seed=3):
    rng = random.Random(seed)
    results = []
    for i in range(n):
        hit = rng.random() < 0.6
        results.append({
            "hop": (i % 3) + 1,
            "hit1": int(hit), "hit_any": int(hit),
            "precision": rng.random(), "recall": rng.random(), "f1": rng.random(),
            "exact_match": hit, "tp": int(hit), "pred_n": 1 + i % 2, "gold_n": 1 + i % 3,
            "citation_accuracy": rng.random(), "citation_recall": rng.random(),
            "hallucination_rate": 0.0, "format_ok": True,
            "mmr_answer_path_hit": hit, "is_rejection": not hit and i % 2 == 0,
        })
    return results


class TestParseOutputParity(unittest.TestCase):
    def test_all_formats_match_legacy(self):
        from kgqa.pfit import eval as pfit_eval
        for fmt, raws in _RAW_OUTPUTS.items():
            for raw in raws:
                with self.subTest(fmt=fmt, raw=raw[:30]):
                    a = legacy.parse_output(raw, fmt)
                    b = pfit_eval.parse_output(raw, fmt)
                    self.assertEqual(a, b)

    def test_rejection_detection_parity(self):
        from kgqa.pfit import eval as pfit_eval
        for parsed in ({"answers": ["(none)"]}, {"answers": ["Foo"]}, {"answers": []}):
            self.assertEqual(legacy.is_rejection_response(parsed),
                             pfit_eval.is_rejection_response(parsed))

    def test_v2_rejection_normalizes_to_none(self):
        """拒答规范形式改为 None;旧 (none) 仍识别并归一化到 None。"""
        from kgqa.pfit import eval as pfit_eval
        for raw in ("Supporting Paths: None\nAnswer: None",
                    "Supporting Paths: (none)\nAnswer: (none)"):
            with self.subTest(raw=raw):
                parsed = pfit_eval.parse_output(raw, "v2")
                self.assertEqual(parsed["answers"], ["None"])
                self.assertTrue(parsed["format_ok"])
                self.assertTrue(pfit_eval.is_rejection_response(parsed))


class TestMetricsParity(unittest.TestCase):
    def test_answer_metrics_parity(self):
        from kgqa.pfit import eval as pfit_eval
        cases = [
            (["Foo", "Bar"], ["foo"]),
            ([], ["x"]),
            (["a"], []),
            ([], []),
            (["A", "B"], ["a", "b"]),
        ]
        for pred, gold in cases:
            with self.subTest(pred=pred, gold=gold):
                self.assertEqual(legacy.compute_answer_metrics(pred, gold),
                                 pfit_eval.compute_answer_metrics(pred, gold))

    def test_faithfulness_parity(self):
        from kgqa.pfit import eval as pfit_eval
        cases = [
            ({1, 2}, {1}, ["Foo"], {"foo"}),
            (set(), {1}, ["Bar"], {"foo"}),
            ({3}, set(), ["(none)"], set()),
        ]
        for cited, golden_idx, pred, ents in cases:
            self.assertEqual(
                legacy.compute_faithfulness(cited, golden_idx, pred, ents),
                pfit_eval.compute_faithfulness(cited, golden_idx, pred, ents))

    def test_faithfulness_accepts_truncated_path_entity_prefix(self):
        from kgqa.core.answer_metrics import compute_faithfulness as core_compute
        from kgqa.pfit import eval as pfit_eval

        path_entities = {"andrea del verrocchio", "unrelated entity"}
        for compute in (core_compute, pfit_eval.compute_faithfulness):
            with self.subTest(module=compute.__module__):
                metrics = compute(set(), set(), ["Andrea del Verroc"], path_entities)
                self.assertEqual(metrics["hallucination_rate"], 0.0)
                self.assertEqual(metrics["hallucinated_entities"], [])

    def test_faithfulness_rejects_answer_extending_path_entity(self):
        from kgqa.core.answer_metrics import compute_faithfulness as core_compute
        from kgqa.pfit import eval as pfit_eval

        path_entities = {"andrea del verroc"}
        for compute in (core_compute, pfit_eval.compute_faithfulness):
            with self.subTest(module=compute.__module__):
                metrics = compute(set(), set(), ["Andrea del Verrocchio"], path_entities)
                self.assertEqual(metrics["hallucination_rate"], 1.0)
                self.assertEqual(metrics["hallucinated_entities"], ["Andrea del Verrocchio"])

    def test_aggregate_and_rejection_parity(self):
        from kgqa.pfit import eval as pfit_eval
        results = _fake_results()
        self.assertEqual(legacy.aggregate(results), pfit_eval.aggregate(results))
        self.assertEqual(legacy.compute_rejection_metrics(results),
                         pfit_eval.compute_rejection_metrics(results))

    def test_expand_pred_parity(self):
        from kgqa.pfit import eval as pfit_eval
        rev = {"foo": {"m.1", "m.2"}, "bar": {"m.3"}}
        path_mids = {"m.2"}
        self.assertEqual(
            legacy.expand_pred_answers_with_path_constraint(["Foo", "Bar", "Zzz"], rev, path_mids),
            pfit_eval.expand_pred_answers_with_path_constraint(["Foo", "Bar", "Zzz"], rev, path_mids))


class TestByHop(unittest.TestCase):
    def test_summarize_groups_by_hop(self):
        from kgqa.pfit import eval as pfit_eval
        results = _fake_results(21)
        summary = pfit_eval.summarize(results, group_by_hop=True)
        self.assertEqual(summary["overall"], pfit_eval.aggregate(results))
        self.assertEqual(set(summary["by_hop"]), {"1", "2", "3"})
        for hop_key, m in summary["by_hop"].items():
            group = [r for r in results if str(r["hop"]) == hop_key]
            self.assertEqual(m, pfit_eval.aggregate(group))

    def test_summarize_without_hop(self):
        from kgqa.pfit import eval as pfit_eval
        results = _fake_results(10)
        summary = pfit_eval.summarize(results, group_by_hop=False)
        self.assertEqual(summary["overall"], pfit_eval.aggregate(results))
        self.assertNotIn("by_hop", summary)


if __name__ == "__main__":
    unittest.main()
