"""pfit.eval 干预模式:路径筛选与编号映射(引用因果验证实验)。"""
import json
import tempfile
import unittest
from pathlib import Path

from kgqa.pfit import eval as pfit_eval


def _make_paths(n):
    """构造 n 条仅含单边的假路径,log_score 递减。"""
    return [{"path": [[f"e{i}", f"r{i}", f"e{i+1}"]], "log_score": -float(i)}
            for i in range(n)]


PATHS5 = _make_paths(5)


class TestApplyIntervention(unittest.TestCase):
    def test_keep_cited(self):
        kept, idx = pfit_eval.apply_intervention(PATHS5, "keep_cited", {2, 4}, "q")
        self.assertEqual(idx, [2, 4])
        self.assertEqual([p["log_score"] for p in kept], [-1.0, -3.0])

    def test_drop_cited(self):
        kept, idx = pfit_eval.apply_intervention(PATHS5, "drop_cited", {2, 4}, "q")
        self.assertEqual(idx, [1, 3, 5])
        self.assertEqual(len(kept), 3)

    def test_drop_uncited_matched_drops_only_uncited(self):
        cited = {2, 4}
        kept, idx = pfit_eval.apply_intervention(PATHS5, "drop_uncited_matched", cited, "q")
        self.assertEqual(len(kept), 3)
        self.assertTrue(set(idx) >= cited)

    def test_drop_uncited_matched_deterministic(self):
        a = pfit_eval.apply_intervention(PATHS5, "drop_uncited_matched", {2}, "question")
        b = pfit_eval.apply_intervention(PATHS5, "drop_uncited_matched", {2}, "question")
        self.assertEqual(a, b)

    def test_drop_uncited_matched_cited_exceeds_uncited(self):
        # 引用 4 条、未引用仅 1 条:k=1 删光未引用,等价 keep_cited
        kept, idx = pfit_eval.apply_intervention(PATHS5, "drop_uncited_matched",
                                                 {1, 2, 3, 4}, "q")
        self.assertEqual(idx, [1, 2, 3, 4])

    def test_cited_empty(self):
        self.assertEqual(pfit_eval.apply_intervention(PATHS5, "keep_cited", set(), "q")[1], [])
        self.assertEqual(pfit_eval.apply_intervention(PATHS5, "drop_cited", set(), "q")[1],
                         [1, 2, 3, 4, 5])
        self.assertEqual(pfit_eval.apply_intervention(
            PATHS5, "drop_uncited_matched", set(), "q")[1], [1, 2, 3, 4, 5])

    def test_no_intervention_passthrough(self):
        kept, idx = pfit_eval.apply_intervention(PATHS5, None, set(), "q")
        self.assertEqual(idx, [1, 2, 3, 4, 5])
        self.assertIs(kept[0], PATHS5[0])

    def test_unknown_intervention_raises(self):
        with self.assertRaises(ValueError):
            pfit_eval.apply_intervention(PATHS5, "bogus", set(), "q")


class TestLoadCiteSrc(unittest.TestCase):
    def test_aligns_by_sample_index(self):
        lines = [
            {"sample_index": 1, "cited_indices": [2, 4]},
            {"sample_index": 7, "cited_indices": [1]},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "predictions.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
            cite_map = pfit_eval.load_cite_src(str(p))
        self.assertEqual(cite_map, {1: {2, 4}, 7: {1}})

    def test_missing_sample_index_skipped(self):
        lines = [{"cited_indices": [1]}, {"sample_index": 3, "cited_indices": [2]}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "predictions.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
            cite_map = pfit_eval.load_cite_src(str(p))
        self.assertEqual(cite_map, {3: {2}})


if __name__ == "__main__":
    unittest.main()
