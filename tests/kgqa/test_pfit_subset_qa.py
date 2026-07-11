"""pfit.subset_qa:按 hop 分层生成子集 qa 文件(dump_scores 直接以 --qa_file 使用)。"""
import json
import os
import tempfile
import unittest


def _qa_items():
    # hop1/2/3 = 30/20/10
    items = []
    for hop, n in ((1, 30), (2, 20), (3, 10)):
        for i in range(n):
            items.append({"question": f"what does E_S do {hop}_{i}",
                          "topic_entity": f"Ent{hop}_{i}",
                          "answers": [f"Ans{hop}_{i}"],
                          "hop": hop})
    return items


class TestSubsetQA(unittest.TestCase):
    def _run(self, n, seed=42):
        from kgqa.pfit.subset_qa import make_subset
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "train.json")
            dst = os.path.join(d, "train_sub.json")
            with open(src, "w", encoding="utf-8") as f:
                json.dump(_qa_items(), f, ensure_ascii=False)
            make_subset(src, dst, n=n, seed=seed)
            with open(dst, encoding="utf-8") as f:
                return json.load(f)

    def test_stratified_counts(self):
        sub = self._run(30)
        hops = [x["hop"] for x in sub]
        self.assertEqual(len(sub), 30)
        self.assertEqual(hops.count(1), 15)
        self.assertEqual(hops.count(2), 10)
        self.assertEqual(hops.count(3), 5)

    def test_items_preserved_verbatim(self):
        src_by_q = {x["question"]: x for x in _qa_items()}
        sub = self._run(12)
        for item in sub:
            self.assertEqual(item, src_by_q[item["question"]])

    def test_deterministic_with_seed(self):
        self.assertEqual(self._run(12, seed=7), self._run(12, seed=7))
        self.assertNotEqual(self._run(12, seed=7), self._run(12, seed=8))

    def test_n_exceeding_total_keeps_all(self):
        sub = self._run(999)
        self.assertEqual(len(sub), 60)


if __name__ == "__main__":
    unittest.main()
