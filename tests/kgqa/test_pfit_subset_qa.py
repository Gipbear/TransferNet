"""pfit.subset_qa:按 hop 分层生成子集 qa 文件(dump_scores 直接以 --qa_file 使用)。

pickle 仅用于读写本测试自建的临时 .pt fixture(模拟 MetaQA_KB 预处理产物),无外部输入。
"""
import json
import os
import pickle
import tempfile
import unittest

import numpy as np


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


def _pt_inputs():
    """MetaQA_KB 预处理 .pt 形态:questions/topic_entities/answers/hops,hop1/2/3=30/20/10。"""
    hops = np.array([1] * 30 + [2] * 20 + [3] * 10)
    n = len(hops)
    questions = np.arange(n * 4).reshape(n, 4)          # 每行唯一,便于对齐校验
    topic_entities = np.arange(n).reshape(n, 1) + 1000
    answers = np.arange(n * 2).reshape(n, 2) + 5000
    return questions, topic_entities, answers, hops


class TestSubsetQuestionPt(unittest.TestCase):
    def _run(self, n, seed=42):
        from kgqa.pfit.subset_qa import make_subset
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "train.pt")
            dst = os.path.join(d, "train_sub.pt")
            with open(src, "wb") as f:
                for arr in _pt_inputs():
                    pickle.dump(arr, f)
            make_subset(src, dst, n=n, seed=seed)
            out = []
            with open(dst, "rb") as f:
                for _ in range(4):
                    out.append(pickle.load(f))
            return out

    def test_stratified_counts_and_alignment(self):
        q, te, a, h = self._run(30)
        hops = [int(x) for x in h]
        self.assertEqual(len(q), 30)
        self.assertEqual(hops.count(1), 15)
        self.assertEqual(hops.count(2), 10)
        self.assertEqual(hops.count(3), 5)
        # 四数组同索引切片:凭 questions 行值反推源行号,逐行核对其余三个数组
        src_q, src_te, src_a, src_h = _pt_inputs()
        for row in range(len(q)):
            src_idx = int(q[row][0]) // 4
            np.testing.assert_array_equal(q[row], src_q[src_idx])
            np.testing.assert_array_equal(te[row], src_te[src_idx])
            np.testing.assert_array_equal(a[row], src_a[src_idx])
            self.assertEqual(int(h[row]), int(src_h[src_idx]))

    def test_deterministic_with_seed(self):
        a1 = self._run(12, seed=7)
        a2 = self._run(12, seed=7)
        b = self._run(12, seed=8)
        for x, y in zip(a1, a2):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        self.assertFalse(all(np.array_equal(np.asarray(x), np.asarray(y))
                             for x, y in zip(a1, b)))

    def test_n_exceeding_total_keeps_all(self):
        q, _, _, _ = self._run(999)
        self.assertEqual(len(q), 60)


if __name__ == "__main__":
    unittest.main()
