"""A1 增益归因的公共计算测试：Micro/Macro 口径与配对 bootstrap。"""

import unittest

from scripts.ch5_analysis_common import bootstrap_delta, bucketize, macro_f1, micro_f1


def make_row(index: int, tp_closed: int, tp_open: int, pred_open: int, closed_n: int,
             gold_n: int, group_key: str | None = None, group_value: float = 0.0) -> dict:
    def f1(tp: int, pred: int, gold: int) -> float:
        if not tp or not pred or not gold:
            return 0.0
        precision, recall = tp / pred, tp / gold
        return 2 * precision * recall / (precision + recall)

    row = {
        "sample_index": index,
        "tp_closed": tp_closed, "tp_open": tp_open, "pred_open": pred_open,
        "closed_n": closed_n, "gold_n": gold_n,
        "f1_closed": f1(tp_closed, closed_n, gold_n),
        "f1_open": f1(tp_open, pred_open, gold_n),
        "triggered": 1, "expanded_n": 1, "expanded_gold": 1,
    }
    if group_key is not None:
        row[group_key] = group_value
    return row


class TestMetrics(unittest.TestCase):
    def test_micro_f1_matches_hand_computation(self):
        rows = [make_row(0, 1, 2, 2, 1, 2), make_row(1, 0, 1, 2, 2, 2)]
        # 开补全：tp=3, pred=4, gold=4 -> P=R=0.75 -> F1=0.75
        self.assertAlmostEqual(micro_f1(rows, True), 0.75, places=9)
        # 关补全：tp=1, pred=3, gold=4 -> P=1/3, R=0.25 -> F1=0.2857
        self.assertAlmostEqual(micro_f1(rows, False), 2 * (1 / 3) * 0.25 / (1 / 3 + 0.25), places=9)

    def test_micro_f1_empty_is_zero(self):
        self.assertEqual(micro_f1([], True), 0.0)

    def test_macro_f1_is_sample_mean(self):
        rows = [make_row(0, 1, 2, 2, 1, 2), make_row(1, 0, 1, 2, 2, 2)]
        expected = sum(row["f1_open"] for row in rows) / 2
        self.assertAlmostEqual(macro_f1(rows, True), expected, places=9)


class TestBootstrap(unittest.TestCase):
    def test_point_estimates_match_direct_computation(self):
        rows = [make_row(i, i % 3, (i % 3) + 1, 3, 3, 4, group_value=float(i % 2)) for i in range(40)]
        stat = bootstrap_delta(rows, n_boot=200)
        self.assertEqual(stat["n"], 40)
        self.assertAlmostEqual(stat["macro"], macro_f1(rows, True) - macro_f1(rows, False), places=9)
        self.assertAlmostEqual(stat["micro"], micro_f1(rows, True) - micro_f1(rows, False), places=9)

    def test_identical_rows_give_zero_width_interval(self):
        """所有样本的答案完全相同时，任何重采样都不改变指标，区间宽度必须为 0。"""
        rows = [make_row(i, 2, 2, 2, 2, 2) for i in range(30)]
        stat = bootstrap_delta(rows, n_boot=200)
        self.assertAlmostEqual(stat["macro"], 0.0, places=12)
        self.assertAlmostEqual(stat["macro_lo"], 0.0, places=12)
        self.assertAlmostEqual(stat["macro_hi"], 0.0, places=12)
        self.assertAlmostEqual(stat["micro_hi"] - stat["micro_lo"], 0.0, places=12)

    def test_interval_brackets_point_estimate(self):
        rows = [make_row(i, i % 4, (i % 4) + 1, 5, 4, 6, group_key="v", group_value=float(i))
                for i in range(120)]
        stat = bootstrap_delta(rows, n_boot=500)
        self.assertLessEqual(stat["macro_lo"], stat["macro"])
        self.assertLessEqual(stat["macro"], stat["macro_hi"])
        self.assertLessEqual(stat["micro_lo"], stat["micro"])
        self.assertLessEqual(stat["micro"], stat["micro_hi"])

    def test_seed_is_reproducible(self):
        rows = [make_row(i, i % 3, (i % 3) + 1, 3, 3, 4, group_key="v", group_value=float(i))
                for i in range(50)]
        first = bootstrap_delta(rows, n_boot=200, seed=7)
        second = bootstrap_delta(rows, n_boot=200, seed=7)
        self.assertEqual(first, second)

    def test_single_row_interval_collapses(self):
        rows = [make_row(0, 1, 2, 2, 1, 2)]
        stat = bootstrap_delta(rows, n_boot=100)
        self.assertAlmostEqual(stat["macro_lo"], stat["macro"], places=12)
        self.assertAlmostEqual(stat["macro_hi"], stat["macro"], places=12)

    def test_gate_falls_back_to_closed_for_rejected_rows(self):
        """门控判否的样本必须回退到关补全口径，而不是被丢弃。"""
        rows = [make_row(0, 1, 3, 3, 1, 3),   # 开补全更好：f1 0.5 -> 1.0
                make_row(1, 2, 2, 5, 2, 2)]   # 开补全更差：f1 1.0 -> 0.571
        everywhere = bootstrap_delta(rows, n_boot=200)
        gated = bootstrap_delta(rows, n_boot=200, gate=lambda row: row["sample_index"] == 0)
        self.assertAlmostEqual(gated["n"], 2)
        # 门控版保留样本 0 的增益、屏蔽样本 1 的损失，故 ΔMacro 必须更高
        self.assertGreater(gated["macro"], everywhere["macro"])
        self.assertAlmostEqual(gated["macro"], (1.0 + 1.0) / 2 - (0.5 + 1.0) / 2, places=9)

    def test_gate_rejecting_all_equals_no_expansion(self):
        rows = [make_row(0, 1, 3, 3, 1, 3), make_row(1, 2, 2, 5, 2, 2)]
        stat = bootstrap_delta(rows, n_boot=200, gate=lambda row: False)
        self.assertAlmostEqual(stat["macro"], 0.0, places=9)
        self.assertAlmostEqual(stat["micro"], 0.0, places=9)


class TestBucketize(unittest.TestCase):
    def test_boundaries_are_half_open(self):
        rows = [{"v": 0.0}, {"v": 0.25}, {"v": 0.5}, {"v": 0.75}, {"v": 1.0}]
        result = bucketize(rows, "v", [0.25, 0.5, 0.75])
        self.assertEqual([label for label, _ in result], ["<0.25", "[0.25, 0.5)", "[0.5, 0.75)", "≥0.75"])
        self.assertEqual([len(members) for _, members in result], [1, 1, 1, 2])

    def test_empty_buckets_are_kept(self):
        rows = [{"v": 0.0}, {"v": 1.0}]
        result = bucketize(rows, "v", [0.25, 0.5, 0.75])
        self.assertEqual(len(result), 4)
        self.assertEqual([len(members) for _, members in result], [1, 0, 0, 1])

    def test_missing_key_is_treated_as_zero(self):
        result = bucketize([{}], "absent", [0.5])
        self.assertEqual([len(members) for _, members in result], [1, 0])


if __name__ == "__main__":
    unittest.main()
