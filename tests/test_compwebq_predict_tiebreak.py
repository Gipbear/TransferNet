"""最高分并列的打破规则测试。

elem 钳位把所有 >1 的分数压成精确的 1.0,12%~15% 的样本因此在最高分并列,
torch.max 返回下标最小者,argmax 退化成「按实体 id 取第一个」。
model.py 推理时另留了一份未钳位的 e_score_raw(汇入路径越多分越高),用它做 tie-break。
"""
import unittest

import torch

from CompWebQ.predict import argmax_with_tiebreak


class TestArgmaxWithTiebreak(unittest.TestCase):
    def test_no_tie_matches_plain_argmax(self):
        e_score = torch.tensor([[0.2, 0.9, 0.4]])
        e_raw = torch.tensor([[5.0, 1.0, 3.0]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [1], "无并列时 raw 不得改变结果")

    def test_tie_broken_by_raw_not_by_entity_id(self):
        # 实体 0 和 2 都被钳位成 1.0;raw 显示实体 2 汇入更多路径
        e_score = torch.tensor([[1.0, 0.3, 1.0]])
        e_raw = torch.tensor([[2.5, 0.3, 6.0]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [2], "并列时应取 raw 更高者,而非 id 更小者")

    def test_plain_argmax_would_pick_first_on_tie(self):
        """对照:说明不做 tie-break 时确实退化成取 id 最小。"""
        e_score = torch.tensor([[1.0, 0.3, 1.0]])
        self.assertEqual(e_score.argmax(dim=1).tolist(), [0])

    def test_none_raw_falls_back_to_plain_argmax(self):
        e_score = torch.tensor([[1.0, 0.3, 1.0]])
        idx = argmax_with_tiebreak(e_score, None)
        self.assertEqual(idx.tolist(), [0])

    def test_batch_mixes_tied_and_untied_rows(self):
        e_score = torch.tensor([[1.0, 0.3, 1.0],
                                [0.2, 0.7, 0.1]])
        e_raw = torch.tensor([[2.5, 0.3, 6.0],
                              [9.0, 0.7, 0.1]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [2, 1], "未并列的行不受 raw 影响")

    def test_raw_also_tied_falls_back_to_lowest_id(self):
        """raw 也并列时结果必须确定,取下标最小者。"""
        e_score = torch.tensor([[1.0, 1.0, 0.2]])
        e_raw = torch.tensor([[4.0, 4.0, 0.2]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [0])

    def test_non_top_entity_never_selected_even_with_huge_raw(self):
        """raw 只在并列内部起作用:钳位后不是最高分的实体不得被 raw 顶上来。"""
        e_score = torch.tensor([[1.0, 0.99, 0.2]])
        e_raw = torch.tensor([[1.2, 99.0, 0.2]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [0])

    def test_float_noise_within_tolerance_counts_as_tie(self):
        """hop_attn 加权后并列值未必逐位相等,容差内应视为并列。"""
        e_score = torch.tensor([[1.0, 1.0 - 1e-9, 0.2]])
        e_raw = torch.tensor([[2.0, 8.0, 0.2]])
        idx = argmax_with_tiebreak(e_score, e_raw)
        self.assertEqual(idx.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
