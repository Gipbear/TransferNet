"""predict.py 的关系词表一致性校验。

原 main 调 load_data 时未传 add_rev(默认 False),配合 load_state_dict(strict=False),
用 --rev 训练的 ckpt(关系数 13298)配上未扩展的词表(6649)时,关系分类器会保持
随机初始化且**不报错**,得分全是噪声。

历史上未触发:唯一调用 main 的 scripts/run_grid.sh 用的是非 rev 的旧 CWQ ckpt(6649),
恰好与默认值匹配。此处补断言防止将来踩中。
"""
import unittest

import torch

from CompWebQ.predict import assert_relation_vocab


class TestAssertRelationVocab(unittest.TestCase):
    def test_id_head_matching_vocab_passes(self):
        state = {"rel-way_0.weight": torch.zeros(6649, 768)}
        assert_relation_vocab(state, 6649, rev=False)  # 不抛错即通过

    def test_id_head_mismatch_raises(self):
        state = {"rel-way_0.weight": torch.zeros(13298, 768)}
        with self.assertRaises(ValueError) as ctx:
            assert_relation_vocab(state, 6649, rev=False)
        self.assertIn("13298", str(ctx.exception))
        self.assertIn("6649", str(ctx.exception))

    def test_error_message_hints_correct_direction(self):
        """ckpt 比词表大 → 应提示加上 --rev;反之提示去掉。"""
        big = {"rel-way_0.weight": torch.zeros(13298, 768)}
        with self.assertRaises(ValueError) as ctx:
            assert_relation_vocab(big, 6649, rev=False)
        self.assertIn("加上 --rev", str(ctx.exception))

        small = {"rel-way_0.weight": torch.zeros(6649, 768)}
        with self.assertRaises(ValueError) as ctx:
            assert_relation_vocab(small, 13298, rev=True)
        self.assertIn("去掉 --rev", str(ctx.exception))

    def test_unknown_structure_is_tolerated(self):
        """认不出关系头时不阻断加载,保持对未来结构的兼容。"""
        assert_relation_vocab({"something.else": torch.zeros(3)}, 6649, rev=False)


if __name__ == "__main__":
    unittest.main()
