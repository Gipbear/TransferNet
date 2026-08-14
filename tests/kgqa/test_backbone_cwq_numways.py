"""CWQScoreProducer 对 num_ways>1 的守卫测试。

TransferNet.forward 在 way 循环内重新赋值 rel_probs/ent_probs，返回的只是最后一个
way 的分布（e_score 才是跨 way 的 torch.prod）。MMR beam search 吃的正是
rel_probs/ent_probs，num_ways>1 时它只能看到半个模型，得分缓存会静默失真。
"""
import unittest

from kgqa.backbone.cwq import CWQScoreProducer


class TestNumWaysGuard(unittest.TestCase):
    def test_num_ways_one_accepted(self):
        producer = CWQScoreProducer(num_ways=1)
        self.assertEqual(producer.num_ways, 1)

    def test_num_ways_default_is_one(self):
        self.assertEqual(CWQScoreProducer().num_ways, 1)

    def test_num_ways_two_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            CWQScoreProducer(num_ways=2)
        self.assertIn("num_ways", str(ctx.exception))

    def test_num_ways_message_mentions_mmr_input(self):
        """报错必须指出坏的是 MMR 的输入，否则使用者会以为只是性能问题。"""
        with self.assertRaises(ValueError) as ctx:
            CWQScoreProducer(num_ways=3)
        message = str(ctx.exception)
        self.assertIn("ent_probs", message)


if __name__ == "__main__":
    unittest.main()
