import unittest

from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


@unittest.skipUnless(artifact_test_available(CKPT, TEST_PT), ARTIFACT_TEST_SKIP_REASON)
class TestMetaQAScoreProducer(unittest.TestCase):
    def test_produce_small_stratified(self):
        from kgqa.backbone.metaqa import MetaQAScoreProducer
        producer = MetaQAScoreProducer(per_hop_limit=2)
        producer.load_checkpoint(CKPT)
        bundle = producer.produce(INPUT_DIR, TEST_PT, split="test", batch_size=64, topk=500)
        # 每跳 2 条 → 覆盖三个 hop
        hops = sorted({s.hop for s in bundle.samples})
        self.assertEqual(hops, [1, 2, 3])
        s = bundle.samples[0]
        # 合成 hop_attn：argmax()+1 == gold hop
        self.assertEqual(int(s.hop_attn.argmax().item()) + 1, s.hop)
        self.assertEqual(len(s.rel_probs), 3)
        self.assertGreater(s.e_score_values.numel(), 0)
        self.assertIsInstance(s.question, str)
        self.assertTrue(bundle.meta.id2ent)


if __name__ == "__main__":
    unittest.main()
