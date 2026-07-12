import os
import tempfile
import unittest

import torch

from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


class TestDumpBundleHopField(unittest.TestCase):
    def test_bundle_to_cache_writes_hop(self):
        # 纯函数测试，无需 ckpt
        from kgqa.cli.dump_scores import _bundle_to_cache
        from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle
        s = SampleScore(question="q", topic_ids=[1], gold_ids=[1],
                        hop_attn=torch.tensor([1.0, 0.0, 0.0]),
                        rel_probs=[torch.tensor([0.0])], ent_indices=[torch.tensor([1])],
                        ent_scores=[torch.tensor([0.5])],
                        e_score_indices=torch.tensor([1]), e_score_values=torch.tensor([0.9]),
                        sample_index=0, hop=2)
        meta = CacheMeta(dataset="MetaQA", split="test", id2ent={}, id2rel={}, num_samples=1)
        cache = _bundle_to_cache(ScoreBundle(meta=meta, samples=[s]))
        self.assertEqual(cache["samples"][0]["hop"], 2)


@unittest.skipUnless(artifact_test_available(CKPT, TEST_PT), ARTIFACT_TEST_SKIP_REASON)
class TestDumpMetaQAEndToEnd(unittest.TestCase):
    def test_dump_metaqa_small(self):
        from kgqa.cli.dump_scores import main
        out = os.path.join(tempfile.mkdtemp(), "metaqa_small.pt")
        main(["--dataset", "metaqa", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
              "--qa_file", TEST_PT, "--output", out, "--per_hop_limit", "2",
              "--batch_size", "64"])
        cache = torch.load(out, weights_only=False)
        self.assertEqual(cache["meta"]["dataset"], "MetaQA")
        self.assertTrue(all("hop" in s for s in cache["samples"]))
        self.assertEqual(sorted({s["hop"] for s in cache["samples"]}), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
