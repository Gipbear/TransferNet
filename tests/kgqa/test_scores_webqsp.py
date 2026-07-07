import os
import unittest

from kgqa.scores.webqsp import WebQSPScoreLoader

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestWebQSPScoreLoader(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CACHE), "缓存缺失，跳过")
    def test_load_bundle(self):
        bundle = WebQSPScoreLoader().load(CACHE)
        self.assertEqual(bundle.meta.dataset, "WebQSP")
        self.assertEqual(bundle.meta.num_samples, len(bundle.samples))
        self.assertGreater(len(bundle.samples), 0)
        s = bundle.samples[0]
        self.assertIsInstance(s.question, str)
        self.assertTrue(hasattr(s, "hop_attn"))
        self.assertTrue(hasattr(s, "e_score_values"))
        self.assertEqual(s.sample_index, 0)


if __name__ == "__main__":
    unittest.main()
