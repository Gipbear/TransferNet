import unittest

from kgqa.scores.webqsp import WebQSPScoreLoader
from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestWebQSPScoreLoader(unittest.TestCase):
    @unittest.skipUnless(artifact_test_available(CACHE), ARTIFACT_TEST_SKIP_REASON)
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
