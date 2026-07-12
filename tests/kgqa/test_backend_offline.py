import unittest

from kgqa.datasets.registry import get_adapter
from kgqa.retrieve.backends.offline import OfflineBackend
from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"


class TestOfflineBackend(unittest.TestCase):
    @unittest.skipUnless(artifact_test_available(CACHE), ARTIFACT_TEST_SKIP_REASON)
    def test_retrieve_single(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        backend = OfflineBackend(adapter, cache_path=CACHE)
        r = backend.retrieve(0, beam_size=50, lambda_val=0.2)
        self.assertEqual(r.sample_index, 0)
        self.assertGreaterEqual(len(r.paths), 1)
        self.assertTrue(all("path" in p and "log_score" in p for p in r.paths))

    @unittest.skipUnless(artifact_test_available(CACHE), ARTIFACT_TEST_SKIP_REASON)
    def test_retrieve_all_len(self):
        adapter = get_adapter("webqsp", input_dir="data/input/WebQSP")
        backend = OfflineBackend(adapter, cache_path=CACHE)
        results = backend.retrieve_all(beam_size=10, limit=5)
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
