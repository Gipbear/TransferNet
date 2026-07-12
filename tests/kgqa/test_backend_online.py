import unittest

from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

CKPT = "data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt"
INPUT_DIR = "data/input/WebQSP"
QA = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"


class TestOnlineBackend(unittest.TestCase):
    @unittest.skipUnless(artifact_test_available(CKPT, QA), ARTIFACT_TEST_SKIP_REASON)
    def test_online_retrieve_smoke(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.backbone.webqsp import WebQSPScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend

        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        backend = OnlineBackend(
            adapter, WebQSPScoreProducer(), ckpt_path=CKPT,
            input_dir=INPUT_DIR, qa_file=QA, split="test",
        )
        r = backend.retrieve(0, beam_size=50)
        self.assertGreaterEqual(len(r.paths), 1)


if __name__ == "__main__":
    unittest.main()
