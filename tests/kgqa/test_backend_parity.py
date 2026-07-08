import os
import unittest

CKPT = "data/ckpt/WebQSP_run_20260518_2241/model-49-0.7154.pt"
CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"
QA = "data/input/WebQSP/QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(CACHE) and os.path.isfile(QA),
                     "ckpt/缓存/QA 缺失，跳过")
class TestBackendParity(unittest.TestCase):
    def test_online_matches_offline_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        from kgqa.models.webqsp import WebQSPScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend

        params = dict(beam_size=50, method="tail_blend", lambda_val=0.2,
                      threshold=0.01, alpha_final=1.0)
        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        offline = OfflineBackend(adapter, cache_path=CACHE)
        online = OnlineBackend(adapter, WebQSPScoreProducer(), ckpt_path=CKPT,
                               input_dir=INPUT_DIR, qa_file=QA, split="test", limit=3)

        for i in range(3):
            ro = offline.retrieve(i, **params)
            rn = online.retrieve(i, **params)
            rels_o = [[e[1] for e in p["path"]] for p in ro.paths]
            rels_n = [[e[1] for e in p["path"]] for p in rn.paths]
            self.assertEqual(rels_n, rels_o, f"sample {i} online/offline rels 不一致")


if __name__ == "__main__":
    unittest.main()
