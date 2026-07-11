import os
import tempfile
import unittest

import torch

CKPT = "data/ckpt/CWQ/model-29-0.4206.pt"
INPUT_DIR = "data/input/CWQ"
QA_FILE = "data/input/CWQ/test_simple.json"


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(QA_FILE), "ckpt/数据缺失，跳过")
class TestCWQEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgqa.cli.dump_scores import main as dump_main
        cls.cache = os.path.join(tempfile.mkdtemp(), "cwq_small.pt")
        dump_main(["--dataset", "cwq", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
                   "--qa_file", QA_FILE, "--output", cls.cache, "--limit", "20"])

    def _offline(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        adapter = get_adapter("cwq", input_dir=INPUT_DIR)
        return OfflineBackend(adapter, cache_path=self.cache)

    def test_cache_contains_triples(self):
        cache = torch.load(self.cache, weights_only=False)
        self.assertEqual(len(cache["samples"]), 20)
        self.assertTrue(all(s.get("triples") for s in cache["samples"]))

    def test_offline_retrieves_paths(self):
        backend = self._offline()
        results = backend.retrieve_all()
        self.assertEqual(len(results), len(backend.bundle.samples))
        self.assertTrue(any(r.paths for r in results))

    def test_answer_eval_hit1_positive(self):
        from kgqa.cli.eval import _gold_strings
        from kgqa.eval.answer_eval import answer_record, answer_summary
        backend = self._offline()
        adapter = backend.adapter
        spec = adapter.metric_spec()
        id2ent = backend.bundle.meta.id2ent
        results = backend.retrieve_all()
        records = []
        for r, s in zip(results, backend.bundle.samples):
            gold = _gold_strings(s, adapter, id2ent, spec.gold_key)
            records.append(answer_record(pred=list(r.prediction.keys()),
                                         gold=sorted(gold), hop=s.hop))
        summary = answer_summary(records, spec)
        self.assertIn("hit1", summary["overall"])
        # 口径一致时 hit1 不应为 0（ckpt acc 0.42，20 条全 miss 概率约 1.8e-5）
        self.assertGreater(summary["overall"]["hit1"], 0.0)

    def test_online_offline_parity_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.models.cwq import CWQScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = get_adapter("cwq", input_dir=INPUT_DIR)
        online = OnlineBackend(adapter, CWQScoreProducer(limit=20), ckpt_path=CKPT,
                               input_dir=INPUT_DIR, qa_file=QA_FILE)
        off = self._offline()
        for idx in range(3):
            ro = online.retrieve(idx)
            rf = off.retrieve(idx)
            self.assertEqual([p["path"] for p in ro.paths], [p["path"] for p in rf.paths])


if __name__ == "__main__":
    unittest.main()
