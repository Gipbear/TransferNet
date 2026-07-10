import os
import tempfile
import unittest

CKPT = "data/ckpt/MetaQA_KB/model_epoch-6_acc-0.9937.pt"
INPUT_DIR = "data/input/MetaQA_KB"
TEST_PT = "data/input/MetaQA_KB/test.pt"


class TestGoldStringsNameKey(unittest.TestCase):
    def test_name_key_maps_int_id_via_id2ent(self):
        # gold_key=name 时整数 gold_id 须先经 id2ent 还原实体名（与 pred 同口径）
        from types import SimpleNamespace
        from kgqa.cli.eval import _gold_strings

        class _IdentityAdapter:
            def entity_name(self, e):
                return e

        sample = SimpleNamespace(gold_ids=[174])
        id2ent = {174: "Before the Rain"}
        gold = _gold_strings(sample, _IdentityAdapter(), id2ent, "name")
        self.assertEqual(gold, {"Before the Rain"})


@unittest.skipUnless(os.path.isfile(CKPT) and os.path.isfile(TEST_PT), "ckpt/数据缺失，跳过")
class TestMetaQAEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kgqa.cli.dump_scores import main as dump_main
        cls.cache = os.path.join(tempfile.mkdtemp(), "metaqa_small.pt")
        dump_main(["--dataset", "metaqa", "--ckpt", CKPT, "--input_dir", INPUT_DIR,
                   "--qa_file", TEST_PT, "--output", cls.cache,
                   "--per_hop_limit", "3", "--batch_size", "64"])

    def _offline(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.retrieve.backends.offline import OfflineBackend
        adapter = get_adapter("metaqa", input_dir=INPUT_DIR)
        return OfflineBackend(adapter, cache_path=self.cache)

    def test_offline_retrieves_paths_all_hops(self):
        backend = self._offline()
        results = backend.retrieve_all()
        self.assertTrue(results)
        # 覆盖 3 个 hop，且 3-hop 样本能产出路径（验证 engine 在 3 跳下工作）
        by_hop = {}
        for r, s in zip(results, backend.bundle.samples):
            by_hop.setdefault(s.hop, []).append(r)
        self.assertEqual(sorted(by_hop), [1, 2, 3])
        self.assertTrue(any(r.paths for r in by_hop[3]))

    def test_answer_eval_by_hop(self):
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
        self.assertEqual(set(summary["by_hop"]), {"1", "2", "3"})
        self.assertIn("hit1", summary["overall"])
        # 口径一致时 hit1 不应为 0（模型 acc 0.99；全 0 说明 gold/pred 口径不齐）
        self.assertGreater(summary["overall"]["hit1"], 0.0)

    def test_online_offline_parity_first3(self):
        from kgqa.datasets.registry import get_adapter
        from kgqa.models.metaqa import MetaQAScoreProducer
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = get_adapter("metaqa", input_dir=INPUT_DIR)
        online = OnlineBackend(adapter, MetaQAScoreProducer(per_hop_limit=3),
                               ckpt_path=CKPT, input_dir=INPUT_DIR, qa_file=TEST_PT,
                               batch_size=64, limit=0)
        off = self._offline()
        for idx in range(3):
            ro = online.retrieve(idx)
            rf = off.retrieve(idx)
            self.assertEqual([p["path"] for p in ro.paths], [p["path"] for p in rf.paths])


if __name__ == "__main__":
    unittest.main()
