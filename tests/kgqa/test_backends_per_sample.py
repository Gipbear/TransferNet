import unittest
from unittest import mock

import torch

from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle


def _fake_bundle(n=2):
    samples = [
        SampleScore(
            question=f"q{i}", topic_ids=[0], gold_ids=[1],
            hop_attn=torch.tensor([1.0, 0.0]),
            rel_probs=[torch.zeros(3), torch.zeros(3)],
            ent_indices=[torch.tensor([1]), torch.tensor([1])],
            ent_scores=[torch.tensor([0.5]), torch.tensor([0.5])],
            e_score_indices=torch.tensor([1]),
            e_score_values=torch.tensor([0.9]),
            sample_index=i,
            triples=[[0, 0, 1]],
        )
        for i in range(n)
    ]
    meta = CacheMeta(dataset="fake", split="test", id2ent={}, id2rel={}, num_samples=n)
    return ScoreBundle(meta=meta, samples=samples)


class _RecordingAdapter:
    """记录 kg_edge_source 收到的 sample；load 返回内存 bundle。"""

    def __init__(self, bundle):
        self._bundle = bundle
        self.calls = []

    def score_loader(self):
        outer = self

        class _Loader:
            def load(self, path):
                return outer._bundle

        return _Loader()

    def kg_edge_source(self, sample=None):
        self.calls.append(sample)
        return f"kg-for-{getattr(sample, 'sample_index', None)}"


class _FakeProducer:
    def load_checkpoint(self, ckpt_path):
        pass

    def produce(self, input_dir, qa_file, *, split="test", batch_size=16, topk=500):
        return _fake_bundle()


class TestOfflinePerSample(unittest.TestCase):
    def test_each_sample_gets_own_edge_source(self):
        from kgqa.retrieve.backends.offline import OfflineBackend
        bundle = _fake_bundle()
        adapter = _RecordingAdapter(bundle)
        backend = OfflineBackend(adapter, cache_path="unused")
        # 旧实现在 __init__ 里就调 kg_edge_source()（无 sample）——新实现不应有该调用
        self.assertEqual(adapter.calls, [])
        with mock.patch("kgqa.retrieve.backends.offline.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve_all()
        self.assertEqual(adapter.calls, bundle.samples)
        got = [c.args[1] for c in m.call_args_list]
        self.assertEqual(got, ["kg-for-0", "kg-for-1"])

    def test_retrieve_single_passes_sample(self):
        from kgqa.retrieve.backends.offline import OfflineBackend
        bundle = _fake_bundle()
        adapter = _RecordingAdapter(bundle)
        backend = OfflineBackend(adapter, cache_path="unused")
        with mock.patch("kgqa.retrieve.backends.offline.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve(1)
        self.assertEqual(adapter.calls, [bundle.samples[1]])


class TestOnlinePerSample(unittest.TestCase):
    def test_each_sample_gets_own_edge_source(self):
        from kgqa.retrieve.backends.online import OnlineBackend
        adapter = _RecordingAdapter(_fake_bundle())
        backend = OnlineBackend(adapter, _FakeProducer(), ckpt_path="x",
                                input_dir="d", qa_file="q")
        self.assertEqual(adapter.calls, [])
        with mock.patch("kgqa.retrieve.backends.online.engine.retrieve_one") as m:
            m.return_value = "r"
            backend.retrieve_all()
        self.assertEqual([getattr(s, "sample_index", None) for s in adapter.calls], [0, 1])
        got = [c.args[1] for c in m.call_args_list]
        self.assertEqual(got, ["kg-for-0", "kg-for-1"])


if __name__ == "__main__":
    unittest.main()
