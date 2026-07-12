"""ReaRev-WebQSP 适配器接入统一 retrieve 数据集注册表。"""
from __future__ import annotations

import unittest
from tempfile import NamedTemporaryFile

import torch

from kgqa.core.contracts import SampleScore
from kgqa.retrieve.datasets.registry import get_adapter


class TestReaRevWebQSPAdapter(unittest.TestCase):
    def test_registry_returns_canonical_adapter(self):
        from kgqa.retrieve.datasets.rearev_webqsp import ReaRevWebQSPAdapter

        adapter = get_adapter("webqsp-rearev")
        self.assertIsInstance(adapter, ReaRevWebQSPAdapter)
        self.assertEqual(adapter.name, "webqsp-rearev")
        self.assertEqual(adapter.max_hop, 3)
        self.assertEqual(adapter.metric_spec().gold_key, "mid")

    def test_per_sample_subgraph_uses_cached_triples(self):
        adapter = get_adapter("webqsp-rearev")
        sample = SampleScore(
            question="q",
            topic_ids=[0],
            gold_ids=[2],
            hop_attn=torch.tensor([1.0]),
            rel_probs=[],
            ent_indices=[],
            ent_scores=[],
            e_score_indices=torch.tensor([2]),
            e_score_values=torch.tensor([1.0]),
            triples=[[0, 1, 2]],
        )

        self.assertEqual(adapter.kg_edge_source(sample).neighbors(0), [(1, 2)])

    def test_score_loader_reads_rearev_dump_schema(self):
        cache = {
            "version": 1,
            "meta": {
                "dataset": "webqsp-rearev",
                "split": "test",
                "id2ent": {0: "m.topic", 2: "m.answer"},
                "id2rel": {1: "rel"},
                "num_samples": 1,
            },
            "samples": [{
                "question": "q",
                "topic_ids": [0],
                "gold_ids": [2],
                "hop_attn": torch.tensor([1.0]),
                "rel_probs": [],
                "ent_indices": [],
                "ent_scores": [],
                "e_score_indices": torch.tensor([2]),
                "e_score_values": torch.tensor([1.0]),
                "triples": [[0, 1, 2]],
            }],
        }
        with NamedTemporaryFile(suffix=".pt") as output:
            torch.save(cache, output.name)
            bundle = get_adapter("webqsp-rearev").score_loader().load(output.name)

        self.assertEqual(bundle.meta.dataset, "webqsp-rearev")
        self.assertEqual(bundle.samples[0].triples, [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
