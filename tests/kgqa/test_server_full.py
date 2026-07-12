"""kgqa/server 全功能检索服务测试(stage3 上移自 oh_my_agent/path_retrieve_server)。

覆盖:question/topic_entities 定位、θ 阈值 prediction、group_tails 在线构建、
drop_loopback 开关、schema 兼容;真实缓存下与 legacy CachedPathRetriever parity。
"""
import os
import unittest

import torch

from kgqa.kg.global_kg import GlobalKG
from kgqa.scores.base import CacheMeta, SampleScore, ScoreBundle

CACHE = "data/output/WebQSP/path_retrieve_server/score_cache/webqsp_test_1581.pt"
INPUT_DIR = "data/input/WebQSP"


def _toy_sample() -> SampleScore:
    return SampleScore(
        question="[CLS] toy question [SEP]",
        topic_ids=[0],
        gold_ids=[1],
        hop_attn=torch.tensor([0.9, 0.1]),
        rel_probs=[torch.tensor([0.0, 0.8, 0.7]), torch.tensor([0.0, 0.0, 0.0])],
        ent_indices=[torch.tensor([1, 2]), torch.tensor([], dtype=torch.long)],
        ent_scores=[torch.tensor([0.6, 0.5]), torch.tensor([])],
        e_score_indices=torch.tensor([1, 2]),
        e_score_values=torch.tensor([0.95, 0.4]),
        sample_index=0,
    )


class _ToyLoader:
    def load(self, cache_path):
        return ScoreBundle(
            meta=CacheMeta(
                dataset="webqsp", split="test",
                id2ent={0: "m.topic", 1: "m.gold", 2: "m.other"},
                id2rel={1: "rel.one", 2: "rel.two"},
                num_samples=1,
            ),
            samples=[_toy_sample()],
        )


class _ToyAdapter:
    def __init__(self):
        self._kg = GlobalKG.from_triples([[0, 1, 1], [0, 2, 2]])

    def kg_edge_source(self, sample=None):
        return self._kg

    def score_loader(self):
        return _ToyLoader()


class TestPathRetrieveService(unittest.TestCase):
    def _service(self, **kwargs):
        from kgqa.server.service import PathRetrieveService
        return PathRetrieveService(_ToyAdapter(), cache_path="toy.pt", **kwargs)

    def test_retrieve_by_sample_index(self):
        r = self._service().retrieve(sample_index=0)
        self.assertEqual(r.topics, ["m.topic"])
        self.assertEqual(r.hop, 1)
        rels = sorted(p["path"][0][1] for p in r.mmr_reason_paths)
        self.assertEqual(rels, ["rel.one", "rel.two"])
        self.assertEqual(r.cache_path, "toy.pt")

    def test_retrieve_by_question_normalized(self):
        # [CLS]/[SEP]/大小写/多空格 归一化后可命中
        r = self._service().retrieve(question="Toy   QUESTION")
        self.assertEqual(r.sample_index, 0)

    def test_topic_entities_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._service().retrieve(sample_index=0, topic_entities=["m.wrong"])

    def test_prediction_uses_threshold(self):
        # 默认 θ=0.9:仅 m.gold(0.95)入 prediction
        r = self._service().retrieve(sample_index=0)
        self.assertEqual(set(r.prediction), {"m.gold"})
        # θ 参数化:降到 0.3 后 m.other(0.4)也进
        r_low = self._service(prediction_threshold=0.3).retrieve(sample_index=0)
        self.assertEqual(set(r_low.prediction), {"m.gold", "m.other"})

    def test_group_tails_online(self):
        # 最后一跳按 prediction 过滤:rel.one 组尾=m.gold;rel.two 组尾被过滤为空
        r = self._service().retrieve(sample_index=0)
        self.assertEqual(r.group_tails.get("m.topic|rel.one"), ["m.gold"])
        self.assertEqual(r.group_tails.get("m.topic|rel.two"), [])

    def test_missing_lookup_key_raises(self):
        with self.assertRaises(ValueError):
            self._service().retrieve()

    def test_unknown_question_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self._service().retrieve(question="no such question")


class TestServiceApp(unittest.TestCase):
    def setUp(self):
        from kgqa.server.path_retrieve_server import create_service_app
        from kgqa.server.service import PathRetrieveService
        service = PathRetrieveService(_ToyAdapter(), cache_path="toy.pt")
        app = create_service_app(service)
        self.endpoints = {
            route.path: route.endpoint
            for route in app.routes
            if hasattr(route, "endpoint")
        }

    def test_health_and_info(self):
        self.assertEqual(self.endpoints["/health"]()["status"], "ok")
        info = self.endpoints["/info"]()
        self.assertTrue(info["cache_loaded"])
        self.assertEqual(info["num_samples"], 1)

    def test_retrieve_endpoint_schema_compat(self):
        from kgqa.server.schema import RetrieveRequest
        body = self.endpoints["/retrieve"](
            RetrieveRequest(question="toy question", topic_entities=["m.topic"])
        )
        # legacy 响应字段全在位
        for key in ("question", "sample_index", "topics", "hop", "mmr_reason_paths",
                    "prediction", "elapsed_ms", "alpha_final", "threshold",
                    "beam_size", "lambda_val", "cache_path", "group_tails"):
            self.assertIn(key, body)

    def test_retrieve_endpoint_404_on_unknown_question(self):
        from fastapi import HTTPException
        from kgqa.server.schema import RetrieveRequest
        with self.assertRaises(HTTPException) as ctx:
            self.endpoints["/retrieve"](RetrieveRequest(question="no such question"))
        self.assertEqual(ctx.exception.status_code, 404)


@unittest.skipUnless(os.path.isfile(CACHE) and os.path.isdir(INPUT_DIR),
                     "真实缓存/数据缺失,跳过 parity")
class TestServiceLegacyParity(unittest.TestCase):
    """新 kgqa/server 服务与 legacy CachedPathRetriever 逐样本 parity(免 GPU)。"""

    N_SAMPLES = 5

    @classmethod
    def setUpClass(cls):
        from kgqa.datasets.registry import get_adapter
        from kgqa.server.service import PathRetrieveService
        from oh_my_agent.path_retrieve_server.service import CachedPathRetriever

        adapter = get_adapter("webqsp", input_dir=INPUT_DIR)
        cls.new = PathRetrieveService(adapter, cache_path=CACHE)
        cls.old = CachedPathRetriever(cache_path=CACHE, input_dir=INPUT_DIR)

    def test_responses_match(self):
        params = dict(beam_size=20, lambda_val=0.2, threshold=0.01, alpha_final=1.0)
        for i in range(self.N_SAMPLES):
            with self.subTest(sample=i):
                rn = self.new.retrieve(sample_index=i, **params).to_dict()
                ro = self.old.retrieve(sample_index=i, **params).to_dict()
                self.assertEqual(rn["question"], ro["question"])
                self.assertEqual(rn["topics"], ro["topics"])
                self.assertEqual(rn["hop"], ro["hop"])
                self.assertEqual(rn["prediction"], ro["prediction"])
                self.assertEqual(rn["group_tails"], ro["group_tails"])
                self.assertEqual(len(rn["mmr_reason_paths"]), len(ro["mmr_reason_paths"]))
                for pn, po in zip(rn["mmr_reason_paths"], ro["mmr_reason_paths"]):
                    self.assertEqual(pn["path"], po["path"])
                    self.assertAlmostEqual(pn["log_score"], po["log_score"], places=5)

    def test_question_lookup_matches(self):
        q = self.old.samples[0]["question"]
        rn = self.new.retrieve(question=q)
        ro = self.old.retrieve(question=q)
        self.assertEqual(rn.sample_index, ro.sample_index)


if __name__ == "__main__":
    unittest.main()
