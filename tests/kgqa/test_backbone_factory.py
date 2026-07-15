"""backbone 是在线 ScoreProducer 的 canonical 入口。"""
from __future__ import annotations

import unittest


class TestBackboneFactory(unittest.TestCase):
    def test_factory_dispatches_and_preserves_constructor_options(self):
        from kgqa.backbone import make_score_producer
        from kgqa.backbone.cwq import CWQScoreProducer
        from kgqa.backbone.metaqa import MetaQAScoreProducer
        from kgqa.backbone.webqsp import WebQSPScoreProducer

        webqsp = make_score_producer("webqsp")
        metaqa = make_score_producer("metaqa", per_hop_limit=3)
        cwq = make_score_producer("cwq", limit=5)

        self.assertIsInstance(webqsp, WebQSPScoreProducer)
        self.assertEqual(webqsp.bert_name, "BAAI/bge-base-en-v1.5")
        self.assertIsInstance(metaqa, MetaQAScoreProducer)
        self.assertEqual(metaqa.per_hop_limit, 3)
        self.assertIsInstance(cwq, CWQScoreProducer)
        self.assertEqual(cwq.limit, 5)

    def test_unknown_dataset_raises_keyerror(self):
        from kgqa.backbone import make_score_producer

        with self.assertRaises(KeyError):
            make_score_producer("unknown")


if __name__ == "__main__":
    unittest.main()
