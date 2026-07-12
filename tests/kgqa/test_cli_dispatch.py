import unittest


class TestRetrieveProducerDispatch(unittest.TestCase):
    def test_make_producer_by_dataset(self):
        from kgqa.retrieve.cli.retrieve import _make_producer
        from kgqa.backbone.cwq import CWQScoreProducer
        from kgqa.backbone.metaqa import MetaQAScoreProducer
        from kgqa.backbone.webqsp import WebQSPScoreProducer
        self.assertIsInstance(_make_producer("webqsp"), WebQSPScoreProducer)
        self.assertIsInstance(_make_producer("metaqa"), MetaQAScoreProducer)
        self.assertIsInstance(_make_producer("cwq"), CWQScoreProducer)

    def test_make_producer_unknown_raises(self):
        from kgqa.retrieve.cli.retrieve import _make_producer
        with self.assertRaises(SystemExit):
            _make_producer("nope")

    def test_rearev_only_allows_offline_adapter(self):
        from kgqa.retrieve.cli.retrieve import _adapter_name, _make_producer
        self.assertEqual(_adapter_name("webqsp", "rearev"), "webqsp-rearev")
        with self.assertRaises(SystemExit):
            _adapter_name("metaqa", "rearev")
        with self.assertRaises(SystemExit):
            _make_producer("webqsp", "rearev")


class TestDumpParserLimit(unittest.TestCase):
    def test_limit_arg(self):
        from kgqa.retrieve.cli.dump_scores import build_parser
        args = build_parser().parse_args(
            ["--dataset", "cwq", "--ckpt", "c", "--input_dir", "d",
             "--qa_file", "q", "--output", "o", "--limit", "4"])
        self.assertEqual(args.limit, 4)
        self.assertEqual(args.backbone, "transfernet")
        self.assertTrue(hasattr(args, "run_dir"))


if __name__ == "__main__":
    unittest.main()
