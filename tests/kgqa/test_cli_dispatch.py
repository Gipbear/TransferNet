import unittest


class TestRetrieveProducerDispatch(unittest.TestCase):
    def test_make_producer_by_dataset(self):
        from kgqa.cli.retrieve import _make_producer
        from kgqa.models.cwq import CWQScoreProducer
        from kgqa.models.metaqa import MetaQAScoreProducer
        from kgqa.models.webqsp import WebQSPScoreProducer
        self.assertIsInstance(_make_producer("webqsp"), WebQSPScoreProducer)
        self.assertIsInstance(_make_producer("metaqa"), MetaQAScoreProducer)
        self.assertIsInstance(_make_producer("cwq"), CWQScoreProducer)

    def test_make_producer_unknown_raises(self):
        from kgqa.cli.retrieve import _make_producer
        with self.assertRaises(SystemExit):
            _make_producer("nope")


class TestDumpParserLimit(unittest.TestCase):
    def test_limit_arg(self):
        from kgqa.cli.dump_scores import build_parser
        args = build_parser().parse_args(
            ["--dataset", "cwq", "--ckpt", "c", "--input_dir", "d",
             "--qa_file", "q", "--output", "o", "--limit", "4"])
        self.assertEqual(args.limit, 4)


if __name__ == "__main__":
    unittest.main()
