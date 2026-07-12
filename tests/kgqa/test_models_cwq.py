import json
import os
import tempfile
import unittest


class TestCWQProducerHelpers(unittest.TestCase):
    def test_read_vocab_line_order(self):
        from kgqa.backbone.cwq import _read_vocab
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("m.0a\nm.0b\n")
        try:
            vocab = _read_vocab(path)
        finally:
            os.unlink(path)
        self.assertEqual(vocab, {"m.0a": 0, "m.0b": 1})

    def test_valid_lines_skips_empty_subgraph_and_limits(self):
        from kgqa.backbone.cwq import _valid_lines
        rows = [
            {"question": "q1", "subgraph": {"tuples": [[0, 0, 1]]}},
            {"question": "q2", "subgraph": {"tuples": []}},
            {"question": "q3", "subgraph": {"tuples": [[1, 0, 2]]}},
            {"question": "q4", "subgraph": {"tuples": [[2, 0, 3]]}},
        ]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        try:
            all_lines = _valid_lines(path)
            limited = _valid_lines(path, limit=2)
        finally:
            os.unlink(path)
        self.assertEqual([json.loads(l)["question"] for l in all_lines], ["q1", "q3", "q4"])
        self.assertEqual([json.loads(l)["question"] for l in limited], ["q1", "q3"])


if __name__ == "__main__":
    unittest.main()
