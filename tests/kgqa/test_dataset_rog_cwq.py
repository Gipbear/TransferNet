import json
import os
import tempfile
import unittest

from kgqa.data.rog_cwq import build_vocab, convert_split, iter_split


def _write_parquet(data_dir, split, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(data_dir, exist_ok=True)
    table = pa.table({
        "id": [r["id"] for r in rows],
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "q_entity": [r["q_entity"] for r in rows],
        "a_entity": [r["a_entity"] for r in rows],
        "graph": [r["graph"] for r in rows],
    })
    pq.write_table(table, os.path.join(data_dir, f"{split}-00000-of-00001.parquet"))


def _rows():
    return [
        {   # 常规样本：答案在子图内
            "id": "q1", "question": " Where is Rome? ", "answer": ["Italy"],
            "q_entity": ["Rome"], "a_entity": ["Italy"],
            "graph": [["Rome", "location.contained_by", "Italy"],
                      ["Italy", "location.currency", "Euro"]],
        },
        {   # 答案不在子图内（CWQ 常态，约两成样本）
            "id": "q2", "question": "Who wrote it?", "answer": ["Dickens"],
            "q_entity": ["Book"], "a_entity": ["Dickens"],
            "graph": [["Book", "book.genre", "Novel"]],
        },
        {   # 空子图，应被跳过
            "id": "q3", "question": "Empty?", "answer": ["X"],
            "q_entity": ["Y"], "a_entity": ["X"], "graph": [],
        },
    ]


class TestRoGCWQConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        _write_parquet(os.path.join(self.repo, "data"), "test", _rows())

    def tearDown(self):
        self.tmp.cleanup()

    def test_iter_split_yields_every_row(self):
        rows = list(iter_split(self.repo, "test"))

        self.assertEqual([r["id"] for r in rows], ["q1", "q2", "q3"])

    def test_vocab_covers_graph_topic_and_out_of_graph_answers(self):
        ent2id, rel2id = build_vocab(self.repo, splits=("test",))

        for name in ["Rome", "Italy", "Euro", "Book", "Novel"]:
            self.assertIn(name, ent2id)
        # q3 空子图，但它的 topic/answer 仍须进词表，否则下游 ent2id 查找会 KeyError
        self.assertIn("Dickens", ent2id)
        self.assertIn("X", ent2id)
        self.assertIn("Y", ent2id)
        self.assertEqual(set(rel2id), {"location.contained_by", "location.currency", "book.genre"})
        self.assertEqual(sorted(ent2id.values()), list(range(len(ent2id))))

    def test_vocab_normalizes_names_so_downstream_ids_stay_aligned(self):
        """带首尾空白的名字必须在建表时就归并。

        下游 CompWebQ/data.py 用 ``ent2id[line.strip()] = len(ent2id)`` 建表，
        若词表里同时存在 ' Frank Harris' 和 'Frank Harris'（RoG-cwq 真实存在），
        读回的 ent2id 会比文件行数短，*_simple.json 里按行号写的 id 全体错位。
        """
        rows = [{
            "id": "q1", "question": "Who?", "answer": ["Frank Harris"],
            "q_entity": [" Frank Harris"], "a_entity": ["Frank Harris"],
            "graph": [[" Frank Harris", "people.profession", "Writer\nEditor"]],
        }]
        repo = tempfile.TemporaryDirectory()
        self.addCleanup(repo.cleanup)
        _write_parquet(os.path.join(repo.name, "data"), "test", rows)

        ent2id, rel2id = build_vocab(repo.name, splits=("test",))

        # 两种写法归一为同一实体，而不是两个 id
        self.assertIn("Frank Harris", ent2id)
        self.assertNotIn(" Frank Harris", ent2id)
        self.assertEqual(len(ent2id), 2)  # Frank Harris + Writer Editor
        # 模拟下游读法：写出多少行，就得读回多少个不同的 key
        keys = [k for k, _ in sorted(ent2id.items(), key=lambda kv: kv[1])]
        self.assertEqual(len({k.strip() for k in keys}), len(keys))
        self.assertTrue(all("\n" not in k for k in keys))

        out = os.path.join(repo.name, "test_simple.json")
        convert_split(repo.name, "test", out, ent2id, rel2id)
        with open(out, encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        self.assertEqual(rec["entities"], [ent2id["Frank Harris"]])
        self.assertEqual(rec["answers"], [{"kb_id": "Frank Harris", "text": "Frank Harris"}])

    def test_convert_split_emits_nsm_schema(self):
        ent2id, rel2id = build_vocab(self.repo, splits=("test",))
        out = os.path.join(self.tmp.name, "test_simple.json")

        stats = convert_split(self.repo, "test", out, ent2id, rel2id)

        self.assertEqual(stats["written"], 2)
        self.assertEqual(stats["empty_graph"], 1)
        self.assertEqual(stats["answer_out_of_graph"], 1)  # 仅 q2

        with open(out, encoding="utf-8") as fh:
            recs = [json.loads(l) for l in fh]
        self.assertEqual([r["id"] for r in recs], ["q1", "q2"])
        first = recs[0]
        self.assertEqual(sorted(first["subgraph"].keys()), ["entities", "tuples"])
        self.assertEqual(first["question"], "Where is Rome?")  # 去掉首尾空格
        self.assertEqual(first["entities"], [ent2id["Rome"]])
        self.assertEqual(first["answers"], [{"kb_id": "Italy", "text": "Italy"}])
        self.assertEqual(first["subgraph"]["tuples"][0],
                         [ent2id["Rome"], rel2id["location.contained_by"], ent2id["Italy"]])
        self.assertEqual(first["subgraph"]["entities"],
                         sorted({ent2id["Rome"], ent2id["Italy"], ent2id["Euro"]}))


if __name__ == "__main__":
    unittest.main()
