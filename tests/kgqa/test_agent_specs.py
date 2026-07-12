"""AgentDatasetSpec 注册表测试(stage3)。"""
import json
import os
import tempfile
import unittest


class TestWebqspSpec(unittest.TestCase):
    def setUp(self):
        from kgqa.agent.specs import get_agent_spec
        self.spec = get_agent_spec("webqsp")

    def test_basic_fields(self):
        self.assertEqual(self.spec.name, "webqsp")
        self.assertEqual(self.spec.hops, (1, 2))
        self.assertIsNotNone(self.spec.entity_map_path)
        self.assertFalse(self.spec.group_by_hop)

    def test_load_qa_parses_tab_line_with_topic_mid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write("[CLS] what does jamaican people speak [SEP] [m.03_r3]\t"
                     "m.01428y|m.04ygk0\n")
            path = fh.name
        try:
            samples = self.spec.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s.question, "what does jamaican people speak")
        self.assertEqual(s.topic_id, "m.03_r3")
        self.assertEqual(s.gold_ids, ["m.01428y", "m.04ygk0"])
        self.assertIsNone(s.sample_index)


class TestMetaqaSpec(unittest.TestCase):
    def setUp(self):
        from kgqa.agent.specs import get_agent_spec
        self.spec = get_agent_spec("metaqa")

    def test_basic_fields(self):
        self.assertEqual(self.spec.hops, (1, 2, 3))
        self.assertIsNone(self.spec.entity_map_path)
        self.assertEqual(self.spec.load_entity_map(), {})  # 恒等映射
        self.assertTrue(self.spec.group_by_hop)

    def test_load_qa_from_retrieve_jsonl(self):
        rec = {
            "question": "what does e_s appear in",
            "topics": ["Grégoire Colin"],
            "hop": 1,
            "golden": ["Son frère"],
            "sample_index": 7,
            "mmr_reason_paths": [],
            "prediction": {},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            path = fh.name
        try:
            samples = self.spec.load_qa(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        # 展示问题:e_s 回填 topic 实体名;检索定位走 sample_index
        self.assertEqual(s.question, "what does Grégoire Colin appear in")
        self.assertEqual(s.question_raw, "what does e_s appear in")
        self.assertEqual(s.topic_id, "Grégoire Colin")
        self.assertEqual(s.gold_ids, ["Son frère"])
        self.assertEqual(s.sample_index, 7)

    def test_load_qa_limit(self):
        recs = [
            {"question": f"q{i} e_s", "topics": [f"t{i}"], "golden": [f"g{i}"],
             "sample_index": i}
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            path = fh.name
        try:
            samples = self.spec.load_qa(path, limit=2)
        finally:
            os.unlink(path)
        self.assertEqual(len(samples), 2)


class TestRegistry(unittest.TestCase):
    def test_unknown_dataset_raises(self):
        from kgqa.agent.specs import get_agent_spec
        with self.assertRaises(KeyError):
            get_agent_spec("nope")


if __name__ == "__main__":
    unittest.main()
