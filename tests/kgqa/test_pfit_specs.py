"""PfitDatasetSpec:数据集差异钩子注册表。"""
import unittest


class TestWebQSPSpec(unittest.TestCase):
    def setUp(self):
        from kgqa.pfit.specs import get_pfit_spec
        self.spec = get_pfit_spec("webqsp")

    def test_entity_reprs(self):
        self.assertEqual(set(self.spec.entity_reprs), {"mid", "name"})
        self.assertTrue(self.spec.entity_map_path.endswith(
            "data/resources/WebQSP/fbwq_full/mapped_entities.txt"))

    def test_clean_question_strips_bert_tokens(self):
        q = "[CLS] who plays ken bar ##low [SEP]"
        self.assertEqual(self.spec.clean_question(q, ["m.01_2n"]),
                         "who plays ken barlow")

    def test_rejection_supported(self):
        self.assertTrue(self.spec.supports_rejection)

    def test_no_hop_grouping(self):
        self.assertFalse(self.spec.group_by_hop)


class TestMetaQASpec(unittest.TestCase):
    def setUp(self):
        from kgqa.pfit.specs import get_pfit_spec
        self.spec = get_pfit_spec("metaqa")

    def test_entity_reprs_name_only(self):
        self.assertEqual(self.spec.entity_reprs, ("name",))
        self.assertIsNone(self.spec.entity_map_path)

    def test_clean_question_fills_topic(self):
        q = "what does E_S appear in"
        self.assertEqual(self.spec.clean_question(q, ["Grégoire Colin"]),
                         "what does Grégoire Colin appear in")

    def test_clean_question_fills_lowercase_placeholder(self):
        # score 缓存经 vocab 解码后的真实形态是小写 e_s(retrieve 输出即此形态)
        q = "what does e_s appear in"
        self.assertEqual(self.spec.clean_question(q, ["Grégoire Colin"]),
                         "what does Grégoire Colin appear in")

    def test_clean_question_without_topic_keeps_text(self):
        q = "what does E_S appear in"
        self.assertEqual(self.spec.clean_question(q, []), q)

    def test_rejection_unsupported(self):
        self.assertFalse(self.spec.supports_rejection)

    def test_hop_grouping(self):
        self.assertTrue(self.spec.group_by_hop)
        self.assertEqual(self.spec.hops, (1, 2, 3))


class TestRegistry(unittest.TestCase):
    def test_unknown_dataset_raises(self):
        from kgqa.pfit.specs import get_pfit_spec
        with self.assertRaises(KeyError):
            get_pfit_spec("cwq")  # CWQ 暂缓,stage2 不注册


if __name__ == "__main__":
    unittest.main()
