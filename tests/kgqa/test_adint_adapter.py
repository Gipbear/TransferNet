"""验证 ADInt 在统一检索与第四章流水线中的数据集契约。"""

import json
import tempfile
import unittest
from pathlib import Path


class TestADIntAdapter(unittest.TestCase):
    def test_registry_loads_name_native_qa_and_graph(self):
        # Given: 名称原生的最小 ADInt 问答与双向知识图谱。
        from kgqa.retrieve.datasets.registry import get_adapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kb").mkdir()
            (root / "kb/kb.tsv").write_text("drug\tTREATS\tdisease\n", encoding="utf-8")
            qa_path = root / "test.json"
            qa_path.write_text(json.dumps([{
                "question_id": "q1",
                "question": "What does the drug treat?",
                "topic_entity": "drug",
                "answers": ["disease"],
                "hop": 1,
            }]), encoding="utf-8")

            # When: 通过统一注册表读取问答和图邻居。
            adapter = get_adapter("adint", input_dir=str(root))
            sample = adapter.load_qa(str(qa_path))[0]
            graph = adapter.kg_edge_source(sample)

        # Then: 实体保持名称，图 ID 与 ADInt TransferNet 的首见顺序一致。
        self.assertEqual(adapter.name, "adint")
        self.assertEqual(sample.topic_ids, ["drug"])
        self.assertEqual(sample.gold_ids, ["disease"])
        self.assertEqual(sample.hop, 1)
        self.assertEqual(adapter.entity_name("drug"), "drug")
        self.assertEqual(graph.neighbors(0), [(0, 1)])
        self.assertEqual(graph.neighbors(1), [(1, 0)])
        self.assertEqual(adapter.metric_spec().gold_key, "name")


class TestADIntDispatch(unittest.TestCase):
    def test_score_producer_dispatches_to_adint(self):
        # Given: 第三章统一在线得分生产入口。
        from kgqa.backbone import make_score_producer

        # When: 使用 ADInt 数据集名称分发。
        producer = make_score_producer("adint")

        # Then: 返回独立的名称原生 ADInt producer。
        self.assertEqual(type(producer).__name__, "ADIntScoreProducer")
        self.assertEqual(type(producer).__module__, "kgqa.backbone.adint")

    def test_ch3_and_ch4_parsers_accept_adint(self):
        from experiments.ch3.run import _default_config, build_parser as build_ch3_parser
        from experiments.ch4.run import build_parser as build_ch4_parser

        ch3_args = build_ch3_parser().parse_args(["--dataset", "adint", "--dry_run"])
        ch4_args = build_ch4_parser().parse_args([
            "--dataset", "adint",
            "--config", "matrix.json",
            "--profile", "profile.json",
        ])

        self.assertEqual(ch3_args.dataset, "adint")
        self.assertEqual(ch4_args.dataset, "adint")
        self.assertTrue(_default_config(Path.cwd(), "adint", "transfernet").is_file())


if __name__ == "__main__":
    unittest.main()
