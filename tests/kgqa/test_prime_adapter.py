"""验证 Prime 在统一第三章检索框架中的数据集契约。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch


class PrimeAdapterTest(unittest.TestCase):
    def test_registry_loads_native_ids_and_tensor_graph(self):
        # Given
        from kgqa.retrieve.datasets.registry import get_adapter

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            graph_dir = root / "graph"
            graph_dir.mkdir()
            torch.save(torch.tensor([[0, 1], [1, 2]]), graph_dir / "edge_index.pt")
            torch.save(torch.tensor([0, 1]), graph_dir / "edge_types.pt")
            (root / "entity_names.json").write_text(json.dumps(["topic", "middle", "answer"]), encoding="utf-8")
            (graph_dir / "edge_type_dict.json").write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
            qa_path = root / "test.json"
            qa_path.write_text(json.dumps([{
                "question_id": "q1",
                "question": "Which answer?",
                "topic_entity": "0",
                "answers": ["2"],
                "hop": 2,
            }]), encoding="utf-8")

            # When
            adapter = get_adapter("prime", input_dir=str(root))
            sample = adapter.load_qa(str(qa_path))[0]
            graph = adapter.kg_edge_source(sample)

        # Then
        self.assertEqual(adapter.name, "prime")
        self.assertEqual(adapter.max_hop, 3)
        self.assertEqual(sample.topic_ids, [0])
        self.assertEqual(sample.gold_ids, [2])
        self.assertEqual(graph.neighbors(0), [(0, 1)])
        self.assertEqual(graph.neighbors(1), [(1, 2)])
        self.assertEqual(adapter.metric_spec().gold_key, "mid")
        self.assertEqual(adapter.metric_spec().group_by, "hop")

    def test_score_producer_and_ch3_parser_accept_prime(self):
        # Given
        from experiments.ch3.run import build_parser
        from kgqa.backbone import make_score_producer

        # When
        producer = make_score_producer("prime")
        args = build_parser().parse_args(["--dataset", "prime", "--dry_run"])

        # Then
        self.assertEqual(type(producer).__name__, "PrimeScoreProducer")
        self.assertEqual(args.dataset, "prime")


if __name__ == "__main__":
    unittest.main()
