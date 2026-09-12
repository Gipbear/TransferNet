"""验证 Prime TransferNet 数据入口。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class RecordingTokenizer:
    def __call__(self, texts, **kwargs):
        shape = (len(texts), kwargs["max_length"])
        return {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


class PrimeDataTest(unittest.TestCase):
    def test_load_data_uses_native_ids_and_lazy_three_hop_range(self):
        # Given
        from Prime.data import load_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_fixture(root)

            # When
            with patch("Prime.data.from_pretrained_local_first", return_value=RecordingTokenizer()):
                ent2id, rel2id, triples, _, dev_loader, _ = load_data(str(root), "unused", 2)
                batch = next(iter(dev_loader))

        # Then
        self.assertEqual(len(ent2id), 5)
        self.assertEqual(rel2id, {"rel-a": 0, "rel-b": 1})
        self.assertEqual(triples.shape, (4, 3))
        self.assertEqual(batch[0][0].tolist(), [0])
        self.assertEqual(batch[2][0].tolist(), [3])
        self.assertEqual(set(batch[3][0].tolist()), {1, 2, 3, 4})
        self.assertEqual(dev_loader.hops, [3])
        self.assertEqual(dev_loader.id2ent[4], "duplicate")

    @staticmethod
    def _write_fixture(root: Path) -> None:
        graph = root / "graph"
        graph.mkdir()
        torch.save(torch.tensor([[0, 1, 2, 0], [1, 2, 3, 4]]), graph / "edge_index.pt")
        torch.save(torch.tensor([0, 1, 1, 0]), graph / "edge_types.pt")
        (graph / "edge_type_dict.json").write_text(json.dumps({"rel-a": 0, "rel-b": 1}), encoding="utf-8")
        names = ["topic", "duplicate", "middle", "answer", "duplicate"]
        (root / "entity_names.json").write_text(json.dumps(names), encoding="utf-8")
        row = {"question_id": "q", "question": "Which?", "topic_entity": "0", "answers": ["3"], "hop": 3}
        for split in ("train", "dev", "test"):
            (root / f"{split}.json").write_text(json.dumps([row]), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
