"""验证 ADInt 名称原生 TransferNet 数据入口。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        width = kwargs["max_length"]
        shape = (len(texts), width)
        return {
            "input_ids": torch.zeros(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


class ADIntDataTest(unittest.TestCase):
    def test_load_data_encodes_names_without_external_alias_mapping(self):
        # Given
        from ADInt.data import load_data

        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "kb").mkdir()
            (root / "kb/kb.tsv").write_text(
                "drug|alias\tTREATS\tdisease\n"
                "disease\tASSOCIATED_WITH\tgene [family]\n",
                encoding="utf-8",
            )
            row = {
                "question_id": "q1",
                "question": "Which gene?",
                "topic_entity": "drug|alias",
                "answers": ["gene [family]"],
                "hop": 2,
            }
            for split in ("train", "dev", "test"):
                (root / f"{split}.json").write_text(json.dumps([row]), encoding="utf-8")

            # When
            with patch("ADInt.data.from_pretrained_local_first", return_value=tokenizer):
                ent2id, rel2id, triples, train_loader, dev_loader, test_loader = load_data(
                    str(root),
                    "unused-tokenizer",
                    batch_size=2,
                )

        # Then
        self.assertEqual(set(ent2id), {"drug|alias", "disease", "gene [family]"})
        self.assertEqual(
            set(rel2id),
            {"TREATS", "TREATS_reverse", "ASSOCIATED_WITH", "ASSOCIATED_WITH_reverse"},
        )
        self.assertEqual(triples.shape, (4, 3))
        self.assertEqual([len(loader.dataset) for loader in (train_loader, dev_loader, test_loader)], [1, 1, 1])
        batch = next(iter(dev_loader))
        self.assertEqual(batch[0][0].tolist(), [ent2id["drug|alias"]])
        self.assertEqual(batch[2][0].tolist(), [ent2id["gene [family]"]])
        self.assertEqual(set(batch[3][0].tolist()), {ent2id["disease"], ent2id["gene [family]"]})
        self.assertEqual(dev_loader.qa_text, ["Which gene?"])
        self.assertEqual(len(tokenizer.calls), 3)

    def test_data_loader_keeps_unlinked_topic_as_empty_seed(self):
        # Given
        from ADInt.data import DataLoader, LoaderSpec, load_graph

        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "kb").mkdir()
            (root / "kb/kb.tsv").write_text("drug\tTREATS\tdisease\n", encoding="utf-8")
            row = {
                "question_id": "q1",
                "question": "Which disease?",
                "topic_entity": "unlinked topic",
                "answers": ["disease"],
                "hop": None,
            }
            qa_file = root / "test.json"
            qa_file.write_text(json.dumps([row]), encoding="utf-8")
            graph = load_graph(str(root))

            # When
            with patch("ADInt.data.from_pretrained_local_first", return_value=tokenizer):
                loader = DataLoader(LoaderSpec(root, "test", "unused-tokenizer", 1), graph)
                batch = next(iter(loader))

        # Then
        self.assertEqual(batch[0][0].numel(), 0)
        self.assertEqual(batch[2][0].tolist(), [graph.ent2id["disease"]])
        self.assertEqual(batch[3][0].numel(), 0)
        self.assertEqual(loader.hops, [None])


if __name__ == "__main__":
    unittest.main()
