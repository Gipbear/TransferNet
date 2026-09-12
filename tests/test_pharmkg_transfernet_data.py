"""验证 PharmKG TransferNet 数据入口的解析、编码与数据集分割。"""

import gc
import tempfile
import unittest
import warnings
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


class PharmKGDataTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        graph_dir = root / "fbwq_full"
        qa_dir = root / "QA_data" / "PharmKG"
        graph_dir.mkdir(parents=True)
        qa_dir.mkdir(parents=True)
        (graph_dir / "entities.dict").write_text("drug\t0\ntarget\t1\nanswer\t2\n", encoding="utf-8")
        (graph_dir / "relations.dict").write_text("binds\t0\nbinds_reverse\t1\n", encoding="utf-8")
        (graph_dir / "train.txt").write_text(
            "drug\tbinds\ttarget\ntarget\tbinds\tanswer\n",
            encoding="utf-8",
        )
        qa_rows = {
            "qa_train_pharmkg.txt": "train Poly [ADP-ribose] question [drug]\tanswer\n",
            "qa_dev_pharmkg.txt": "dev question [drug]\tanswer\n",
            "qa_test_pharmkg.txt": "test question [drug]\tanswer\n",
        }
        for name, content in qa_rows.items():
            (qa_dir / name).write_text(content, encoding="utf-8")

    def test_load_data_uses_final_topic_marker_and_three_splits(self):
        # Given: 含问题内部方括号的三套 PharmKG QA 文件。
        from PharmKG.data import load_data

        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_fixture(root)

            # When: 通过 PharmKG 数据入口加载完整分割。
            with patch("PharmKG.data.from_pretrained_local_first", return_value=tokenizer):
                _, _, _, train_loader, dev_loader, test_loader = load_data(
                    str(root),
                    "unused-tokenizer",
                    batch_size=2,
                )

        # Then: 内部方括号保留，主题实体取最后标记，并使用 128-token 截断编码。
        self.assertEqual(train_loader.qa_text, ["train Poly [ADP-ribose] question"])
        self.assertEqual(dev_loader.qa_text, ["dev question"])
        self.assertEqual(test_loader.qa_text, ["test question"])
        self.assertEqual([len(loader.dataset) for loader in (train_loader, dev_loader, test_loader)], [1, 1, 1])
        self.assertEqual(len(tokenizer.calls), 3)
        for _, kwargs in tokenizer.calls:
            self.assertEqual(kwargs["max_length"], 128)
            self.assertTrue(kwargs["truncation"])
            self.assertEqual(kwargs["padding"], "max_length")

    def test_load_graph_closes_graph_files(self):
        # Given: 三份图谱文件与记录全部 ResourceWarning 的告警上下文。
        from PharmKG.data import load_graph

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_fixture(root)

            # When: 只加载图谱，不构造 DataLoader。
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                ent2id, rel2id, triples = load_graph(str(root))
                gc.collect()

        # Then: 解析成功且不得残留未关闭的图谱文件句柄。
        self.assertEqual(len(ent2id), 3)
        self.assertEqual(len(rel2id), 2)
        self.assertEqual(len(triples), 4)
        leaked = [item for item in caught if "PharmKG" in str(item.filename)]
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()
