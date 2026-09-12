"""测试 ADInt 名称原生格式转换。"""

import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.convert_adint_name_native import convert_dataset


class _ArbitraryPayload:
    """非基础类型：外部归档中出现的该类实例必须被拒绝加载。"""

    def __init__(self):
        self.tag = "arbitrary"


class TestConvertAdintNameNative(unittest.TestCase):
    def test_convert_dataset_writes_name_native_two_hop_layout(self):
        # Given
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            (source_dir / "ADint_KG.txt").write_text(
                "0\t0\t1\n1\t1\t2\n0\t0\t3\n3\t1\t4\n4\t1\t5\n",
                encoding="utf-8",
            )
            self._write_qa(source_dir / "ADint_train_with_topic_entity.json", train=True)
            self._write_qa(source_dir / "ADint_test_with_topic_entity.json", train=False)
            metadata_zip = root / "metadata.zip"
            self._write_metadata(metadata_zip)

            # When
            summary = convert_dataset(source_dir, metadata_zip, output_dir)

            # Then
            self.assertEqual(summary.entities, 6)
            self.assertEqual(summary.relations, 2)
            self.assertEqual(summary.triples, 5)
            self.assertEqual(summary.qa_rows, {"train": 2, "dev": 0, "test": 4})
            self.assertEqual(summary.dropped_rows, {"hop0": 0, "unlinked": 1, "beyond_max_hop": 1})
            self.assertFalse((output_dir / "entities.dict").exists())
            self.assertFalse((output_dir / "mid2name.txt").exists())
            self.assertEqual(
                (output_dir / "kb/kb.tsv").read_text(encoding="utf-8"),
                "drug|alias\tTREATS\tdisease\n"
                "disease\tASSOCIATED_WITH\tgene\n"
                "drug|alias\tTREATS\tfar-a\n"
                "far-a\tASSOCIATED_WITH\tfar-b\n"
                "far-b\tASSOCIATED_WITH\tfar-c\n",
            )
            train = json.loads((output_dir / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(train[0]["topic_entity"], "drug|alias")
            self.assertEqual(train[0]["answers"], ["disease"])
            self.assertEqual(train[0]["hop"], 1)
            self.assertEqual(train[1]["answers"], ["gene"])
            self.assertEqual(train[1]["hop"], 2)
            test = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
            self.assertEqual([row["question_id"] for row in test], ["q1", "q5", "q6", "q7"])
            self.assertEqual(test[1]["topic_entity"], "missing topic")
            self.assertIsNone(test[1]["hop"])
            self.assertEqual(test[2]["answers"], ["far-c"])
            self.assertIsNone(test[2]["hop"])
            self.assertEqual(test[3]["hop"], 0)

    @staticmethod
    def _write_qa(path: Path, train: bool) -> None:
        rows = [
            {
                "question_id": "q1",
                "question": "Which disease?",
                "answer": {"1": "disease"},
                "topic_entity": {"topic_name": "drug|alias", "topic_id": 0},
                "hop1_neighbors": [],
            }
        ]
        if train:
            rows.extend(
                [
                    {
                        "question_id": "q2",
                        "question": "Which gene?",
                        "answer": {"2": "gene"},
                        "topic_entity": {"topic_name": "drug|alias", "topic_id": 0},
                        "hop1_neighbors": [],
                    },
                    {
                        "question_id": "q3",
                        "question": "Which far entity?",
                        "answer": {"5": "far-c"},
                        "topic_entity": {"topic_name": "drug|alias", "topic_id": 0},
                        "hop1_neighbors": [],
                    },
                    {
                        "question_id": "q4",
                        "question": "Which missing entity?",
                        "answer": {"1": "disease"},
                        "topic_entity": "missing topic",
                        "hop1_neighbors": "None",
                    },
                ]
            )
        else:
            rows.extend(
                [
                    {
                        "question_id": "q5",
                        "question": "Which missing entity?",
                        "answer": {"1": "disease"},
                        "topic_entity": "missing topic",
                        "hop1_neighbors": "None",
                    },
                    {
                        "question_id": "q6",
                        "question": "Which far entity?",
                        "answer": {"5": "far-c"},
                        "topic_entity": {"topic_name": "drug|alias", "topic_id": 0},
                        "hop1_neighbors": [],
                    },
                    {
                        "question_id": "q7",
                        "question": "Which topic entity?",
                        "answer": {"0": "drug|alias"},
                        "topic_entity": {"topic_name": "drug|alias", "topic_id": 0},
                        "hop1_neighbors": [],
                    },
                ]
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_rejects_metadata_member_with_arbitrary_class(self):
        # Given: 元数据归档的 node_info.pkl 混入任意类实例。
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            (source_dir / "ADint_KG.txt").write_text("0\t0\t1\n", encoding="utf-8")
            self._write_qa(source_dir / "ADint_train_with_topic_entity.json", train=True)
            self._write_qa(source_dir / "ADint_test_with_topic_entity.json", train=False)
            metadata_zip = root / "metadata.zip"
            self._write_metadata(metadata_zip, node_info={0: _ArbitraryPayload()})

            # When/Then: 外部归档中的任意对象必须被拒绝，而不是执行其反序列化逻辑。
            with self.assertRaises(pickle.UnpicklingError):
                convert_dataset(source_dir, metadata_zip, output_dir)

    @staticmethod
    def _write_metadata(path: Path, node_info: dict | None = None) -> None:
        if node_info is None:
            names = ["drug|alias", "disease", "gene", "far-a", "far-b", "far-c"]
            node_info = {
                entity_id: {"name": name, "description": "", "source": "UMLS", "type": "concept"}
                for entity_id, name in enumerate(names)
            }
        edge_types = {0: "TREATS", 1: "ASSOCIATED_WITH"}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("ADint/node_info.pkl", pickle.dumps(node_info))
            archive.writestr("ADint/edge_type_dict.pkl", pickle.dumps(edge_types))


if __name__ == "__main__":
    unittest.main()
