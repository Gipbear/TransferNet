import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.convert_pharmkg_to_webqsp import convert_dataset


class _ArbitraryPayload:
    """非基础类型：外部归档中出现的该类实例必须被拒绝加载。"""

    def __init__(self):
        self.tag = "arbitrary"


class TestConvertPharmKGToWebQSP(unittest.TestCase):
    def test_convert_dataset_writes_webqsp_compatible_layout(self):
        # Given
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            (source_dir / "Pharm_KG.txt").write_text("0\t0\t1\n", encoding="utf-8")
            self._write_qa(source_dir / "pharmKG_train_QA_with_topic_entity.json", quoted=True)
            self._write_qa(source_dir / "pharmKG_dev_with_topic_entity.json")
            self._write_qa(source_dir / "pharmKG_test_QA_with_topic_entity.json")
            metadata_zip = root / "metadata.zip"
            self._write_metadata(metadata_zip)

            # When
            summary = convert_dataset(source_dir, metadata_zip, output_dir)

            # Then
            self.assertEqual(summary.entities, 2)
            self.assertEqual(summary.relations, 1)
            self.assertEqual(summary.triples, 1)
            self.assertEqual(summary.qa_rows, {"train": 1, "dev": 1, "test": 1})
            self.assertEqual(
                (output_dir / "fbwq_full/entities.dict").read_text(encoding="utf-8"),
                "p.0\t0\np.1\t1\n",
            )
            self.assertEqual(
                (output_dir / "fbwq_full/relations.dict").read_text(encoding="utf-8"),
                "activates\t0\nactivates_reverse\t1\n",
            )
            self.assertEqual(
                (output_dir / "fbwq_full/train.txt").read_text(encoding="utf-8"),
                "p.0\tactivates\tp.1\n",
            )
            qa_path = output_dir / "QA_data/PharmKG/qa_train_pharmkg.txt"
            self.assertEqual(qa_path.read_text(encoding="utf-8"), "Which target? [p.0]\tp.1\n")
            entity_info = output_dir / "fbwq_full/entity_info.jsonl"
            first_entity = json.loads(entity_info.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first_entity["name"], "gene-a")
            self.assertEqual(first_entity["type"], "gene")

    def test_rejects_metadata_member_with_arbitrary_class(self):
        # Given: 元数据归档的 node_info.pkl 混入任意类实例。
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            (source_dir / "Pharm_KG.txt").write_text("0\t0\t1\n", encoding="utf-8")
            self._write_qa(source_dir / "pharmKG_train_QA_with_topic_entity.json", quoted=True)
            self._write_qa(source_dir / "pharmKG_dev_with_topic_entity.json")
            self._write_qa(source_dir / "pharmKG_test_QA_with_topic_entity.json")
            metadata_zip = root / "metadata.zip"
            self._write_metadata(metadata_zip, node_info={0: _ArbitraryPayload()})

            # When/Then: 外部归档中的任意对象必须被拒绝，而不是执行其反序列化逻辑。
            with self.assertRaises(pickle.UnpicklingError):
                convert_dataset(source_dir, metadata_zip, output_dir)

    @staticmethod
    def _write_qa(path: Path, quoted: bool = False) -> None:
        question = '"Which target?"' if quoted else "Which target?"
        row = {
            "question_id": "q1",
            "question": question,
            "answer": {"1": "drug-b"},
            "topic_entity": {"topic_name": "gene-a", "topic_id": 0},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    @staticmethod
    def _write_metadata(path: Path, node_info: dict | None = None) -> None:
        if node_info is None:
            node_info = {
                0: {"name": "gene-a", "description": "Gene A.", "source": "UMLS", "type": "Gene"},
                1: {"name": "drug-b", "description": None, "source": "UMLS", "type": "Chemical"},
            }
        edge_types = {0: "activates"}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("pharmKG/node_info.pkl", pickle.dumps(node_info))
            archive.writestr("pharmKG/edge_type_dict.pkl", pickle.dumps(edge_types))


if __name__ == "__main__":
    unittest.main()
