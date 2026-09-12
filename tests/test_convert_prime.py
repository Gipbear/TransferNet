"""验证 Prime 原生 ID 数据转换。"""

import io
import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from scripts.convert_prime import SourceQA


class _ArbitraryPayload:
    """非基础类型：外部归档中出现的该类实例必须被拒绝加载。"""

    def __init__(self):
        self.tag = "arbitrary"


class PrimeConversionTest(unittest.TestCase):
    def test_convert_dataset_keeps_native_ids_and_three_hop_rows(self):
        # Given
        from scripts.convert_prime import convert_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            source.mkdir()
            self._write_qa(source / "Prime_train_QA_with_topic_entity.json", include_dropped=True)
            self._write_qa(source / "Prime_test_QA_with_topic_entity.json", include_dropped=False)
            archive = root / "prime.zip"
            self._write_graph(archive)

            # When
            summary = convert_dataset(source, archive, root / "output")

            # Then
            self.assertEqual(summary.entities, 6)
            self.assertEqual(summary.relations, 2)
            self.assertEqual(summary.triples, 5)
            self.assertEqual(summary.qa_rows, {"train": 3, "dev": 0, "test": 3})
            self.assertEqual(summary.dropped_rows, {"unlinked": 1, "hop0": 1, "beyond_max_hop": 1})
            rows = json.loads((root / "output/train.json").read_text(encoding="utf-8"))
            self.assertEqual([row["topic_entity"] for row in rows], ["0", "0", "0"])
            self.assertEqual([row["hop"] for row in rows], [1, 2, 3])
            self.assertEqual(rows[-1]["answers"], ["3", "4"])
            edge_index = torch.load(root / "output/graph/edge_index.pt", weights_only=True)
            self.assertEqual(edge_index.shape, (2, 5))

    @staticmethod
    def _write_qa(path: Path, include_dropped: bool) -> None:
        rows = [
            PrimeConversionTest._row("q1", {"1": "duplicate"}),
            PrimeConversionTest._row("q2", {"2": "middle"}),
            PrimeConversionTest._row("q3", {"3": "far", "4": "duplicate"}),
        ]
        if include_dropped:
            rows.extend(
                [
                    PrimeConversionTest._row("q4", {"5": "unreachable"}),
                    PrimeConversionTest._row("q5", {"0": "topic"}),
                    {
                        "question_id": "q6",
                        "question": "Unlinked?",
                        "answer": {"1": "duplicate"},
                        "topic_entity": [],
                    },
                ]
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _row(question_id: str, answers: dict[str, str]) -> SourceQA:
        return {
            "question_id": question_id,
            "question": f"Question {question_id}?",
            "answer": answers,
            "topic_entity": [{"topic_name": "topic", "topic_id": "0"}],
        }

    def test_rejects_metadata_member_with_arbitrary_class(self):
        # Given: 元数据归档的 node_info.pkl 混入任意类实例。
        from scripts.convert_prime import convert_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source"
            source.mkdir()
            self._write_qa(source / "Prime_train_QA_with_topic_entity.json", include_dropped=True)
            self._write_qa(source / "Prime_test_QA_with_topic_entity.json", include_dropped=False)
            archive = root / "prime.zip"
            self._write_graph(archive, nodes={0: _ArbitraryPayload()})

            # When/Then: 外部归档中的任意对象必须被拒绝，而不是执行其反序列化逻辑。
            with self.assertRaises(pickle.UnpicklingError):
                convert_dataset(source, archive, root / "output")

    @staticmethod
    def _write_graph(path: Path, nodes: dict | None = None) -> None:
        edge_index = torch.tensor([[0, 1, 2, 0, 4], [1, 2, 3, 4, 4]], dtype=torch.long)
        edge_types = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
        node_types = torch.zeros(6, dtype=torch.long)
        if nodes is None:
            nodes = {
                index: {"id": index, "type": "concept", "name": name, "source": "fixture", "details": ""}
                for index, name in enumerate(["topic", "duplicate", "middle", "far", "duplicate", "unreachable"])
            }
        with zipfile.ZipFile(path, "w") as archive:
            for member, value in (
                ("prime/edge_index.pt", edge_index),
                ("prime/edge_types.pt", edge_types),
                ("prime/node_types.pt", node_types),
            ):
                buffer = io.BytesIO()
                torch.save(value, buffer)
                archive.writestr(member, buffer.getvalue())
            archive.writestr("prime/node_info.pkl", pickle.dumps(nodes))
            archive.writestr("prime/node_type_dict.pkl", pickle.dumps({"concept": 0}))
            archive.writestr("prime/edge_type_dict.pkl", pickle.dumps({"rel-a": 0, "rel-b": 1}))


if __name__ == "__main__":
    unittest.main()
