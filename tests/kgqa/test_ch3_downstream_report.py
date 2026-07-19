import json
import tempfile
import unittest
from pathlib import Path

from experiments.ch3.downstream_qa import CONDITION_IDS
from experiments.ch3.downstream_report import write_report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class TestCh3DownstreamReport(unittest.TestCase):
    def test_report_includes_same_sample_path_metrics_and_qa_micro_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conditions = []
            input_paths = {}
            input_info = {}
            for condition_id in CONDITION_IDS:
                input_path = root / f"{condition_id}.jsonl"
                write_jsonl(input_path, [{
                    "sample_index": 0, "question": "q", "topics": ["s"], "hop": 1,
                    "golden": ["a"],
                    "mmr_reason_paths": [{"path": [["s", "r", "a"]], "log_score": -1.0}],
                }])
                conditions.append({"id": condition_id, "label": condition_id, "method": {}})
                input_paths[condition_id] = input_path
                input_info[condition_id] = {"input": {"sha256": condition_id}}
                summary_path = root / "runs" / condition_id / "eval" / "summary.json"
                summary_path.parent.mkdir(parents=True)
                summary_path.write_text(json.dumps({"overall": {
                    "n": 1, "hit1": 1.0, "hit_any": 1.0, "macro_p": 1.0,
                    "macro_r": 1.0, "macro_f1": 1.0, "micro_p": 1.0,
                    "micro_r": 1.0, "micro_f1": 1.0, "exact_match": 1.0,
                }}), encoding="utf-8")

            report_dir = root / "reports"
            matrix = write_report(
                config={"dataset": "webqsp", "backbone": "transfernet", "config_id": "v1",
                        "_profile_path": "profile.json", "evaluation": {}, "conditions": conditions},
                input_info=input_info, input_paths=input_paths, layer_dir=root / "runs", report_dir=report_dir,
            )
            self.assertIsNone(matrix["conditions"][0]["path"])
            self.assertEqual(matrix["conditions"][1]["path"]["answer_hit"], 1.0)
            self.assertEqual(matrix["conditions"][1]["qa"]["micro_f1"], 1.0)
            summary = (report_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Path Answer Hit", summary)
            self.assertIn("QA Micro-F1", summary)
