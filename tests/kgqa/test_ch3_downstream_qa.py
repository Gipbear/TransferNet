import json
import tempfile
import unittest
from pathlib import Path

from experiments.ch3_downstream_qa import (
    build_eval_command,
    load_downstream_config,
    validate_condition_inputs,
    write_stratified_smoke_inputs,
)


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def make_config(root: Path) -> Path:
    profile = write_json(root / "profile.json", {
        "kind": "ch3_retrieval_profile", "status": "confirmed", "dataset": "webqsp",
        "backbone": "transfernet", "config_id": "v1", "topk": 500, "retrieve": {},
    })
    rows = [
        {"question_raw": "q1", "golden": ["a"], "paths": []},
        {"question_raw": "q2", "golden": ["b"], "paths": []},
    ]
    conditions = []
    methods = {
        "no_path": {"no_paths": True},
        "shortest_path": {"method": "shortest_path_postprocess"},
        "score_beam": {"beam_size": 20, "lambda_val": 0.0, "eta": 0.0},
        "terminal_score_beam": {"beam_size": 20, "lambda_val": 0.0, "eta": 1.0},
        "tarrs": {"beam_size": 20, "lambda_val": 0.2, "eta": 1.0},
    }
    for condition_id, method in methods.items():
        input_path = write_jsonl(root / f"{condition_id}.jsonl", rows)
        condition = {"id": condition_id, "label": condition_id, "input": str(input_path), "method": method}
        if condition_id == "no_path":
            condition["no_paths"] = True
        conditions.append(condition)
    return write_json(root / "config.json", {
        "kind": "ch3_downstream_qa", "dataset": "webqsp", "backbone": "transfernet",
        "config_id": "v1", "profile": str(profile),
        "evaluation": {
            "model": "model", "format": "v2", "path_format": "chain", "entity_repr": "name",
            "max_new_tokens": 256, "batch_size": 4, "path_budget": 20,
        },
        "conditions": conditions, "fixed_pfit_adapter": None,
    })


class TestCh3DownstreamQa(unittest.TestCase):
    def test_validate_inputs_returns_identical_qa_signatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_downstream_config(make_config(root), root)
            inputs = validate_condition_inputs(config, root)
            self.assertEqual(set(inputs), {"no_path", "shortest_path", "score_beam", "terminal_score_beam", "tarrs"})
            self.assertEqual({item["samples"] for item in inputs.values()}, {2})
            self.assertEqual(len({item["qa_signature"] for item in inputs.values()}), 1)

    def test_rejects_terminal_aware_as_ordinary_score_beam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = make_config(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            next(item for item in config["conditions"] if item["id"] == "score_beam")["method"]["eta"] = 1.5
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "score_beam 的方法定义"):
                load_downstream_config(config_path, root)

    def test_rejects_misaligned_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = make_config(root)
            config = load_downstream_config(config_path, root)
            write_jsonl(root / "tarrs.jsonl", [
                {"question_raw": "q1", "golden": ["wrong"], "paths": []},
                {"question_raw": "q2", "golden": ["b"], "paths": []},
            ])
            with self.assertRaisesRegex(ValueError, "题目或 golden"):
                validate_condition_inputs(config, root)

    def test_no_path_command_is_explicit_and_adapter_is_optional(self):
        command = build_eval_command(
            dataset="webqsp", condition={"id": "no_path"},
            evaluation={"format": "v2", "path_format": "chain", "entity_repr": "name", "model": "model", "max_new_tokens": 256, "batch_size": 4},
            input_path=Path("input.jsonl"), exp_dir=Path("out"), adapter=None, run_dir=Path("out"),
        )
        self.assertIn("--no_paths", command)
        self.assertNotIn("--adapter", command)

    def test_stratified_smoke_inputs_keep_both_hops_and_conditions_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = make_config(root)
            config_json = json.loads(config_path.read_text(encoding="utf-8"))
            rows = [
                {"question": "q1", "golden": ["a"], "hop": 1, "mmr_reason_paths": []},
                {"question": "q2", "golden": ["b"], "hop": 1, "mmr_reason_paths": []},
                {"question": "q3", "golden": ["c"], "hop": 2, "mmr_reason_paths": []},
                {"question": "q4", "golden": ["d"], "hop": 2, "mmr_reason_paths": []},
            ]
            for condition in config_json["conditions"]:
                write_jsonl(Path(condition["input"]), rows)
            config = load_downstream_config(config_path, root)
            outputs = write_stratified_smoke_inputs(config, root, root / "smoke", 2)
            selected = [json.loads(line) for line in outputs["tarrs"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["hop"] for row in selected], [1, 2])
            self.assertEqual(
                outputs["score_beam"].read_text(encoding="utf-8"),
                outputs["tarrs"].read_text(encoding="utf-8"),
            )
