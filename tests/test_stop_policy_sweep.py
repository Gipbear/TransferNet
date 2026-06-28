import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.common import (
    build_eval_record,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    llm_produced_answers,
    summarize_checked_batch_records,
)
from oh_my_agent.common.qa_data import WebQSPQASample
from oh_my_agent.tools.answer_with_paths import AnswerWithPathsToolResult
from oh_my_agent.tools.cited_path_check import CitedPathCheckResult
from oh_my_agent.tools.path_retrieve import PathRetrieveToolResult
from scripts import sweep_stop_policies


class _PathTool:
    def __init__(self, result, entity_map):
        self.result = result
        self.entity_map = entity_map

    def __call__(self, *args, **kwargs):
        return self.result


class _AnswerTool:
    def __init__(self):
        self.cursor = 0
        self.scripts = [
            {"answers": ["A", "B", "C"], "cited": [1, 2, 3]},
            {"answers": ["D"], "cited": [1]},
        ]

    def __call__(self, question, batch_named, **kwargs):
        script = self.scripts[self.cursor]
        return AnswerWithPathsToolResult(
            prompt="prompt",
            raw_output="output",
            answer_names=list(script["answers"]),
            cited_path_indices=list(script["cited"]),
            format_ok=True,
            used_adapter=True,
            tokens_generated=1,
            elapsed_ms=1.0,
        )


class _CheckTool:
    def __init__(self, answer_tool):
        self.answer_tool = answer_tool
        self.accepts = [{1}, {1}]

    def __call__(self, question, batch_named, *, cited_path_indices, raw_paths, **kwargs):
        cursor = self.answer_tool.cursor
        self.answer_tool.cursor += 1
        cited = [idx for idx in cited_path_indices if 0 < idx <= len(batch_named)]
        accepted = [idx for idx in cited if idx in self.accepts[cursor]]
        return CitedPathCheckResult(
            question=question,
            cited_path_indices=cited,
            accepted_path_indices=accepted,
            total_tokens_generated=1,
            total_elapsed_ms=1.0,
        )


def _record_from_result(result):
    sample = WebQSPQASample(
        question_raw=result.question,
        question=result.question,
        topic_mid=result.topic_mid,
        gold_mids=["m.a", "m.d"],
    )
    answer_metrics = compute_answer_metrics(result.pred_answer_disambiguated_mids, sample.gold_mids)
    faith = compute_faithfulness(
        cited_indices=set(result.final_accepted_path_indices)
        | set(result.relation_expanded_path_indices),
        golden_indices=label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids),
        pred_answers=llm_produced_answers(
            result.pred_answer_names,
            result.pred_answer_disambiguated_mids,
            result.large_answer_expanded_mids,
        ),
        path_entities=get_all_path_entities(result.named_mmr_reason_paths),
    )
    return build_eval_record(0, sample, result, answer_metrics, faith)


class StopPolicySweepTests(unittest.TestCase):
    def test_refuses_expansion_replay_when_group_tails_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            record = {
                "sample_index": 0,
                "question": "q",
                "topic_mid": "m.topic",
                "gold_mids": ["m.a"],
                "iterations": [],
            }
            (source_dir / "checked_batch_eval.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (source_dir / "checked_batch_eval_summary.json").write_text(
                json.dumps(
                    {
                        "batch_size": 20,
                        "check_mode": "hybrid-reject-list",
                        "large_answer_expansion": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text("m.topic\tTopic\nm.a\tA\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as ctx:
                sweep_stop_policies.main(
                    [
                        "--source_dir",
                        str(source_dir),
                        "--output_dir",
                        str(tmp_path / "sweep"),
                        "--entity_map",
                        str(entity_map_path),
                        "--mixed_stop_ratios",
                        "1/3",
                        "--max_batches",
                        "all",
                        "--all_wrong_modes",
                        "on",
                    ]
                )

        self.assertIn("lack group_tails", str(ctx.exception))

    def test_offline_sweep_can_compare_earlier_mixed_stop_with_keep_going(self):
        entity_map = {
            "m.topic": "Topic",
            "m.a": "A",
            "m.b": "B",
            "m.c": "C",
            "m.d": "D",
        }
        raw_paths = [
            {"path": [["m.topic", "rel.a", "m.a"]], "log_score": -1.0},
            {"path": [["m.topic", "rel.b", "m.b"]], "log_score": -2.0},
            {"path": [["m.topic", "rel.c", "m.c"]], "log_score": -3.0},
            {"path": [["m.topic", "rel.d", "m.d"]], "log_score": -4.0},
        ]
        named_paths = [
            {"path": [["Topic", "rel.a", "A"]], "log_score": -1.0},
            {"path": [["Topic", "rel.b", "B"]], "log_score": -2.0},
            {"path": [["Topic", "rel.c", "C"]], "log_score": -3.0},
            {"path": [["Topic", "rel.d", "D"]], "log_score": -4.0},
        ]
        retrieval = PathRetrieveToolResult(
            question="where is example from",
            topic_mid="m.topic",
            hop=1,
            raw_topics=["m.topic"],
            named_topics=["Topic"],
            raw_mmr_reason_paths=raw_paths,
            named_mmr_reason_paths=named_paths,
            raw_prediction={},
            named_prediction={},
            elapsed_ms=1.0,
            group_tails={},
        )
        answer_tool = _AnswerTool()
        result = CheckedBatchWebQAgent(
            path_tool=_PathTool(retrieval, entity_map),
            answer_tool=answer_tool,
            check_tool=_CheckTool(answer_tool),
        ).run(
            retrieval.question,
            retrieval.topic_mid,
            batch_size=3,
            no_early_stop=True,
            enable_relation_expansion=False,
            drop_topic_self=False,
        )
        source_record = _record_from_result(result)
        self.assertEqual(len(source_record["iterations"]), 2)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            (source_dir / "checked_batch_eval.jsonl").write_text(
                json.dumps(source_record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            summary = summarize_checked_batch_records([source_record])
            summary.update(
                {
                    "batch_size": 3,
                    "check_mode": "reject-answer-list",
                    "relation_expansion": False,
                    "score_margin": None,
                    "hop_filter": False,
                    "large_answer_expansion": False,
                }
            )
            (source_dir / "checked_batch_eval_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False), encoding="utf-8"
            )
            entity_map_path = tmp_path / "mapped_entities.txt"
            entity_map_path.write_text(
                "\n".join(f"{mid}\t{name}" for mid, name in entity_map.items()) + "\n",
                encoding="utf-8",
            )
            out_dir = tmp_path / "sweep"

            exit_code = sweep_stop_policies.main(
                [
                    "--source_dir",
                    str(source_dir),
                    "--output_dir",
                    str(out_dir),
                    "--entity_map",
                    str(entity_map_path),
                    "--mixed_stop_ratios",
                    "1/3,off",
                    "--max_batches",
                    "all",
                    "--all_wrong_modes",
                    "on",
                    "--no_new_batches",
                    "none",
                    "--topic_guard",
                    "off",
                ]
            )

            self.assertEqual(exit_code, 0)
            mixed_summary = json.loads(
                (out_dir / "mix0p333333_maxall_awon_nonewoff" / "checked_batch_eval_summary.json")
                .read_text(encoding="utf-8")
            )
            keep_summary = json.loads(
                (out_dir / "mixoff_maxall_awon_nonewoff" / "checked_batch_eval_summary.json")
                .read_text(encoding="utf-8")
            )

        self.assertTrue(mixed_summary["complete_support"])
        self.assertEqual(mixed_summary["avg_batches_used"], 1.0)
        self.assertEqual(mixed_summary["macro_f1"], 0.6667)
        self.assertEqual(keep_summary["avg_batches_used"], 2.0)
        self.assertEqual(keep_summary["macro_f1"], 1.0)
        self.assertEqual(keep_summary["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
