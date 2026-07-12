"""gatev2 回放 parity 常驻护栏:kgqa.agent.replay 抽样回放 full_trace,
逐记录与 Ch5 终版 score2_hopoff_top3_max2_gatev2 官方产物逐位一致。

全量 1581 条核验已在 Task 6 一次性通过(2026-07-12);本测试固化 ≥50 条
等距抽样,防止 kgqa/agent 后处理链路未来回归。依赖 gitignored 实验产物,
按 tests/kgqa/integration.py 惯例显式启用:RUN_KGQA_ARTIFACT_TESTS=1。
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.kgqa.integration import ARTIFACT_TEST_SKIP_REASON, artifact_test_available

RERUN = ROOT / "data/output/WebQSP/checked_batch_agent/ch5_full_rerun_20260627_2306"
TRACE_PATH = RERUN / "full_trace/checked_batch_eval.jsonl"
OFFICIAL_PATH = RERUN / "score2_hopoff_top3_max2_gatev2/checked_batch_eval.jsonl"
ENTITY_MAP_PATH = ROOT / "data/resources/WebQSP/fbwq_full/mapped_entities.txt"

# gatev2 终版配置(表5-3 PV-GAC 行;与官方 summary 及 scripts/_sweep_ch5_thresholds.py 一致)
GATEV2_FLAGS = dict(
    score_margin=2.0,
    hop_filter=False,
    large_answer_expansion=True,
    enable_relation_expansion=True,
    drop_topic_self=True,
    mixed_stop_ratio=0.5,
    max_batches=2,
)
BATCH_SIZE = 20
EXPANSION_MIN_ANSWERS = 8
EXPANSION_TOP_GROUPS = 3

SAMPLE_COUNT = 53  # 等距抽样条数(≥50)


def _sampled_lines(path: Path, indices: set[int]) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in indices and line.strip():
                records[i] = json.loads(line)
    return records


@unittest.skipUnless(
    artifact_test_available(str(TRACE_PATH), str(OFFICIAL_PATH), str(ENTITY_MAP_PATH)),
    ARTIFACT_TEST_SKIP_REASON,
)
class GateV2ReplayParityTests(unittest.TestCase):
    def test_sampled_replay_matches_official_gatev2_records(self):
        from kgqa.agent.common import (
            build_eval_record,
            cited_indices_for_answers,
            compute_answer_metrics,
            compute_faithfulness,
            get_all_path_entities,
            label_golden_indices,
            llm_produced_answers,
            load_entity_map,
        )
        from kgqa.agent.common.qa_data import WebQSPQASample
        from kgqa.agent.replay import _ReplaySession

        total = 1581
        stride = total // SAMPLE_COUNT
        indices = set(range(0, total, stride))
        traces = _sampled_lines(TRACE_PATH, indices)
        officials = _sampled_lines(OFFICIAL_PATH, indices)
        self.assertGreaterEqual(len(traces), 50)
        self.assertEqual(sorted(traces), sorted(officials))

        session = _ReplaySession(load_entity_map(str(ENTITY_MAP_PATH)), hybrid_check=True)
        for i in sorted(traces):
            rec, ref = traces[i], officials[i]
            sample = WebQSPQASample(
                question_raw=rec.get("question_raw", rec.get("question", "")),
                question=rec.get("question", ""),
                topic_mid=rec.get("topic_mid", ""),
                gold_mids=list(rec.get("gold_mids", [])),
            )
            result = session.replay(
                rec,
                allow_prefix=True,
                batch_size=BATCH_SIZE,
                expansion_min_answers=EXPANSION_MIN_ANSWERS,
                expansion_top_groups=EXPANSION_TOP_GROUPS,
                **GATEV2_FLAGS,
            )
            answer_metrics = compute_answer_metrics(
                result.pred_answer_disambiguated_mids, sample.gold_mids
            )
            faith = compute_faithfulness(
                cited_indices=cited_indices_for_answers(
                    set(result.final_accepted_path_indices)
                    | set(result.relation_expanded_path_indices),
                    result.raw_mmr_reason_paths,
                    result.pred_answer_disambiguated_mids,
                ),
                golden_indices=label_golden_indices(
                    result.raw_mmr_reason_paths, sample.gold_mids
                ),
                pred_answers=llm_produced_answers(
                    result.pred_answer_names,
                    result.pred_answer_disambiguated_mids,
                    result.large_answer_expanded_mids,
                ),
                path_entities=get_all_path_entities(result.named_mmr_reason_paths),
            )
            mine = build_eval_record(
                rec.get("sample_index", 0), sample, result, answer_metrics, faith
            )
            self.assertEqual(
                json.dumps(mine, sort_keys=True, ensure_ascii=False),
                json.dumps(ref, sort_keys=True, ensure_ascii=False),
                msg=f"line {i}(sample_index={ref.get('sample_index')})回放与官方 gatev2 记录不一致",
            )


if __name__ == "__main__":
    unittest.main()
