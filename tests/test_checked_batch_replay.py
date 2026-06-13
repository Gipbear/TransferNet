"""离线回放确定性后处理:从 canonical 录制的 jsonl 复现消融阶梯,不重跑 LLM。

核心保证:**replay(record, 配置X) 必须逐位等于 真实 agent.run(配置X)**。
检索、答题、check 都与后处理标志无关(后处理在 check 之后、确定性),所以同一份
录制可复现 base / +margin / +hop / +expansion / +guard 全部档位,省掉 4 次全量 LLM 跑。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.agent.checked_batch_replay import replay_record
from oh_my_agent.common.eval_records import build_eval_record
from oh_my_agent.common import get_all_path_entities
from oh_my_agent.common.metrics import (
    compute_answer_metrics,
    compute_faithfulness,
    label_golden_indices,
    llm_produced_answers,
)
from oh_my_agent.common.qa_data import WebQSPQASample
from oh_my_agent.tools.answer_with_paths import AnswerWithPathsToolResult
from oh_my_agent.tools.cited_path_check import CitedPathCheckResult
from oh_my_agent.tools.path_retrieve import PathRetrieveToolResult


# ---- 脚本化(fake)工具:生成"真实运行"的录制,answer/check 与后处理标志无关 ----
class _ScriptedPath:
    def __init__(self, result: PathRetrieveToolResult, entity_map: dict[str, str]):
        self._result = result
        self.entity_map = entity_map

    def __call__(self, *args, **kwargs) -> PathRetrieveToolResult:
        return self._result


class _ScriptedAnswer:
    """按批次顺序返回脚本好的答案;不前进游标(check 才前进,与回放一致)。"""

    def __init__(self, scripts: list[dict]):
        self.scripts = scripts
        self.cursor = 0

    def __call__(self, question, batch_named, **kwargs) -> AnswerWithPathsToolResult:
        script = self.scripts[self.cursor]
        return AnswerWithPathsToolResult(
            prompt="prompt",
            raw_output="output",
            answer_names=list(script["answer_names"]),
            cited_path_indices=list(script["cited_local"]),
            format_ok=True,
            used_adapter=True,
            tokens_generated=1,
            elapsed_ms=1.0,
        )


class _ScriptedCheck:
    """accept = cited 中落在脚本 accept 集合里的;返回后前进游标。"""

    def __init__(self, answer_tool: _ScriptedAnswer, accepts: list[set[int]]):
        self.answer_tool = answer_tool
        self.accepts = accepts

    def __call__(self, question, batch_named, *, cited_path_indices, raw_paths, **kwargs):
        cursor = self.answer_tool.cursor
        local_cited = sorted(i for i in cited_path_indices if 0 < i <= len(batch_named))
        local_accept = [i for i in local_cited if i in self.accepts[cursor]]
        self.answer_tool.cursor += 1
        return CitedPathCheckResult(
            question=question,
            cited_path_indices=local_cited,
            accepted_path_indices=local_accept,
            total_tokens_generated=1,
            total_elapsed_ms=1.0,
        )


def _named(path):
    return {"path": [[h, r, t] for h, r, t in path["edges"]], "log_score": path["score"]}


def _build_scenario():
    """4 条路径 / 2 批:Alpha/Beta(1 跳)、Gamma(2 跳)、Topic(自指)。"""
    entity_map = {
        "m.topic": "Topic", "m.a": "Alpha", "m.b": "Beta",
        "m.c": "Gamma", "m.x": "Mid",
    }
    raw_paths = [
        {"path": [["m.topic", "r1", "m.a"]], "log_score": -1.0},
        {"path": [["m.topic", "r1", "m.b"]], "log_score": -1.0},
        {"path": [["m.topic", "r2", "m.x"], ["m.x", "r3", "m.c"]], "log_score": -2.0},
        {"path": [["m.topic", "r1", "m.topic"]], "log_score": -1.0},
    ]
    named_paths = [
        {"path": [["Topic", "r1", "Alpha"]], "log_score": -1.0},
        {"path": [["Topic", "r1", "Beta"]], "log_score": -1.0},
        {"path": [["Topic", "r2", "Mid"], ["Mid", "r3", "Gamma"]], "log_score": -2.0},
        {"path": [["Topic", "r1", "Topic"]], "log_score": -1.0},
    ]
    retrieval = PathRetrieveToolResult(
        question="who is topic",
        topic_mid="m.topic",
        hop=1,
        raw_topics=["m.topic"],
        named_topics=["Topic"],
        raw_mmr_reason_paths=raw_paths,
        named_mmr_reason_paths=named_paths,
        raw_prediction={"m.a": 0.95, "m.b": 0.95, "m.c": 0.95, "m.topic": 0.95},
        named_prediction={"Alpha": 0.95},
        elapsed_ms=1.0,
        group_tails={},
    )
    answer_scripts = [
        {"answer_names": ["Alpha", "Beta"], "cited_local": [1, 2]},
        {"answer_names": ["Gamma", "Topic"], "cited_local": [1, 2]},
    ]
    accepts = [{1, 2}, {1, 2}]
    return entity_map, retrieval, answer_scripts, accepts


def _run_real(entity_map, retrieval, answer_scripts, accepts, *, batch_size=2, **flags):
    answer_tool = _ScriptedAnswer(answer_scripts)
    agent = CheckedBatchWebQAgent(
        path_tool=_ScriptedPath(retrieval, entity_map),
        answer_tool=answer_tool,
        check_tool=_ScriptedCheck(answer_tool, accepts),
        check_tool_after_first=None,
    )
    return agent.run(
        retrieval.question, retrieval.topic_mid, batch_size=batch_size, **flags
    )


def _to_record(result):
    sample = WebQSPQASample(
        question_raw="who is topic",
        question="who is topic",
        topic_mid="m.topic",
        gold_mids=["m.a", "m.b"],
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


# 阶梯各档(键 -> 传给 run/replay 的 flags),全部 enable_relation_expansion 默认 True
LADDER = {
    "base": dict(drop_topic_self=False),
    "margin": dict(score_margin=4.0, drop_topic_self=False),
    "margin_hop": dict(score_margin=4.0, hop_filter=True, drop_topic_self=False),
    "canonical": dict(score_margin=4.0, hop_filter=True, drop_topic_self=True),
}


class ReplayEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.scenario = _build_scenario()
        # 录制源 = canonical 全开跑一遍
        canonical_result = _run_real(*self.scenario, **LADDER["canonical"])
        self.record = _to_record(canonical_result)

    def test_record_carries_group_tails(self):
        self.assertIn("group_tails", self.record)

    def test_replay_matches_real_run_for_each_ladder_config(self):
        entity_map = self.scenario[0]
        for name, flags in LADDER.items():
            with self.subTest(config=name):
                real = _run_real(*self.scenario, **flags)
                replayed = replay_record(self.record, entity_map=entity_map, batch_size=2, **flags)
                self.assertEqual(
                    replayed.pred_answer_disambiguated_mids,
                    real.pred_answer_disambiguated_mids,
                    msg=f"{name}: disambiguated mids mismatch",
                )
                self.assertEqual(replayed.pred_answer_names, real.pred_answer_names)
                self.assertEqual(
                    replayed.final_accepted_path_indices, real.final_accepted_path_indices
                )
                self.assertEqual(
                    replayed.large_answer_expanded_mids, real.large_answer_expanded_mids
                )

    def test_ladder_configs_produce_distinct_answers(self):
        # 保证场景确实区分各档(否则等价测试是空的)
        entity_map = self.scenario[0]
        answers = {
            name: tuple(replay_record(self.record, entity_map=entity_map, batch_size=2, **flags)
                        .pred_answer_disambiguated_mids)
            for name, flags in LADDER.items()
        }
        self.assertEqual(set(answers["base"]), {"m.a", "m.b", "m.c", "m.topic"})
        self.assertNotIn("m.c", answers["margin_hop"])      # hop 过滤掉 2 跳 Gamma
        self.assertNotIn("m.topic", answers["canonical"])   # guard 过滤掉自指 Topic


class ReplayExpansionTests(unittest.TestCase):
    def _expansion_scenario(self):
        entity_map = {f"m.{i}": f"E{i}" for i in range(1, 11)}
        entity_map["m.topic"] = "Topic"
        raw_paths = [
            {"path": [["m.topic", "r1", f"m.{i}"]], "log_score": -1.0}
            for i in range(1, 9)
        ]
        named_paths = [
            {"path": [["Topic", "r1", f"E{i}"]], "log_score": -1.0}
            for i in range(1, 9)
        ]
        prediction = {f"m.{i}": 0.95 for i in range(1, 11)}
        retrieval = PathRetrieveToolResult(
            question="list the things of topic",
            topic_mid="m.topic",
            hop=1,
            raw_topics=["m.topic"],
            named_topics=["Topic"],
            raw_mmr_reason_paths=raw_paths,
            named_mmr_reason_paths=named_paths,
            raw_prediction=prediction,
            named_prediction={},
            elapsed_ms=1.0,
            group_tails={"m.topic|r1": [f"m.{i}" for i in range(1, 11)]},
        )
        answer_scripts = [{"answer_names": [f"E{i}" for i in range(1, 9)],
                           "cited_local": list(range(1, 9))}]
        accepts = [set(range(1, 9))]
        return entity_map, retrieval, answer_scripts, accepts

    def test_replay_reproduces_large_answer_expansion(self):
        scenario = self._expansion_scenario()
        flags = dict(large_answer_expansion=True, expansion_top_groups=1)
        real = _run_real(*scenario, batch_size=20, **flags)
        record = _to_record_expansion(real)
        replayed = replay_record(record, entity_map=scenario[0], batch_size=20, **flags)
        # m.9, m.10 是组内但未被引用的 KG 尾,应被 expansion 补出
        self.assertEqual(real.large_answer_expanded_mids, ["m.9", "m.10"])
        self.assertEqual(
            replayed.large_answer_expanded_mids, real.large_answer_expanded_mids
        )
        self.assertEqual(
            replayed.pred_answer_disambiguated_mids, real.pred_answer_disambiguated_mids
        )


def _to_record_expansion(result):
    sample = WebQSPQASample(
        question_raw="list the things of topic",
        question="list the things of topic",
        topic_mid="m.topic",
        gold_mids=["m.1"],
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


if __name__ == "__main__":
    unittest.main()
