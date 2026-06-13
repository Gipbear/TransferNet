"""离线回放 checked-batch agent 的确定性后处理。

检索 / 答题 / check 的结果与后处理标志(score_margin / hop_filter /
large_answer_expansion / topic guard)无关——后处理在 check 之后、且完全确定性。
因此一份录制好的 canonical jsonl 可以复现整条消融阶梯,无需重跑 LLM。

实现方式:把录制的检索/答题/check 结果包成 **mock 工具**喂给**真实**
``CheckedBatchWebQAgent.run``,后处理逻辑(margin/hop/expansion/guard/
relation_expansion)100% 复用线上代码,杜绝"另写一份离线后处理"导致的口径漂移。
"""

from __future__ import annotations

from typing import Any

from .checked_batch_webqsp_agent import CheckedBatchWebQAgent, CheckedBatchWebQAgentResult
from oh_my_agent.tools.answer_with_paths import AnswerWithPathsToolResult
from oh_my_agent.tools.path_retrieve import PathRetrieveToolResult


class _BatchCursor:
    """answer 与 check 在同一批共享游标;check 调用后前进一格。"""

    def __init__(self, iterations: list[dict[str, Any]]) -> None:
        self.iterations = iterations
        self.index = 0

    def current(self) -> dict[str, Any]:
        if self.index >= len(self.iterations):
            raise IndexError(
                "回放请求的批次数超过录制 iterations 数——配置改变了分批/早停行为,"
                "无法用该录制回放(只有与 canonical 相同 check/检索/答题的配置可回放)"
            )
        return self.iterations[self.index]


class _ReplayPathTool:
    def __init__(self, record: dict[str, Any], entity_map: dict[str, str]) -> None:
        self.entity_map = entity_map
        self._result = PathRetrieveToolResult(
            question=record.get("question", ""),
            topic_mid=record.get("topic_mid", ""),
            hop=record.get("hop", 1),
            raw_topics=record.get("raw_topics", []),
            named_topics=record.get("named_topics", []),
            raw_mmr_reason_paths=record.get("raw_mmr_reason_paths", []),
            named_mmr_reason_paths=record.get("named_mmr_reason_paths", []),
            raw_prediction=record.get("raw_prediction", {}),
            named_prediction=record.get("named_prediction", {}),
            elapsed_ms=record.get("retrieval_elapsed_ms", 0.0),
            group_tails=record.get("group_tails", {}) or {},
        )

    def __call__(self, *args, **kwargs) -> PathRetrieveToolResult:
        return self._result


class _ReplayAnswerTool:
    def __init__(self, cursor: _BatchCursor) -> None:
        self.cursor = cursor

    def __call__(self, question, batch_named, **kwargs) -> AnswerWithPathsToolResult:
        it = self.cursor.current()
        return AnswerWithPathsToolResult(
            prompt=it.get("answer_prompt", ""),
            raw_output=it.get("raw_llm_output", ""),
            answer_names=list(it.get("answer_names", [])),
            cited_path_indices=[],  # 仅作为真实 check 的输入;此处 check 被 mock,无用
            format_ok=it.get("format_ok", False),
            used_adapter=it.get("used_adapter", False),
            tokens_generated=it.get("answer_tokens_generated", 0),
            elapsed_ms=it.get("answer_elapsed_ms", 0.0),
        )


class _ReplayCheckResult:
    """鸭子类型的 check 结果:暴露 agent 用到的字段 + to_dict 原样回放 path_check。"""

    def __init__(self, path_check: dict[str, Any]) -> None:
        self._path_check = path_check
        self.cited_path_indices = list(path_check.get("cited_path_indices", []))
        self.accepted_path_indices = list(path_check.get("accepted_path_indices", []))
        self.total_tokens_generated = path_check.get("total_tokens_generated", 0)
        self.total_elapsed_ms = path_check.get("total_elapsed_ms", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return self._path_check


class _ReplayCheckTool:
    def __init__(self, cursor: _BatchCursor) -> None:
        self.cursor = cursor

    def __call__(self, question, batch_named, **kwargs) -> _ReplayCheckResult:
        it = self.cursor.current()
        self.cursor.index += 1
        return _ReplayCheckResult(it.get("path_check", {}))


def replay_record(
    record: dict[str, Any],
    *,
    entity_map: dict[str, str],
    batch_size: int = 20,
    score_margin: float | None = None,
    hop_filter: bool = False,
    large_answer_expansion: bool = False,
    drop_topic_self: bool = True,
    enable_relation_expansion: bool = True,
    expansion_min_answers: int = 8,
    expansion_top_groups: int = 1,
    no_early_stop: bool = False,
) -> CheckedBatchWebQAgentResult:
    """用录制的 LLM 轨迹复现一条样本,在给定后处理配置下重新产出结果。

    ``batch_size`` 必须与录制时一致,否则分批切片与 iterations 对不齐——返回前会
    校验各批 ``batch_start_rank`` 与录制吻合,不吻合即抛错。
    """
    cursor = _BatchCursor(record.get("iterations", []))
    agent = CheckedBatchWebQAgent(
        path_tool=_ReplayPathTool(record, entity_map),
        answer_tool=_ReplayAnswerTool(cursor),
        check_tool=_ReplayCheckTool(cursor),
        check_tool_after_first=None,
    )
    result = agent.run(
        record.get("question", ""),
        record.get("topic_mid", ""),
        batch_size=batch_size,
        score_margin=score_margin,
        hop_filter=hop_filter,
        large_answer_expansion=large_answer_expansion,
        drop_topic_self=drop_topic_self,
        enable_relation_expansion=enable_relation_expansion,
        expansion_min_answers=expansion_min_answers,
        expansion_top_groups=expansion_top_groups,
        no_early_stop=no_early_stop,
    )
    _assert_batch_alignment(result, record)
    return result


def _assert_batch_alignment(result: CheckedBatchWebQAgentResult, record: dict[str, Any]) -> None:
    recorded = record.get("iterations", [])
    if len(result.iterations) != len(recorded):
        raise ValueError(
            f"回放批次数({len(result.iterations)})与录制({len(recorded)})不一致:"
            "batch_size 或早停行为改变,回放无效"
        )
    for replayed_it, recorded_it in zip(result.iterations, recorded):
        if replayed_it.batch_start_rank != recorded_it.get("batch_start_rank"):
            raise ValueError(
                "回放分批与录制不对齐(batch_start_rank 不符);请确认 batch_size 与录制一致"
            )
