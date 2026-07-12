"""离线回放 checked-batch agent 的确定性后处理。

检索 / 答题 / check 的结果与后处理标志(score_margin / hop_filter /
large_answer_expansion / topic guard)无关——后处理在 check 之后、且完全确定性。
因此一份录制好的 canonical jsonl 可以复现整条消融阶梯,无需重跑 LLM。

实现方式:把录制的检索/答题/check 结果包成 **mock 工具**喂给**真实**
``CheckedBatchAgent.run``,后处理逻辑(margin/hop/expansion/guard/
relation_expansion)100% 复用线上代码,杜绝"另写一份离线后处理"导致的口径漂移。

性能:agent 的 ``__init__`` 会对整张 entity_map(WebQSP 近 400 万条)构建反向映射,
代价很大。``_ReplaySession`` **只建一次 agent**,mock 工具读 session 的可变状态
(当前记录的检索结果 + 批游标),把上千次回放摊销到一次反向映射构建上。
"""

from __future__ import annotations

from typing import Any, Optional

from .checked_batch import CheckedBatchAgent, CheckedBatchAgentResult
from kgqa.agent.tools.answer_with_paths import AnswerWithPathsToolResult
from kgqa.agent.tools.path_retrieve import PathRetrieveToolResult


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


def _path_result_from_record(
    record: dict[str, Any], entity_map: dict[str, str]
) -> PathRetrieveToolResult:
    return PathRetrieveToolResult(
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


class _SessionPathTool:
    def __init__(self, session: "_ReplaySession", entity_map: dict[str, str]) -> None:
        self._session = session
        self.entity_map = entity_map

    def __call__(self, *args, **kwargs) -> PathRetrieveToolResult:
        return self._session.path_result


class _SessionAnswerTool:
    def __init__(self, session: "_ReplaySession") -> None:
        self._session = session

    def __call__(self, question, batch_named, **kwargs) -> AnswerWithPathsToolResult:
        it = self._session.cursor.current()
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


class _SessionCheckTool:
    def __init__(self, session: "_ReplaySession") -> None:
        self._session = session

    def __call__(self, question, batch_named, **kwargs) -> _ReplayCheckResult:
        cursor = self._session.cursor
        it = cursor.current()
        cursor.index += 1
        return _ReplayCheckResult(it.get("path_check", {}))


class _ReplaySession:
    """复用一个 agent 回放多条记录,只建一次反向映射。

    ``hybrid_check`` 必须与录制时的 check_mode 一致(canonical 是 hybrid → True)。
    agent 的 all_wrong_after_answer 早停只在 check_tool_after_first 非 None 时生效;
    两个 check 入口指向同一个 mock(共享 session 游标),每批仅前进一格。
    """

    def __init__(self, entity_map: dict[str, str], *, hybrid_check: bool = True) -> None:
        self._entity_map = entity_map
        self.cursor: Optional[_BatchCursor] = None
        self.path_result: Optional[PathRetrieveToolResult] = None
        check_tool = _SessionCheckTool(self)
        self.agent = CheckedBatchAgent(
            path_tool=_SessionPathTool(self, entity_map),
            answer_tool=_SessionAnswerTool(self),
            check_tool=check_tool,
            check_tool_after_first=check_tool if hybrid_check else None,
        )

    def replay(
        self,
        record: dict[str, Any],
        *,
        allow_prefix: bool = False,
        **run_flags: Any,
    ) -> CheckedBatchAgentResult:
        self.cursor = _BatchCursor(record.get("iterations", []))
        self.path_result = _path_result_from_record(record, self._entity_map)
        result = self.agent.run(
            record.get("question", ""), record.get("topic_mid", ""), **run_flags
        )
        _assert_batch_alignment(result, record, allow_prefix=allow_prefix)
        return result


def replay_record(
    record: dict[str, Any],
    *,
    entity_map: dict[str, str],
    hybrid_check: bool = True,
    allow_prefix: bool = False,
    **run_flags: Any,
) -> CheckedBatchAgentResult:
    """便捷单条回放(每次新建 session;批量请用 ``_ReplaySession`` 复用 agent)。

    ``run_flags`` 透传给 ``agent.run``(batch_size / score_margin / hop_filter /
    large_answer_expansion / drop_topic_self / expansion_* 等)。``batch_size`` 必须
    与录制一致,否则分批与 iterations 对不齐——返回前会校验 batch_start_rank。

    ``hybrid_check`` 必须与录制时的 check_mode 一致(canonical 是 hybrid → True);
    它决定 agent 的 all_wrong_after_answer 早停是否生效,设错会导致回放越界报错。
    """
    return _ReplaySession(entity_map, hybrid_check=hybrid_check).replay(
        record, allow_prefix=allow_prefix, **run_flags
    )


def _assert_batch_alignment(
    result: CheckedBatchAgentResult,
    record: dict[str, Any],
    *,
    allow_prefix: bool = False,
) -> None:
    recorded = record.get("iterations", [])
    if allow_prefix:
        if len(result.iterations) > len(recorded):
            raise ValueError(
                f"回放批次数({len(result.iterations)})超过录制({len(recorded)}):"
                "源 trace 不足以支持该停止策略"
            )
    elif len(result.iterations) != len(recorded):
        raise ValueError(
            f"回放批次数({len(result.iterations)})与录制({len(recorded)})不一致:"
            "batch_size 或早停行为改变,回放无效"
        )
    for replayed_it, recorded_it in zip(result.iterations, recorded):
        if replayed_it.batch_start_rank != recorded_it.get("batch_start_rank"):
            raise ValueError(
                "回放分批与录制不对齐(batch_start_rank 不符);请确认 batch_size 与录制一致"
            )
