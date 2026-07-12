"""Batch-and-check KGQA agent using cached path retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from kgqa.agent.common import (
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
    tail_from_path_dict,
)
from kgqa.agent.common.metrics import norm_entity
from kgqa.agent.tools import (
    AnswerWithPathsTool,
    PathRetrieveTool,
    PathRetrieveToolResult,
    RejectedAnswerCheckTool,
)


@dataclass(frozen=True)
class CheckedBatchIteration:
    """Trace for one checked path batch."""

    batch_index: int
    batch_start_rank: int
    batch_end_rank: int
    batch_size: int
    local_cited_path_indices: list[int]
    global_cited_path_indices: list[int]
    accepted_path_indices: list[int]
    batch_status: str
    answer_prompt: str
    raw_llm_output: str
    answer_names: list[str]
    format_ok: bool
    used_adapter: bool
    answer_tokens_generated: int
    answer_elapsed_ms: float
    path_check: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckedBatchAgentResult:
    """End-to-end result for the checked batch QA flow."""

    question: str
    topic_mid: str
    hop: int
    raw_topics: list[str]
    named_topics: list[str]
    raw_mmr_reason_paths: list[dict[str, Any]]
    named_mmr_reason_paths: list[dict[str, Any]]
    raw_prediction: dict[str, float]
    named_prediction: dict[str, float]
    iterations: list[CheckedBatchIteration] = field(default_factory=list)
    final_accepted_path_indices: list[int] = field(default_factory=list)
    cited_path_indices: list[int] = field(default_factory=list)
    pred_answer_names: list[str] = field(default_factory=list)
    pred_answer_expanded_mids: list[str] = field(default_factory=list)
    pred_answer_disambiguated_mids: list[str] = field(default_factory=list)
    relation_expanded_path_indices: list[int] = field(default_factory=list)
    large_answer_expanded_mids: list[str] = field(default_factory=list)
    # 检索期算出的"(topic, 关系组) → 全 KG 尾"(prediction 过滤后),随 record 落盘,
    # 供离线回放复现 large_answer_expansion(否则离线无法复算组内补答)
    group_tails: dict[str, list[str]] = field(default_factory=dict)
    batches_used: int = 0
    checked_paths_count: int = 0
    accepted_paths_count: int = 0
    final_answer_count: int = 0
    stop_reason: str = "path_exhausted"
    format_ok: bool = True
    used_adapter: bool = True
    tokens_generated: int = 0
    answer_tokens_generated: int = 0
    check_tokens_generated: int = 0
    retrieval_elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    check_elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["iterations"] = [item.to_dict() for item in self.iterations]
        return data


def _path_entity_sequence(path_dict: dict[str, Any]) -> list[str]:
    edges = path_dict.get("path", [])
    if not edges:
        return []

    entities = [str(edges[0][0])]
    for edge in edges:
        if len(edge) < 3:
            continue
        head = str(edge[0])
        tail = str(edge[2])
        if norm_entity(entities[-1]) != norm_entity(head):
            entities.append(head)
        entities.append(tail)
    return entities


def _answer_pair_from_paths(
    named_path_dict: dict[str, Any],
    raw_path_dict: dict[str, Any] | None,
    *,
    answer_names: list[str],
) -> tuple[str, str]:
    raw_path_dict = raw_path_dict or named_path_dict
    named_answer = tail_from_path_dict(named_path_dict)
    raw_answer = tail_from_path_dict(raw_path_dict)

    answer_keys = {norm_entity(answer) for answer in answer_names if norm_entity(answer)}
    if not answer_keys:
        return named_answer, raw_answer

    named_entities = _path_entity_sequence(named_path_dict)
    raw_entities = _path_entity_sequence(raw_path_dict)
    for offset, named_entity in list(enumerate(named_entities))[1:] + list(enumerate(named_entities))[:1]:
        if norm_entity(named_entity) not in answer_keys:
            continue
        raw_entity = raw_entities[offset] if offset < len(raw_entities) else named_entity
        return named_entity, raw_entity

    return named_answer, raw_answer


def _tail_entity_count(paths: list[dict[str, Any]]) -> int:
    return len({norm_entity(tail_from_path_dict(path)) for path in paths if tail_from_path_dict(path)})


def _tail_entity_count_for_indices(raw_paths: list[dict[str, Any]], indices: set[int]) -> int:
    tails: set[str] = set()
    for index in indices:
        path_offset = index - 1
        if 0 <= path_offset < len(raw_paths):
            tail = tail_from_path_dict(raw_paths[path_offset])
            if tail:
                tails.add(norm_entity(tail))
    return len(tails)


def _dedupe_paths_by_tail(
    raw_paths: list[dict[str, Any]],
    named_paths: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped_raw: list[dict[str, Any]] = []
    deduped_named: list[dict[str, Any]] = []
    seen_tails: set[str] = set()

    for raw_path, named_path in zip(raw_paths, named_paths):
        raw_tail = norm_entity(tail_from_path_dict(raw_path))
        if raw_tail:
            if raw_tail in seen_tails:
                continue
            seen_tails.add(raw_tail)
        deduped_raw.append(raw_path)
        deduped_named.append(named_path)

    return deduped_raw, deduped_named


def _relation_sequence_from_path(path_dict: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(edge[1])
        for edge in path_dict.get("path", [])
        if isinstance(edge, (list, tuple)) and len(edge) >= 2
    )


# 含选择性约束(年份/序数/最高级/角色限定)的问题不做大答案集展开:
# 这类问题的答案是组内子集,展开整组会引入大量假阳性。
_EXPANSION_CONSTRAINT_WORDS = (
    "first", "last", "2008", "2009", "2010", "2011", "2012", "2013", "2014",
    "now", "current", "president", "capital", "main", "biggest", "largest",
    "before", "after", "died", "death", "won", "initially",
    "year", "date", "type", "call", "leader",
)


def _classify_batch(batch_size: int, accepted_count: int) -> str:
    if accepted_count == 0:
        return "all_wrong"
    if accepted_count == batch_size:
        return "all_correct"
    return "mixed"


def _prediction_mid_set(prediction: dict[str, float]) -> set[str]:
    return {norm_entity(mid) for mid in prediction if norm_entity(str(mid))}


@dataclass
class _CheckedBatchState:
    iterations: list[CheckedBatchIteration] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    accepted_indices: list[int] = field(default_factory=list)
    relation_expanded_indices: list[int] = field(default_factory=list)
    final_names: list[str] = field(default_factory=list)
    final_mids: list[str] = field(default_factory=list)
    seen_answer_keys: set[str] = field(default_factory=set)
    large_answer_expanded_mids: list[str] = field(default_factory=list)
    answer_tokens: int = 0
    check_tokens: int = 0
    answer_elapsed_ms: float = 0.0
    check_elapsed_ms: float = 0.0
    stop_reason: str = "path_exhausted"


class CheckedBatchAgent:
    """Retrieve top paths, answer in batches, and keep checked path tails."""

    def __init__(
        self,
        *,
        path_tool: PathRetrieveTool,
        answer_tool: AnswerWithPathsTool,
        check_tool: RejectedAnswerCheckTool,
        check_tool_after_first: RejectedAnswerCheckTool | None = None,
    ) -> None:
        self.path_tool = path_tool
        self.answer_tool = answer_tool
        self.check_tool = check_tool
        self.check_tool_after_first = check_tool_after_first
        self.entity_map = path_tool.entity_map
        self.reverse_entity_map = build_reverse_entity_map(self.entity_map)
        self._score_margin: float | None = None
        self._enable_relation_expansion: bool = True
        self._large_answer_expansion: bool = False
        self._kg_group_tails: dict[str, list[str]] | None = None
        self._expansion_min_answers: int = 8
        self._expansion_top_groups: int = 1

    def run(
        self,
        question: str,
        topic_mid: str,
        *,
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.2,
        batch_size: int = 20,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
        check_use_adapter: bool | None = None,
        check_max_new_tokens: int | None = None,
        sample_index: int | None = None,
        dedupe_tail_paths: bool = False,
        score_margin: float | None = None,
        enable_relation_expansion: bool = True,
        hop_filter: bool = False,
        large_answer_expansion: bool = False,
        kg_group_tails: dict[str, list[str]] | None = None,
        expansion_min_answers: int = 8,
        expansion_top_groups: int = 1,
        no_early_stop: bool = False,
        mixed_stop_ratio: float | None = 1.0 / 3.0,
        max_batches: int | None = None,
        stop_after_no_new_batches: int | None = None,
        no_all_wrong_after_answer_stop: bool = False,
        drop_topic_self: bool = True,
    ) -> CheckedBatchAgentResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if mixed_stop_ratio is not None and mixed_stop_ratio < 0:
            raise ValueError("mixed_stop_ratio must be non-negative or None")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive or None")
        if stop_after_no_new_batches is not None and stop_after_no_new_batches <= 0:
            raise ValueError("stop_after_no_new_batches must be positive or None")
        self._score_margin = score_margin
        self._enable_relation_expansion = enable_relation_expansion
        self._large_answer_expansion = large_answer_expansion
        self._kg_group_tails = kg_group_tails
        self._expansion_min_answers = expansion_min_answers
        self._expansion_top_groups = expansion_top_groups
        self._drop_topic_self = drop_topic_self

        retrieval = self.path_tool(
            question,
            topic_mid,
            alpha_final=alpha_final,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
            sample_index=sample_index,
        )
        # 在线 group_tails(server 实时算 + prediction 过滤)优先于文件 sidecar;
        # 旧 server 不返回该字段时回退到传入的 kg_group_tails
        if getattr(retrieval, "group_tails", None):
            self._kg_group_tails = retrieval.group_tails

        raw_paths = retrieval.raw_mmr_reason_paths
        named_paths = retrieval.named_mmr_reason_paths
        if dedupe_tail_paths:
            raw_paths, named_paths = _dedupe_paths_by_tail(raw_paths, named_paths)
        state = _CheckedBatchState()
        no_new_batches = 0

        for start in range(0, len(named_paths), batch_size):
            batch_named = named_paths[start : start + batch_size]
            batch_raw = raw_paths[start : start + batch_size]
            if not batch_named:
                break

            previous_answer_count = len(state.seen_answer_keys)
            batch_status, accepted_entity_count, batch_entity_count = self._run_checked_batch(
                question,
                start=start,
                batch_named=batch_named,
                batch_raw=batch_raw,
                raw_paths=raw_paths,
                named_paths=named_paths,
                raw_prediction_mids=_prediction_mid_set(retrieval.raw_prediction),
                state=state,
                use_adapter=use_adapter,
                max_new_tokens=max_new_tokens,
                check_use_adapter=check_use_adapter,
                check_max_new_tokens=check_max_new_tokens,
            )

            if len(state.seen_answer_keys) == previous_answer_count:
                no_new_batches += 1
            else:
                no_new_batches = 0

            more_paths = start + batch_size < len(named_paths)
            if (
                max_batches is not None
                and len(state.iterations) >= max_batches
                and more_paths
            ):
                state.stop_reason = "max_batches"
                break
            if (
                not no_early_stop
                and mixed_stop_ratio is not None
                and batch_status == "mixed"
                and accepted_entity_count <= batch_entity_count * mixed_stop_ratio
            ):
                state.stop_reason = "mixed"
                break
            if (
                stop_after_no_new_batches is not None
                and no_new_batches >= stop_after_no_new_batches
                and more_paths
            ):
                state.stop_reason = "no_new_answers"
                break
            if (
                not no_all_wrong_after_answer_stop
                and self.check_tool_after_first is not None
                and start > 0
                and batch_status == "all_wrong"
                and state.accepted_indices
            ):
                state.stop_reason = "all_wrong_after_answer"
                break

        if hop_filter:
            self._apply_hop_filter(state, raw_paths, hop=retrieval.hop)

        return self._build_result(
            question=question,
            topic_mid=topic_mid,
            retrieval=retrieval,
            raw_paths=raw_paths,
            named_paths=named_paths,
            state=state,
        )

    def _run_checked_batch(
        self,
        question: str,
        *,
        start: int,
        batch_named: list[dict[str, Any]],
        batch_raw: list[dict[str, Any]],
        raw_paths: list[dict[str, Any]],
        named_paths: list[dict[str, Any]],
        raw_prediction_mids: set[str],
        state: _CheckedBatchState,
        use_adapter: bool | None,
        max_new_tokens: int | None,
        check_use_adapter: bool | None,
        check_max_new_tokens: int | None,
    ) -> tuple[str, int, int]:
        answer = self.answer_tool(
            question,
            batch_named,
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
        )
        state.answer_tokens += answer.tokens_generated
        state.answer_elapsed_ms += answer.elapsed_ms

        check_tool = (
            self.check_tool_after_first
            if state.iterations and self.check_tool_after_first is not None
            else self.check_tool
        )
        check = check_tool(
            question,
            batch_named,
            cited_path_indices=answer.cited_path_indices,
            raw_paths=batch_raw,
            use_adapter=check_use_adapter,
            max_new_tokens=check_max_new_tokens,
        )
        state.check_tokens += check.total_tokens_generated
        state.check_elapsed_ms += check.total_elapsed_ms

        global_cited = [start + local_idx for local_idx in check.cited_path_indices]
        global_accepted = [
            start + local_idx for local_idx in check.accepted_path_indices
        ]
        self._record_checked_paths(
            global_cited=global_cited,
            global_accepted=global_accepted,
            raw_paths=raw_paths,
            named_paths=named_paths,
            answer_names=answer.answer_names,
            state=state,
        )

        batch_relation_expanded: list[int] = []
        if self._enable_relation_expansion:
            batch_relation_expanded = self._add_relation_expanded_answers(
                global_cited=global_cited,
                global_accepted=global_accepted,
                raw_paths=raw_paths,
                named_paths=named_paths,
                raw_prediction_mids=raw_prediction_mids,
                answer_names=answer.answer_names,
                state=state,
            )
        accepted_count = len(set(global_accepted) | set(batch_relation_expanded))
        accepted_entity_count = _tail_entity_count_for_indices(
            raw_paths,
            set(global_accepted) | set(batch_relation_expanded),
        )
        batch_entity_count = _tail_entity_count(batch_raw)
        batch_status = _classify_batch(
            batch_size=len(batch_named),
            accepted_count=accepted_count,
        )
        state.iterations.append(
            CheckedBatchIteration(
                batch_index=len(state.iterations) + 1,
                batch_start_rank=start + 1,
                batch_end_rank=start + len(batch_named),
                batch_size=len(batch_named),
                local_cited_path_indices=check.cited_path_indices,
                global_cited_path_indices=global_cited,
                accepted_path_indices=global_accepted,
                batch_status=batch_status,
                answer_prompt=answer.prompt,
                raw_llm_output=answer.raw_output,
                answer_names=answer.answer_names,
                format_ok=answer.format_ok,
                used_adapter=answer.used_adapter,
                answer_tokens_generated=answer.tokens_generated,
                answer_elapsed_ms=answer.elapsed_ms,
                path_check=check.to_dict(),
            )
        )
        return batch_status, accepted_entity_count, batch_entity_count

    def _record_checked_paths(
        self,
        *,
        global_cited: list[int],
        global_accepted: list[int],
        raw_paths: list[dict[str, Any]],
        named_paths: list[dict[str, Any]],
        answer_names: list[str],
        state: _CheckedBatchState,
    ) -> None:
        for global_idx in global_cited:
            if global_idx not in state.cited_indices:
                state.cited_indices.append(global_idx)
        state.accepted_indices.extend(global_accepted)

        for global_idx in global_accepted:
            path_offset = global_idx - 1
            raw_path = raw_paths[path_offset] if path_offset < len(raw_paths) else None
            named_answer, raw_answer = _answer_pair_from_paths(
                named_paths[path_offset],
                raw_path,
                answer_names=answer_names,
            )
            self._append_final_answer(
                named_answer=named_answer,
                raw_answer=raw_answer,
                state=state,
            )

    def _add_relation_expanded_answers(
        self,
        *,
        global_cited: list[int],
        global_accepted: list[int],
        raw_paths: list[dict[str, Any]],
        named_paths: list[dict[str, Any]],
        raw_prediction_mids: set[str],
        answer_names: list[str],
        state: _CheckedBatchState,
    ) -> list[int]:
        accepted_index_set = set(global_accepted)
        accepted_relation_sequences = {
            _relation_sequence_from_path(raw_paths[global_idx - 1])
            for global_idx in accepted_index_set
            if 0 < global_idx <= len(raw_paths)
        }
        accepted_relation_sequences.discard(())

        batch_relation_expanded: list[int] = []
        for global_idx in global_cited:
            if global_idx in accepted_index_set or not (
                0 < global_idx <= len(raw_paths)
            ):
                continue
            relation_sequence = _relation_sequence_from_path(raw_paths[global_idx - 1])
            if relation_sequence not in accepted_relation_sequences:
                continue

            named_answer, raw_answer = _answer_pair_from_paths(
                named_paths[global_idx - 1],
                raw_paths[global_idx - 1],
                answer_names=answer_names,
            )
            if norm_entity(raw_answer) not in raw_prediction_mids:
                continue

            batch_relation_expanded.append(global_idx)
            state.relation_expanded_indices.append(global_idx)

            path_offset = global_idx - 1
            self._append_final_answer(
                named_answer=named_answer,
                raw_answer=raw_answer,
                state=state,
            )
        return batch_relation_expanded

    def _append_final_answer(
        self,
        *,
        named_answer: str,
        raw_answer: str,
        state: _CheckedBatchState,
    ) -> None:
        dedupe_key = norm_entity(raw_answer or named_answer)
        if not dedupe_key or dedupe_key in state.seen_answer_keys:
            return
        state.seen_answer_keys.add(dedupe_key)
        state.final_names.append(named_answer)
        state.final_mids.append(raw_answer or named_answer)

    @staticmethod
    def _support_paths(
        state: _CheckedBatchState, raw_paths: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """All accepted + relation-expanded paths backing the final answers."""
        return [
            raw_paths[idx - 1]
            for idx in dict.fromkeys(
                state.accepted_indices + state.relation_expanded_indices
            )
            if 0 < idx <= len(raw_paths)
        ]

    @staticmethod
    def _keep_final_answers(
        state: _CheckedBatchState, kept_pairs: list[tuple[str, str]]
    ) -> None:
        """Replace the final answers. An empty keep-list means the filter
        signal contradicts every answer — distrust the filter and keep all."""
        if not kept_pairs:
            return
        state.final_names = [name for name, _ in kept_pairs]
        state.final_mids = [mid for _, mid in kept_pairs]

    def _apply_hop_filter(
        self,
        state: _CheckedBatchState,
        raw_paths: list[dict[str, Any]],
        *,
        hop: int,
    ) -> None:
        """Drop final answers supported only by relation chains whose length
        differs from the retrieval-predicted hop count."""
        if not state.final_mids:
            return
        mid_seq_lens: dict[str, set[int]] = {}
        for path_dict in self._support_paths(state, raw_paths):
            key = norm_entity(tail_from_path_dict(path_dict))
            seq = _relation_sequence_from_path(path_dict)
            if key and seq:
                mid_seq_lens.setdefault(key, set()).add(len(seq))
        self._keep_final_answers(
            state,
            [
                (name, mid)
                for name, mid in zip(state.final_names, state.final_mids)
                if hop in mid_seq_lens.get(norm_entity(mid), {hop})
            ],
        )

    def _apply_score_margin(
        self,
        state: _CheckedBatchState,
        raw_paths: list[dict[str, Any]],
    ) -> None:
        """Drop final answers whose best supporting-path log_score trails the
        top answer by more than score_margin (relative post-filter)."""
        if self._score_margin is None or not state.final_mids:
            return
        answer_score: dict[str, float] = {}
        for path_dict in self._support_paths(state, raw_paths):
            mid = norm_entity(tail_from_path_dict(path_dict))
            if not mid:
                continue
            score = float(path_dict.get("log_score", float("-inf")))
            answer_score[mid] = max(answer_score.get(mid, float("-inf")), score)

        scores = [
            answer_score.get(norm_entity(mid), float("-inf"))
            for mid in state.final_mids
        ]
        top = max(scores)
        if top == float("-inf"):
            return
        self._keep_final_answers(
            state,
            [
                (name, mid)
                for name, mid, score in zip(state.final_names, state.final_mids, scores)
                if score >= top - self._score_margin
            ],
        )

    def _apply_large_answer_expansion(
        self,
        state: _CheckedBatchState,
        raw_paths: list[dict[str, Any]],
        *,
        question: str,
        topic_mid: str,
        raw_prediction: dict[str, float],
    ) -> None:
        """For enumeration-type questions, expand final answers to all KG tails
        of the winning relation group, gated by the TransferNet prediction.
        Must run after the score margin filter: expanded answers have no beam
        path scores and would otherwise be dropped as -inf."""
        if self._kg_group_tails is None or not state.final_mids:
            return
        if len(state.final_mids) < self._expansion_min_answers:
            return
        question_lower = question.lower()
        if any(word in question_lower for word in _EXPANSION_CONSTRAINT_WORDS):
            return

        final_keys = {norm_entity(mid) for mid in state.final_mids}
        seq_counts: Counter[tuple[str, ...]] = Counter()
        for path_dict in self._support_paths(state, raw_paths):
            seq = _relation_sequence_from_path(path_dict)
            if seq and norm_entity(tail_from_path_dict(path_dict)) in final_keys:
                seq_counts[seq] += 1
        if not seq_counts:
            return
        prediction_keys = _prediction_mid_set(raw_prediction)
        for winning_seq, _ in seq_counts.most_common(self._expansion_top_groups):
            kg_tails = self._kg_group_tails.get("|".join((topic_mid, *winning_seq)), [])
            for mid in kg_tails:
                mid = str(mid)
                key = norm_entity(mid)
                if not key or key not in prediction_keys or key in state.seen_answer_keys:
                    continue
                state.seen_answer_keys.add(key)
                state.final_names.append(self.entity_map.get(mid, mid))
                state.final_mids.append(mid)
                state.large_answer_expanded_mids.append(mid)

    @staticmethod
    def _drop_topic_self_answers(state: _CheckedBatchState, topic_mid: str) -> None:
        """剔除最终答案中 == topic 的自指实体。答案=被问实体本身逻辑上不可能成立
        (测试集 0 gold 反例)。检索层 drop_loopback_paths 只挡"尾==topic 的路径",
        而 topic 仍可经 _answer_pair_from_paths 的 head 匹配 / large_answer_expansion
        进入答案——这里在末端统一收口,覆盖所有入口。剔空则答案为空(不保底留 topic)。"""
        topic_key = norm_entity(topic_mid)
        if not topic_key or not state.final_mids:
            return
        kept = [
            (name, mid)
            for name, mid in zip(state.final_names, state.final_mids)
            if norm_entity(mid) != topic_key
        ]
        state.final_names = [name for name, _ in kept]
        state.final_mids = [mid for _, mid in kept]
        state.large_answer_expanded_mids = [
            mid
            for mid in state.large_answer_expanded_mids
            if norm_entity(mid) != topic_key
        ]

    def _build_result(
        self,
        *,
        question: str,
        topic_mid: str,
        retrieval: PathRetrieveToolResult,
        raw_paths: list[dict[str, Any]],
        named_paths: list[dict[str, Any]],
        state: _CheckedBatchState,
    ) -> CheckedBatchAgentResult:
        self._apply_score_margin(state, raw_paths)
        if self._large_answer_expansion:
            self._apply_large_answer_expansion(
                state,
                raw_paths,
                question=question,
                topic_mid=topic_mid,
                raw_prediction=retrieval.raw_prediction,
            )
        if getattr(self, "_drop_topic_self", True):
            self._drop_topic_self_answers(state, topic_mid)
        expanded_mids, disambiguated_mids = expand_pred_answers_with_path_constraint(
            pred_answers=state.final_names,
            rev_entity_map=self.reverse_entity_map,
            path_mid_entities={norm_entity(mid) for mid in state.final_mids},
        )
        if state.final_mids:
            disambiguated_mids = state.final_mids

        return CheckedBatchAgentResult(
            question=question,
            topic_mid=topic_mid,
            hop=retrieval.hop,
            raw_topics=retrieval.raw_topics,
            named_topics=retrieval.named_topics,
            raw_mmr_reason_paths=raw_paths,
            named_mmr_reason_paths=named_paths,
            raw_prediction=retrieval.raw_prediction,
            named_prediction=retrieval.named_prediction,
            iterations=state.iterations,
            final_accepted_path_indices=state.accepted_indices,
            cited_path_indices=state.cited_indices,
            pred_answer_names=state.final_names,
            pred_answer_expanded_mids=expanded_mids,
            pred_answer_disambiguated_mids=disambiguated_mids,
            relation_expanded_path_indices=state.relation_expanded_indices,
            large_answer_expanded_mids=state.large_answer_expanded_mids,
            group_tails=self._kg_group_tails or {},
            batches_used=len(state.iterations),
            checked_paths_count=sum(
                len(item.local_cited_path_indices) for item in state.iterations
            ),
            accepted_paths_count=len(state.accepted_indices),
            final_answer_count=len(state.final_names),
            stop_reason=state.stop_reason,
            format_ok=(
                all(item.format_ok for item in state.iterations)
                if state.iterations
                else False
            ),
            used_adapter=any(item.used_adapter for item in state.iterations),
            tokens_generated=state.answer_tokens + state.check_tokens,
            answer_tokens_generated=state.answer_tokens,
            check_tokens_generated=state.check_tokens,
            retrieval_elapsed_ms=retrieval.elapsed_ms,
            llm_elapsed_ms=state.answer_elapsed_ms,
            check_elapsed_ms=state.check_elapsed_ms,
        )


# legacy 别名(旧包时代的类名,保留兼容既有引用与录制)
CheckedBatchWebQAgent = CheckedBatchAgent
CheckedBatchWebQAgentResult = CheckedBatchAgentResult
