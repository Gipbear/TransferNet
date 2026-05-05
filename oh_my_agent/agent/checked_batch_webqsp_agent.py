"""Batch-and-check WebQSP QA agent using cached path retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from oh_my_agent.common import (
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
)
from oh_my_agent.common.metrics import norm_entity
from oh_my_agent.tools import (
    AnswerWithPathsTool,
    CitedPathCheckTool,
    PathRetrieveTool,
    PathRetrieveToolResult,
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
class CheckedBatchWebQAgentResult:
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


def _tail_from_path(path_dict: dict[str, Any]) -> str:
    edges = path_dict.get("path", [])
    if not edges:
        return ""
    return str(edges[-1][-1])


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
    named_answer = _tail_from_path(named_path_dict)
    raw_answer = _tail_from_path(raw_path_dict)

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
    return len({norm_entity(_tail_from_path(path)) for path in paths if _tail_from_path(path)})


def _tail_entity_count_for_indices(raw_paths: list[dict[str, Any]], indices: set[int]) -> int:
    tails: set[str] = set()
    for index in indices:
        path_offset = index - 1
        if 0 <= path_offset < len(raw_paths):
            tail = _tail_from_path(raw_paths[path_offset])
            if tail:
                tails.add(norm_entity(tail))
    return len(tails)


def _relation_sequence_from_path(path_dict: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(edge[1])
        for edge in path_dict.get("path", [])
        if isinstance(edge, (list, tuple)) and len(edge) >= 2
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
    answer_tokens: int = 0
    check_tokens: int = 0
    answer_elapsed_ms: float = 0.0
    check_elapsed_ms: float = 0.0
    stop_reason: str = "path_exhausted"


class CheckedBatchWebQAgent:
    """Retrieve top paths, answer in batches, and keep checked path tails."""

    def __init__(
        self,
        *,
        path_tool: PathRetrieveTool,
        answer_tool: AnswerWithPathsTool,
        check_tool: CitedPathCheckTool,
    ) -> None:
        self.path_tool = path_tool
        self.answer_tool = answer_tool
        self.check_tool = check_tool
        self.entity_map = path_tool.entity_map
        self.reverse_entity_map = build_reverse_entity_map(self.entity_map)

    def run(
        self,
        question: str,
        topic_mid: str,
        *,
        method: str = "tail_blend",
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.5,
        batch_size: int = 20,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
        check_use_adapter: bool | None = None,
        check_max_new_tokens: int | None = None,
        sample_index: int | None = None,
    ) -> CheckedBatchWebQAgentResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        retrieval = self.path_tool(
            question,
            topic_mid,
            method=method,
            alpha_final=alpha_final,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
            sample_index=sample_index,
        )

        raw_paths = retrieval.raw_mmr_reason_paths
        named_paths = retrieval.named_mmr_reason_paths
        state = _CheckedBatchState()

        for start in range(0, len(named_paths), batch_size):
            batch_named = named_paths[start : start + batch_size]
            batch_raw = raw_paths[start : start + batch_size]
            if not batch_named:
                break

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

            if batch_status == "mixed" and accepted_entity_count <= batch_entity_count / 3:
                state.stop_reason = "mixed"
                break

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

        check = self.check_tool(
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

    def _build_result(
        self,
        *,
        question: str,
        topic_mid: str,
        retrieval: PathRetrieveToolResult,
        raw_paths: list[dict[str, Any]],
        named_paths: list[dict[str, Any]],
        state: _CheckedBatchState,
    ) -> CheckedBatchWebQAgentResult:
        expanded_mids, disambiguated_mids = expand_pred_answers_with_path_constraint(
            pred_answers=state.final_names,
            rev_entity_map=self.reverse_entity_map,
            path_mid_entities={norm_entity(mid) for mid in state.final_mids},
        )
        if state.final_mids:
            disambiguated_mids = state.final_mids

        return CheckedBatchWebQAgentResult(
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
