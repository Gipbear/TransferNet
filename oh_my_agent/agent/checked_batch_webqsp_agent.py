"""Batch-and-check WebQSP QA agent using cached path retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from oh_my_agent.common import (
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
)
from oh_my_agent.common.metrics import norm_entity
from oh_my_agent.tools import AnswerWithPathsTool, CitedPathCheckTool, PathRetrieveTool


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
        )

        iterations: list[CheckedBatchIteration] = []
        all_cited_indices: list[int] = []
        all_accepted_indices: list[int] = []
        relation_expanded_indices: list[int] = []
        final_names: list[str] = []
        final_mids: list[str] = []
        seen_answer_keys: set[str] = set()
        answer_tokens = 0
        check_tokens = 0
        answer_elapsed = 0.0
        check_elapsed = 0.0
        stop_reason = "path_exhausted"

        raw_paths = retrieval.raw_mmr_reason_paths
        named_paths = retrieval.named_mmr_reason_paths

        for start in range(0, len(named_paths), batch_size):
            batch_named = named_paths[start : start + batch_size]
            batch_raw = raw_paths[start : start + batch_size]
            if not batch_named:
                break

            answer = self.answer_tool(
                question,
                batch_named,
                use_adapter=use_adapter,
                max_new_tokens=max_new_tokens,
            )
            answer_tokens += answer.tokens_generated
            answer_elapsed += answer.elapsed_ms

            check = self.check_tool(
                question,
                batch_named,
                cited_path_indices=answer.cited_path_indices,
                raw_paths=batch_raw,
                use_adapter=check_use_adapter,
                max_new_tokens=check_max_new_tokens,
            )
            check_tokens += check.total_tokens_generated
            check_elapsed += check.total_elapsed_ms

            global_cited = [start + local_idx for local_idx in check.cited_path_indices]
            global_accepted = [start + local_idx for local_idx in check.accepted_path_indices]
            for global_idx in global_cited:
                if global_idx not in all_cited_indices:
                    all_cited_indices.append(global_idx)
            all_accepted_indices.extend(global_accepted)

            for global_idx in global_accepted:
                path_offset = global_idx - 1
                named_tail = _tail_from_path(named_paths[path_offset])
                raw_tail = _tail_from_path(raw_paths[path_offset]) if path_offset < len(raw_paths) else ""
                dedupe_key = norm_entity(raw_tail or named_tail)
                if not dedupe_key or dedupe_key in seen_answer_keys:
                    continue
                seen_answer_keys.add(dedupe_key)
                final_names.append(named_tail)
                final_mids.append(raw_tail or named_tail)

            accepted_index_set = set(global_accepted)
            accepted_relation_sequences = {
                _relation_sequence_from_path(raw_paths[global_idx - 1])
                for global_idx in accepted_index_set
                if 0 < global_idx <= len(raw_paths)
            }
            accepted_relation_sequences.discard(())
            batch_relation_expanded: list[int] = []
            for global_idx in global_cited:
                if global_idx in accepted_index_set or not (0 < global_idx <= len(raw_paths)):
                    continue
                relation_sequence = _relation_sequence_from_path(raw_paths[global_idx - 1])
                if relation_sequence not in accepted_relation_sequences:
                    continue
                batch_relation_expanded.append(global_idx)
                relation_expanded_indices.append(global_idx)

                path_offset = global_idx - 1
                named_tail = (
                    _tail_from_path(named_paths[path_offset])
                    if path_offset < len(named_paths)
                    else ""
                )
                raw_tail = _tail_from_path(raw_paths[path_offset])
                dedupe_key = norm_entity(raw_tail or named_tail)
                if not dedupe_key or dedupe_key in seen_answer_keys:
                    continue
                seen_answer_keys.add(dedupe_key)
                final_names.append(named_tail)
                final_mids.append(raw_tail or named_tail)

            batch_status = _classify_batch(
                batch_size=len(batch_named),
                accepted_count=len(accepted_index_set | set(batch_relation_expanded)),
            )
            iterations.append(
                CheckedBatchIteration(
                    batch_index=len(iterations) + 1,
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

            if batch_status == "mixed":
                stop_reason = "mixed"
                break

        expanded_mids, disambiguated_mids = expand_pred_answers_with_path_constraint(
            pred_answers=final_names,
            rev_entity_map=self.reverse_entity_map,
            path_mid_entities={norm_entity(mid) for mid in final_mids},
        )
        if final_mids:
            disambiguated_mids = final_mids

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
            iterations=iterations,
            final_accepted_path_indices=all_accepted_indices,
            cited_path_indices=all_cited_indices,
            pred_answer_names=final_names,
            pred_answer_expanded_mids=expanded_mids,
            pred_answer_disambiguated_mids=disambiguated_mids,
            relation_expanded_path_indices=relation_expanded_indices,
            batches_used=len(iterations),
            checked_paths_count=sum(
                len(item.local_cited_path_indices) for item in iterations
            ),
            accepted_paths_count=len(all_accepted_indices),
            final_answer_count=len(final_names),
            stop_reason=stop_reason,
            format_ok=all(item.format_ok for item in iterations) if iterations else False,
            used_adapter=any(item.used_adapter for item in iterations),
            tokens_generated=answer_tokens + check_tokens,
            answer_tokens_generated=answer_tokens,
            check_tokens_generated=check_tokens,
            retrieval_elapsed_ms=retrieval.elapsed_ms,
            llm_elapsed_ms=answer_elapsed,
            check_elapsed_ms=check_elapsed,
        )
