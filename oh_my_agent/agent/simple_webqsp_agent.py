"""Fixed two-step WebQSP QA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from oh_my_agent.common import (
    build_reverse_entity_map,
    expand_pred_answers_with_path_constraint,
    get_all_path_entities,
)
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool


@dataclass(frozen=True)
class SimpleWebQAgentResult:
    """End-to-end result of the simple two-step QA flow."""

    question: str
    topic_mid: str
    hop: int
    raw_topics: list[str]
    named_topics: list[str]
    raw_mmr_reason_paths: list[dict[str, Any]]
    named_mmr_reason_paths: list[dict[str, Any]]
    raw_prediction: dict[str, float]
    named_prediction: dict[str, float]
    llm_prompt: str
    raw_llm_output: str
    pred_answer_names: list[str]
    pred_answer_expanded_mids: list[str]
    pred_answer_disambiguated_mids: list[str]
    cited_path_indices: list[int]
    format_ok: bool
    used_adapter: bool
    tokens_generated: int
    retrieval_elapsed_ms: float
    llm_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimpleWebQAgent:
    """Very simple fixed-flow agent: retrieve paths, then answer from them."""

    def __init__(
        self,
        *,
        path_tool: PathRetrievalTool,
        answer_tool: AnswerWithPathsTool,
    ) -> None:
        self.path_tool = path_tool
        self.answer_tool = answer_tool
        self.entity_map = path_tool.entity_map
        self.reverse_entity_map = build_reverse_entity_map(self.entity_map)

    def run(
        self,
        question: str,
        topic_mid: str,
        *,
        hop: int | None = None,
        beam_size: int = 20,
        lambda_val: float = 0.2,
        prediction_threshold: float = 0.9,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> SimpleWebQAgentResult:
        retrieval = self.path_tool(
            question,
            topic_mid,
            hop=hop,
            beam_size=beam_size,
            lambda_val=lambda_val,
            prediction_threshold=prediction_threshold,
        )
        answer = self.answer_tool(
            question,
            retrieval.named_mmr_reason_paths,
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
        )
        path_mid_entities = get_all_path_entities(retrieval.raw_mmr_reason_paths)
        expanded_mids, disambiguated_mids = expand_pred_answers_with_path_constraint(
            pred_answers=answer.answer_names,
            rev_entity_map=self.reverse_entity_map,
            path_mid_entities=path_mid_entities,
        )
        return SimpleWebQAgentResult(
            question=question,
            topic_mid=topic_mid,
            hop=retrieval.hop,
            raw_topics=retrieval.raw_topics,
            named_topics=retrieval.named_topics,
            raw_mmr_reason_paths=retrieval.raw_mmr_reason_paths,
            named_mmr_reason_paths=retrieval.named_mmr_reason_paths,
            raw_prediction=retrieval.raw_prediction,
            named_prediction=retrieval.named_prediction,
            llm_prompt=answer.prompt,
            raw_llm_output=answer.raw_output,
            pred_answer_names=answer.answer_names,
            pred_answer_expanded_mids=expanded_mids,
            pred_answer_disambiguated_mids=disambiguated_mids,
            cited_path_indices=answer.cited_path_indices,
            format_ok=answer.format_ok,
            used_adapter=answer.used_adapter,
            tokens_generated=answer.tokens_generated,
            retrieval_elapsed_ms=retrieval.elapsed_ms,
            llm_elapsed_ms=answer.elapsed_ms,
        )
