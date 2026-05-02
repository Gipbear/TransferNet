"""LLM cited-path checker for path-by-path KGQA validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from oh_my_agent.common.prompting import _format_schema_chain
from oh_my_agent.llm_server import LLMClient


CITED_PATH_CHECK_SYSTEM = (
    "You are a KGQA evaluator. You will be given a question and a single reasoning path "
    "from a knowledge graph.\n"
    "Determine if the relation in this path matches the intent of the question.\n\n"
    "Use a loose standard: KG relation names are often broader or more general than the "
    "exact phrasing of the question. If the relation broadly or approximately covers the "
    "concept the question is asking about, judge Y. Focus on whether the tail entity would "
    "be a reasonable answer to the question, not whether the relation name is an exact "
    "synonym of the question's wording. Only judge N when the relation clearly targets a "
    "different concept entirely.\n\n"
    "Answer ONLY 'Y' if the path correctly answers the question, or 'N' if it does not.\n\n"
    "Examples:\n"
    "Q: What is Obama's father's name?\n"
    "Path: Barack Obama - [people.person.parents] -> Barack Obama Sr.\n"
    "Output: Y\n\n"
    "Q: Who directed Inception?\n"
    "Path: Inception - [film.film.starring] -> Leonardo DiCaprio\n"
    "Output: N"
)


@dataclass(frozen=True)
class CitedPathEvaluation:
    """LLM judgment for one original cited path."""

    path_index: int
    path_text: str
    raw_output: str
    is_correct: bool
    tail_entity: str
    tail_mid: str | None = None
    tokens_generated: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitedPathCheckResult:
    """Structured result from checking cited paths one by one."""

    question: str
    cited_path_indices: list[int]
    path_evaluations: list[CitedPathEvaluation] = field(default_factory=list)
    accepted_path_indices: list[int] = field(default_factory=list)
    predicted_answer_names: list[str] = field(default_factory=list)
    predicted_mids: list[str] = field(default_factory=list)
    total_tokens_generated: int = 0
    total_elapsed_ms: float = 0.0

    @property
    def any_accepted_path(self) -> bool:
        return bool(self.accepted_path_indices)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["any_accepted_path"] = self.any_accepted_path
        return data


def build_cited_path_prompt(question: str, path_edges: list[Any]) -> str:
    path_text = _format_schema_chain(path_edges)
    return f"Q: {question}\nPath: {path_text}\nOutput:"


def parse_cited_path_output(raw_output: str) -> bool:
    """Return True only for an explicit leading Y answer."""
    return (raw_output or "").strip().upper().startswith("Y")


def _tail_from_path(path_edges: list[Any]) -> str:
    if not path_edges:
        return ""
    return str(path_edges[-1][-1])


def _unique_append(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


class CitedPathCheckTool:
    """Check each cited path independently and collect accepted tail entities."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        base_url: str = "http://localhost:8788",
        default_use_adapter: bool = False,
        default_max_new_tokens: int = 2,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client or LLMClient(base_url)
        self.default_use_adapter = default_use_adapter
        self.default_max_new_tokens = default_max_new_tokens
        self.system_prompt = CITED_PATH_CHECK_SYSTEM if system_prompt is None else system_prompt

    def __call__(
        self,
        question: str,
        named_paths: list[dict[str, Any]],
        *,
        cited_path_indices: list[int] | tuple[int, ...],
        raw_paths: list[dict[str, Any]] | None = None,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> CitedPathCheckResult:
        use_adapter = self.default_use_adapter if use_adapter is None else use_adapter
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        raw_paths = raw_paths or []

        filtered_indices = sorted(
            idx for idx in cited_path_indices
            if 0 < idx <= len(named_paths)
        )
        evaluations: list[CitedPathEvaluation] = []
        accepted_indices: list[int] = []
        answer_names: list[str] = []
        predicted_mids: list[str] = []
        total_tokens = 0
        total_elapsed = 0.0

        for orig_idx in filtered_indices:
            path_edges = named_paths[orig_idx - 1].get("path", [])
            if not path_edges:
                continue

            prompt = build_cited_path_prompt(question, path_edges)
            response = self.client.generate(
                prompt,
                use_adapter=use_adapter,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                system_prompt=self.system_prompt,
            )
            is_correct = parse_cited_path_output(response.text)
            tail_entity = _tail_from_path(path_edges)
            tail_mid = None
            raw_idx = orig_idx - 1
            if 0 <= raw_idx < len(raw_paths):
                raw_edges = raw_paths[raw_idx].get("path", [])
                if raw_edges:
                    tail_mid = _tail_from_path(raw_edges)

            total_tokens += response.tokens_generated
            total_elapsed += response.elapsed_ms
            evaluations.append(
                CitedPathEvaluation(
                    path_index=orig_idx,
                    path_text=_format_schema_chain(path_edges),
                    raw_output=response.text,
                    is_correct=is_correct,
                    tail_entity=tail_entity,
                    tail_mid=tail_mid,
                    tokens_generated=response.tokens_generated,
                    elapsed_ms=response.elapsed_ms,
                )
            )

            if is_correct:
                accepted_indices.append(orig_idx)
                _unique_append(answer_names, tail_entity)
                if tail_mid is not None:
                    _unique_append(predicted_mids, tail_mid)

        return CitedPathCheckResult(
            question=question,
            cited_path_indices=filtered_indices,
            path_evaluations=evaluations,
            accepted_path_indices=accepted_indices,
            predicted_answer_names=answer_names,
            predicted_mids=predicted_mids,
            total_tokens_generated=total_tokens,
            total_elapsed_ms=total_elapsed,
        )

    def from_record(
        self,
        record: dict[str, Any],
        *,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> CitedPathCheckResult:
        return self(
            record.get("question", ""),
            record.get("named_mmr_reason_paths", []),
            cited_path_indices=record.get("cited_path_indices", []),
            raw_paths=record.get("raw_mmr_reason_paths", []),
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
        )
