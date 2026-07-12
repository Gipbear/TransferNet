"""LLM reject-list checker for cited-answer KGQA validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kgqa.agent.common.paths import tail_from_edges
from kgqa.agent.common.prompting import format_chain
from kgqa.llm_server.client import LLMClient


REJECTED_ANSWER_CHECK_SYSTEM = (
    "You are a conservative KGQA candidate-answer cleaner. You will be given a question, "
    "cited knowledge-graph paths, and numbered candidate answers. Each candidate lists "
    "the indices of the cited paths that support it; judge a candidate against its own "
    "supporting paths.\n"
    "Return only candidate numbers that are CLEARLY IMPOSSIBLE or CLEARLY WRONG answers "
    "to the question. Do not remove a candidate just because the evidence is indirect, "
    "the candidate is one of many answers, or you are uncertain.\n"
    "This is a multi-answer task. For broad list questions such as movies, books, "
    "songs, battles, countries, senators, or places, many or all candidates may be "
    "valid. Do not remove a candidate merely because it is less famous, one of many, "
    "or not the first answer.\n"
    "Be very recall-preserving: keeping a wrong candidate is less harmful than removing "
    "a correct candidate. Keep borderline, plausible, weakly supported, or type-matching "
    "candidates. If unsure, do not list the candidate number.\n"
    "Output exactly one line. Use ONLY comma-separated candidate numbers like 1,2,5, "
    "or output NONE if no candidate should be removed. Output candidate numbers from "
    "the Candidate Answers list, NOT path numbers. No words, no brackets, no "
    "explanation."
)
STRICT_REJECTED_ANSWER_CHECK_SYSTEM = (
    "You are a strict KGQA candidate-answer cleaner. You will be given a question, "
    "cited knowledge-graph paths, and numbered candidate answers. Each candidate lists "
    "the indices of the cited paths that support it; judge a candidate against its own "
    "supporting paths.\n"
    "Return candidate numbers that should be removed because they do not answer the "
    "question, have the wrong answer type, target a different relation or constraint, "
    "or are only loosely related to the question.\n"
    "This is stricter than a recall-preserving pass: remove a candidate unless the "
    "cited paths make it a clear answer to the exact question. "
    "A valid KG reasoning path can be indirect or two-hop, such as place -> region -> "
    "timezone, as long as the tail entity still satisfies the question. "
    "For multi-answer list questions, keep candidates that are clear members of the "
    "requested list, but remove merely associated entities, locations, people, dates, "
    "or works that do not satisfy the exact requested category.\n"
    "Output exactly one line. Use ONLY comma-separated candidate numbers like 1,2,5, "
    "or output NONE if no candidate should be removed. Output candidate numbers from "
    "the Candidate Answers list, NOT path numbers. No words, no brackets, no "
    "explanation."
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
    check_mode: str = "reject-answer-list"
    raw_output: str = ""
    candidate_answers: list[dict[str, Any]] = field(default_factory=list)
    rejected_answer_indices: list[int] = field(default_factory=list)
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


def build_rejected_answer_prompt(
    question: str,
    cited_paths: list[dict[str, Any]],
    candidate_answers: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> str:
    path_lines = [
        "[{}] {}".format(item["path_index"], item["path_text"])
        for item in cited_paths
    ]
    answer_lines = []
    for item in candidate_answers:
        line = "[{}] {}".format(item["index"], item["name"] or item["mid"])
        path_indices = item.get("path_indices") or []
        if path_indices:
            line += " (supported by paths: {})".format(
                ", ".join(str(i) for i in path_indices)
            )
        answer_lines.append(line)
    loose_task = [
        "List only candidate answers that are clearly impossible or clearly wrong and should be removed.",
        "Use a high threshold for removal. If you are not certain a candidate is wrong, keep it.",
        "Keep candidates that are correct, plausible, weakly supported, or have the right answer type.",
        "For list questions, keep all plausible list members; do not remove valid answers just because there are many of them.",
        "When in doubt, output NONE or omit the uncertain candidate number.",
    ]
    strict_task = [
        "List candidate answers that should be removed from the answer set.",
        "Keep a candidate when the cited KG paths make it a clear answer to the exact question.",
        "Do not reject a candidate merely because the path is indirect or two-hop; KG evidence often uses intermediate entities.",
        "Remove candidates that are only associated with the topic, have the wrong answer type, satisfy a different relation, or miss a question constraint.",
        "For list questions, keep clear list members, but remove entities that are merely related to list members or to the topic.",
        "If uncertain but the candidate does not match the requested answer type or relation, list its number.",
    ]
    task_lines = strict_task if strict else loose_task
    output_lines = [
        "One line only: comma-separated candidate numbers to remove, e.g. 1,2,5",
        "Use candidate numbers from the Candidate Answers list, NOT path numbers.",
        "If no candidate should be removed, output exactly: NONE",
    ]
    return "\n".join(
        [
            f"Question: {question}",
            "",
            "Cited Paths:",
            "\n".join(path_lines) if path_lines else "(none)",
            "",
            "Candidate Answers:",
            "\n".join(answer_lines) if answer_lines else "(none)",
            "",
            "Task:",
            *task_lines,
            "",
            "Output format:",
            *output_lines,
        ]
    )


def parse_rejected_answer_indices(raw_output: str, max_index: int) -> list[int]:
    text = (raw_output or "").strip()
    if not text or text.lower() in {"none", "no", "no answer", "[]"}:
        return []

    indices: list[int] = []
    current = ""
    for ch in text:
        if ch.isdigit():
            current += ch
        elif current:
            index = int(current)
            if 1 <= index <= max_index and index not in indices:
                indices.append(index)
            current = ""
    if current:
        index = int(current)
        if 1 <= index <= max_index and index not in indices:
            indices.append(index)
    return indices


def _unique_append(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _candidate_answers_from_cited_paths(
    named_paths: list[dict[str, Any]],
    raw_paths: list[dict[str, Any]],
    cited_path_indices: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    cited_paths: list[dict[str, Any]] = []
    key_to_candidate: dict[str, dict[str, Any]] = {}

    for orig_idx in cited_path_indices:
        path_offset = orig_idx - 1
        path_edges = named_paths[path_offset].get("path", [])
        if not path_edges:
            continue
        raw_edges = raw_paths[path_offset].get("path", []) if path_offset < len(raw_paths) else []
        name = tail_from_edges(path_edges)
        mid = tail_from_edges(raw_edges) if raw_edges else ""
        key = f"mid:{mid.lower().strip()}" if mid else f"name:{name.lower().strip()}"
        if key not in key_to_candidate:
            candidate = {
                "index": len(candidates) + 1,
                "name": name,
                "mid": mid,
                "path_indices": [],
            }
            candidates.append(candidate)
            key_to_candidate[key] = candidate
        key_to_candidate[key]["path_indices"].append(orig_idx)
        cited_paths.append(
            {
                "path_index": orig_idx,
                "path_text": format_chain(path_edges),
                "candidate_index": key_to_candidate[key]["index"],
            }
        )

    return candidates, cited_paths


class RejectedAnswerCheckTool:
    """Remove clearly wrong cited-answer candidates and keep the complement."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        base_url: str = "http://localhost:8788",
        default_use_adapter: bool = False,
        default_max_new_tokens: int = 48,
        system_prompt: str | None = None,
        reject_policy: str = "loose",
        constrained_decoding: bool = False,
    ) -> None:
        if reject_policy not in {"loose", "strict"}:
            raise ValueError("reject_policy must be 'loose' or 'strict'")
        self.client = client or LLMClient(base_url)
        self.default_use_adapter = default_use_adapter
        self.default_max_new_tokens = default_max_new_tokens
        self.reject_policy = reject_policy
        self.constrained_decoding = constrained_decoding
        if reject_policy == "strict":
            default_system_prompt = STRICT_REJECTED_ANSWER_CHECK_SYSTEM
        else:
            default_system_prompt = REJECTED_ANSWER_CHECK_SYSTEM
        self.system_prompt = (
            default_system_prompt if system_prompt is None else system_prompt
        )

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
        candidates, cited_paths = _candidate_answers_from_cited_paths(
            named_paths,
            raw_paths,
            filtered_indices,
        )
        if not candidates:
            return CitedPathCheckResult(
                question=question,
                cited_path_indices=filtered_indices,
                check_mode=f"reject-answer-list:{self.reject_policy}",
            )

        prompt = build_rejected_answer_prompt(
            question,
            cited_paths,
            candidates,
            strict=self.reject_policy == "strict",
        )
        generate_kwargs: dict[str, Any] = {}
        if self.constrained_decoding:
            generate_kwargs["max_option_index"] = len(candidates)
        response = self.client.generate(
            prompt,
            use_adapter=use_adapter,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            system_prompt=self.system_prompt,
            **generate_kwargs,
        )
        rejected_set = set(parse_rejected_answer_indices(response.text, len(candidates)))
        accepted_candidates = [
            candidate
            for candidate in candidates
            if candidate["index"] not in rejected_set
        ]
        accepted_path_set = {
            path_index
            for candidate in accepted_candidates
            for path_index in candidate["path_indices"]
        }
        accepted_indices = [
            path_index for path_index in filtered_indices if path_index in accepted_path_set
        ]

        answer_names: list[str] = []
        predicted_mids: list[str] = []
        for candidate in accepted_candidates:
            _unique_append(answer_names, str(candidate.get("name", "")))
            _unique_append(predicted_mids, str(candidate.get("mid", "")))

        evaluations: list[CitedPathEvaluation] = []
        for orig_idx in filtered_indices:
            path_edges = named_paths[orig_idx - 1].get("path", [])
            if not path_edges:
                continue
            raw_edges = raw_paths[orig_idx - 1].get("path", []) if orig_idx - 1 < len(raw_paths) else []
            evaluations.append(
                CitedPathEvaluation(
                    path_index=orig_idx,
                    path_text=format_chain(path_edges),
                    raw_output=response.text,
                    is_correct=orig_idx in accepted_path_set,
                    tail_entity=tail_from_edges(path_edges),
                    tail_mid=tail_from_edges(raw_edges) if raw_edges else None,
                )
            )

        return CitedPathCheckResult(
            question=question,
            cited_path_indices=filtered_indices,
            check_mode=f"reject-answer-list:{self.reject_policy}",
            raw_output=response.text,
            candidate_answers=candidates,
            rejected_answer_indices=sorted(rejected_set),
            path_evaluations=evaluations,
            accepted_path_indices=accepted_indices,
            predicted_answer_names=answer_names,
            predicted_mids=predicted_mids,
            total_tokens_generated=response.tokens_generated,
            total_elapsed_ms=response.elapsed_ms,
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
