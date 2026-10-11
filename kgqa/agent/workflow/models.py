"""定义运行时问题、证据、工具决策与完整问答轨迹。"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RuntimeQuestion:
    sample_index: int
    id: str
    question: str
    topic_mids: tuple[str, ...] = ()
    paths: tuple[dict, ...] = ()
    candidates: tuple[dict, ...] = ()

    @classmethod
    def from_record(cls, raw: dict) -> "RuntimeQuestion":
        paths = []
        allowed = ("path", "log_score", "mid_path", "tail_mid", "canonical_tail", "display_to_canonical")
        for item in raw.get("paths", raw.get("mmr_reason_paths", [])):
            paths.append({key: deepcopy(item[key]) for key in allowed if key in item})
        candidates = []
        for item in raw.get("candidates", []):
            candidates.append({key: deepcopy(item[key]) for key in ("name", "mids", "resolution") if key in item})
        return cls(
            int(raw["sample_index"]), str(raw.get("id", raw["sample_index"])), str(raw["question"]),
            tuple(raw.get("topic_mids", [])), tuple(paths), tuple(candidates),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnswerResult:
    answers: tuple[str, ...]
    is_rejection: bool
    format_ok: bool
    cited_indices: tuple[int, ...] = ()
    raw_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    used_path_count: int | None = None
    cache_hit: bool = False
    generation_error: str = ""


@dataclass(frozen=True)
class RouteDecision:
    kind: str
    confidence: float


@dataclass(frozen=True)
class ToolOutcome:
    applied: bool
    reason: str
    answers: tuple[str, ...] | None = None
    paths: tuple[dict, ...] | None = None
    evidence: tuple[dict, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    status: Literal["ok", "empty", "truncated", "failed"]
    rows: tuple[dict, ...] = ()
    query_hash: str = ""
    cache_hit: bool = False
    elapsed_ms: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class WorkflowRecord:
    question: RuntimeQuestion
    route: RouteDecision
    initial: AnswerResult
    final: AnswerResult
    final_paths: tuple[dict, ...]
    source: str
    reason: str
    pfit_calls: int
    elapsed_ms: float
    tool: ToolOutcome | None = None
    initial_paths: tuple[dict, ...] = ()
    call_results: tuple[AnswerResult, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
