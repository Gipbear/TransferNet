"""LLM answer tool for the simple WebQSP QA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from oh_my_agent.common import (
    SYSTEM_PROMPT_V2_NAME,
    build_reasoning_prompt,
    parse_v2_output,
)
from oh_my_agent.llm_server import LLMClient


@dataclass(frozen=True)
class AnswerWithPathsToolResult:
    """Structured output from the LLM answer tool."""

    prompt: str
    raw_output: str
    answer_names: list[str]
    cited_path_indices: list[int]
    format_ok: bool
    used_adapter: bool
    tokens_generated: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerWithPathsTool:
    """Tool wrapper around the local LLM server."""

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        base_url: str = "http://localhost:8788",
        default_use_adapter: bool = True,
        default_max_new_tokens: int = 256,
    ) -> None:
        self.client = client or LLMClient(base_url)
        self.default_use_adapter = default_use_adapter
        self.default_max_new_tokens = default_max_new_tokens

    def __call__(
        self,
        question: str,
        named_paths: list[dict[str, Any]],
        *,
        use_adapter: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> AnswerWithPathsToolResult:
        prompt = build_reasoning_prompt(question, named_paths)
        response = self.client.generate(
            prompt,
            use_adapter=self.default_use_adapter if use_adapter is None else use_adapter,
            max_new_tokens=self.default_max_new_tokens if max_new_tokens is None else max_new_tokens,
            system_prompt=SYSTEM_PROMPT_V2_NAME,
        )
        parsed = parse_v2_output(response.text)
        return AnswerWithPathsToolResult(
            prompt=prompt,
            raw_output=response.text,
            answer_names=parsed.answers,
            cited_path_indices=parsed.cited_indices,
            format_ok=parsed.format_ok,
            used_adapter=response.used_adapter,
            tokens_generated=response.tokens_generated,
            elapsed_ms=response.elapsed_ms,
        )
