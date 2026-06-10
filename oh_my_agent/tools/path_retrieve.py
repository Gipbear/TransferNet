"""Tool wrapper for the cached path retrieve server."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from oh_my_agent.common import apply_entity_map, load_entity_map, map_entities
from oh_my_agent.path_retrieve_server import PathRetrieveClient, PathRetrieveResponse


DEFAULT_ENTITY_MAP_PATH = "data/resources/WebQSP/fbwq_full/mapped_entities.txt"


@dataclass(frozen=True)
class PathRetrieveToolResult:
    """Structured output from the cached path retrieve tool."""

    question: str
    topic_mid: str
    hop: int
    raw_topics: list[str]
    named_topics: list[str]
    raw_mmr_reason_paths: list[dict[str, Any]]
    named_mmr_reason_paths: list[dict[str, Any]]
    raw_prediction: dict[str, float]
    named_prediction: dict[str, float]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathRetrieveTool:
    """Tool wrapper around PathRetrieveClient with deterministic name mapping."""

    def __init__(
        self,
        *,
        client: PathRetrieveClient | None = None,
        base_url: str = "http://localhost:8789",
        entity_map: dict[str, str] | None = None,
        entity_map_path: str = DEFAULT_ENTITY_MAP_PATH,
    ) -> None:
        self.client = client or PathRetrieveClient(base_url)
        self.entity_map = entity_map or load_entity_map(entity_map_path)

    def _named_paths(self, response: PathRetrieveResponse) -> list[dict[str, Any]]:
        return [
            {
                "path": apply_entity_map(path_dict.get("path", []), self.entity_map),
                "log_score": path_dict.get("log_score", 0.0),
            }
            for path_dict in response.mmr_reason_paths
        ]

    def __call__(
        self,
        question: str | None = None,
        topic_mid: str | None = None,
        *,
        method: str = "tail_blend",
        alpha_final: float = 1.0,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.5,
        sample_index: int | None = None,
    ) -> PathRetrieveToolResult:
        response = self.client.retrieve(
            question,
            sample_index=sample_index,
            topic_entities=[topic_mid] if topic_mid is not None else None,
            method=method,
            alpha_final=alpha_final,
            threshold=threshold,
            beam_size=beam_size,
            lambda_val=lambda_val,
        )
        raw_topics = response.topics
        result_topic_mid = topic_mid or (raw_topics[0] if raw_topics else "")

        return PathRetrieveToolResult(
            question=response.question,
            topic_mid=result_topic_mid,
            hop=response.hop,
            raw_topics=raw_topics,
            named_topics=map_entities(raw_topics, self.entity_map),
            raw_mmr_reason_paths=response.mmr_reason_paths,
            named_mmr_reason_paths=self._named_paths(response),
            raw_prediction=response.prediction,
            named_prediction={
                self.entity_map.get(entity, entity): score
                for entity, score in response.prediction.items()
            },
            elapsed_ms=response.elapsed_ms,
        )
