"""Path retrieval tool for the simple WebQSP QA agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from oh_my_agent.common import apply_entity_map, load_entity_map, map_entities
from oh_my_agent.path_server import PathRetrievalClient, PathRetrievalResponse


DEFAULT_ENTITY_MAP_PATH = "data/resources/WebQSP/fbwq_full/mapped_entities.txt"


@dataclass(frozen=True)
class PathRetrievalToolResult:
    """Structured output from the path retrieval tool."""

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


class PathRetrievalTool:
    """Tool wrapper around PathRetrievalClient with deterministic name mapping."""

    def __init__(
        self,
        *,
        client: PathRetrievalClient | None = None,
        base_url: str = "http://localhost:8787",
        entity_map: dict[str, str] | None = None,
        entity_map_path: str = DEFAULT_ENTITY_MAP_PATH,
    ) -> None:
        self.client = client or PathRetrievalClient(base_url)
        self.entity_map = entity_map or load_entity_map(entity_map_path)

    def _get_raw_paths(self, response: PathRetrievalResponse) -> list[dict[str, Any]]:
        return response.raw_mmr_reason_paths or response.mmr_reason_paths

    def _get_raw_topics(self, response: PathRetrievalResponse, topic_mid: str) -> list[str]:
        return response.raw_topics or [topic_mid]

    def _get_raw_prediction(self, response: PathRetrievalResponse) -> dict[str, float]:
        return response.raw_prediction or response.prediction

    def __call__(
        self,
        question: str,
        topic_mid: str,
        *,
        hop: int | None = None,
        beam_size: int = 20,
        lambda_val: float = 0.2,
        prediction_threshold: float = 0.9,
    ) -> PathRetrievalToolResult:
        response = self.client.retrieve(
            question,
            topic_entities=[topic_mid],
            hop=hop,
            beam_size=beam_size,
            lambda_val=lambda_val,
            prediction_threshold=prediction_threshold,
        )

        raw_paths = self._get_raw_paths(response)
        raw_topics = self._get_raw_topics(response, topic_mid)
        raw_prediction = self._get_raw_prediction(response)

        named_paths = [
            {
                "path": apply_entity_map(path_dict.get("path", []), self.entity_map),
                "log_score": path_dict.get("log_score", 0.0),
            }
            for path_dict in raw_paths
        ]

        return PathRetrievalToolResult(
            question=response.question,
            topic_mid=topic_mid,
            hop=response.hop,
            raw_topics=raw_topics,
            named_topics=map_entities(raw_topics, self.entity_map),
            raw_mmr_reason_paths=raw_paths,
            named_mmr_reason_paths=named_paths,
            raw_prediction=raw_prediction,
            named_prediction={
                self.entity_map.get(entity, entity): score
                for entity, score in raw_prediction.items()
            },
            elapsed_ms=response.elapsed_ms,
        )
