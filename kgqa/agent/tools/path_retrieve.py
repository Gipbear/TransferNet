"""Tool wrapper for the cached path retrieve server."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kgqa.agent.common import apply_entity_map, load_entity_map, map_entities
from kgqa.agent.specs import get_agent_spec
from kgqa.retrieve.api.client import PathRetrieveClient, PathRetrieveResponse


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
    # 在线 KG 关系组全尾(已按 prediction 过滤),替代离线 sidecar 文件供 expansion 用
    group_tails: dict[str, list[str]] = field(default_factory=dict)

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
        entity_map_path: str | None = None,
        dataset: str = "webqsp",
    ) -> None:
        self.client = client or PathRetrieveClient(base_url)
        if entity_map is not None:
            self.entity_map = entity_map
        elif entity_map_path is not None:
            self.entity_map = load_entity_map(entity_map_path)
        else:
            self.entity_map = get_agent_spec(dataset).load_entity_map()

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
        eta: float = 1.0,
        alpha_final: float | None = None,
        threshold: float = 0.01,
        beam_size: int = 50,
        lambda_val: float = 0.2,
        sample_index: int | None = None,
    ) -> PathRetrieveToolResult:
        if alpha_final is not None:
            eta = alpha_final
        response = self.client.retrieve(
            question,
            sample_index=sample_index,
            topic_entities=[topic_mid] if topic_mid is not None else None,
            eta=eta,
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
            group_tails=response.group_tails,
        )
