"""缓存路径检索服务的 HTTP schema(兼容 legacy oh_my_agent/path_retrieve_server)。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class RetrieveRequest(BaseModel):
    model_config = {"extra": "forbid"}

    question: Optional[str] = Field(None, min_length=1)
    sample_index: Optional[int] = Field(None, ge=0)
    topic_entities: Optional[list[str]] = None
    eta: float = Field(1.0, ge=0.0, le=10.0, description="终点实体分数融合权重 η")
    alpha_final: float | None = Field(None, ge=0.0, le=10.0, description="历史兼容字段")
    threshold: float = Field(0.01, ge=0.0, le=1.0)
    beam_size: int = Field(50, ge=1, le=200)
    lambda_val: float = Field(0.2, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def require_lookup_key(self):
        if self.sample_index is None and not self.question:
            raise ValueError("question or sample_index is required")
        if self.alpha_final is not None:
            self.eta = self.alpha_final
        return self


class RetrieveResponse(BaseModel):
    question: str
    sample_index: int
    topics: list[str]
    hop: int
    mmr_reason_paths: list[dict[str, Any]]
    prediction: dict[str, float]
    elapsed_ms: float
    eta: float = Field(1.0, ge=0.0, le=10.0)
    alpha_final: float | None = Field(None, ge=0.0, le=10.0, description="历史兼容字段")
    threshold: float
    beam_size: int
    lambda_val: float
    cache_path: str
    group_tails: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_legacy_eta(self):
        if self.alpha_final is not None:
            self.eta = self.alpha_final
        else:
            self.alpha_final = self.eta
        return self
