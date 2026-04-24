"""Shared HTTP schemas for the path retrieval service."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    topic_entities: list[str] = Field(..., min_items=1)
    hop: Optional[int] = Field(None, ge=1)
    beam_size: int = Field(20, ge=1, le=200)
    lambda_val: float = Field(0.2, ge=0.0, le=10.0)
    prediction_threshold: float = Field(0.9, ge=0.0, le=1.0)


class RetrieveResponse(BaseModel):
    question: str
    topics: list[str]
    hop: int
    mmr_reason_paths: list[dict[str, Any]]
    prediction: dict[str, float]
    elapsed_ms: float
    raw_topics: list[str] = Field(default_factory=list)
    raw_mmr_reason_paths: list[dict[str, Any]] = Field(default_factory=list)
    raw_prediction: dict[str, float] = Field(default_factory=dict)
