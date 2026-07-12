"""展示页 HTTP 请求模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveIn(BaseModel):
    sample_index: int = Field(ge=0)
    beam_size: int = Field(50, ge=1, le=200)
    lambda_val: float = Field(0.2, ge=0.0, le=10.0)
    alpha_final: float = Field(1.0, ge=0.0, le=10.0)


class ReplayIn(BaseModel):
    sample_index: int = Field(ge=0)
    score_margin: float = Field(2.0, ge=0.0, le=10.0)
    enable_relation_expansion: bool = True
    expansion_min_answers: int = Field(8, ge=1, le=64)
    expansion_top_groups: int = Field(3, ge=1, le=10)
    eval_view: bool = False
