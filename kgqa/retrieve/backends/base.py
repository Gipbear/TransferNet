"""检索后端接口 + 参数。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kgqa.core.contracts import RetrieveResult


@dataclass(frozen=True)
class RetrieveParams:
    eta: float = 1.0
    step_score_mode: str = "joint"
    threshold: float = 0.01
    beam_size: int = 50
    lambda_val: float = 0.2
    penalty_mode: str = "adaptive"
    relation_normalization: str = "global"
    drop_loopback: bool = True

    def as_kwargs(self) -> dict:
        return {
            "eta": self.eta,
            "step_score_mode": self.step_score_mode,
            "threshold": self.threshold,
            "beam_size": self.beam_size,
            "lambda_val": self.lambda_val,
            "penalty_mode": self.penalty_mode,
            "relation_normalization": self.relation_normalization,
            "drop_loopback": self.drop_loopback,
        }


class RetrieveBackend(ABC):
    @abstractmethod
    def retrieve(self, sample_index: int, **params) -> RetrieveResult: ...

    @abstractmethod
    def retrieve_all(self, *, limit: int = 0, **params) -> list[RetrieveResult]: ...
