"""检索后端接口 + 参数。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kgqa.core.contracts import RetrieveResult


@dataclass(frozen=True)
class RetrieveParams:
    eta: float = 1.0
    threshold: float = 0.01
    beam_size: int = 50
    lambda_val: float = 0.2
    drop_loopback: bool = True
    alpha_final: float | None = None

    def __post_init__(self):
        # 兼容历史 Python 调用；现役调用统一使用 eta。
        if self.alpha_final is not None:
            object.__setattr__(self, "eta", self.alpha_final)

    def as_kwargs(self) -> dict:
        return {
            "eta": self.eta,
            "threshold": self.threshold,
            "beam_size": self.beam_size,
            "lambda_val": self.lambda_val,
            "drop_loopback": self.drop_loopback,
        }


class RetrieveBackend(ABC):
    @abstractmethod
    def retrieve(self, sample_index: int, **params) -> RetrieveResult: ...

    @abstractmethod
    def retrieve_all(self, *, limit: int = 0, **params) -> list[RetrieveResult]: ...
