"""检索后端接口 + 参数。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

from kgqa.types import RetrieveResult


@dataclass(frozen=True)
class RetrieveParams:
    method: str = "tail_blend"
    alpha_final: float = 1.0
    threshold: float = 0.01
    beam_size: int = 50
    lambda_val: float = 0.2
    drop_loopback: bool = True

    def as_kwargs(self) -> dict:
        return asdict(self)


class RetrieveBackend(ABC):
    @abstractmethod
    def retrieve(self, sample_index: int, **params) -> RetrieveResult: ...

    @abstractmethod
    def retrieve_all(self, *, limit: int = 0, **params) -> list[RetrieveResult]: ...
