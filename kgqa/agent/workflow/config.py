"""加载并验证工作流、查询与固定模型的实验配置。"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class WorkflowConfig:
    max_pfit_calls: int = 2
    max_paths: int = 20
    max_input_tokens: int = 1792
    max_new_tokens: int = 256
    type_confidence: float = 0.0
    chain_margin: float = 0.0
    operator_confidence: float = 0.0
    min_chain_score: float = 0.12
    matched_candidate_limit: int = 3
    triple_budget: int = 120
    query_timeout: float = 5.0
    max_query_retries: int = 2
    max_queries: int = 60
    neighbor_k: int = 32
    patterns_per_band: int = 10
    max_topics: int = 3
    step_limit: int = 50
    lambda_val: float = 0.2
    eta: float = 1.0
    penalty_mode: str = "adaptive"
    rescue_scorer: str = "transfernet"
    allow_partial_rescue: bool = False
    allow_template_fallback: bool = False
    allow_cascade_rescue: bool = True
    max_paths_per_relation: int = 0
    comparison_enabled: bool = True
    conjunction_enabled: bool = True
    rescue_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_pfit_calls not in (1, 2):
            raise ValueError("max_pfit_calls 必须为 1 或 2")
        if self.max_paths != 20:
            raise ValueError("正式工作流固定使用最多 20 条路径")
        if self.min_chain_score != 0.12:
            raise ValueError("min_chain_score 固定为训练模板索引的 0.12，修改须重建模板并重新验证")
        if self.max_paths_per_relation < 0:
            raise ValueError("max_paths_per_relation 须非负")
        for name in ("max_input_tokens", "max_new_tokens", "triple_budget", "max_queries", "step_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数")
        if self.query_timeout <= 0 or self.max_query_retries < 0:
            raise ValueError("查询超时须为正数，重试次数须非负")
        if self.rescue_scorer not in ("transfernet", "template"):
            raise ValueError("rescue_scorer 必须显式选择 transfernet 或 template")

    @classmethod
    def from_dict(cls, raw: dict) -> "WorkflowConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"未知工作流配置: {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowConfig":
        return cls.from_dict(json.loads(Path(path).read_text())["workflow"])

    def to_dict(self) -> dict:
        return asdict(self)
