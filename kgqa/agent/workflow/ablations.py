"""按已记录的分支输出执行确定性的模块关闭回放。"""

from copy import deepcopy
from typing import ClassVar


class WorkflowAblations:
    VARIANTS: ClassVar[dict[str, set[str]]] = {"baseline": {"comparative", "superlative", "conjunction", "composition"},
                "without_comparison": {"comparative", "superlative"},
                "without_conjunction": {"conjunction"}, "without_rescue": {"composition"},
                "full": set()}

    @classmethod
    def replay(cls, records: list[dict], variant: str) -> list[dict]:
        if variant not in cls.VARIANTS:
            raise ValueError(f"未知回放组: {variant}")
        rows = deepcopy(records)
        for row in rows:
            if row["route"]["kind"] not in cls.VARIANTS[variant]:
                continue
            row["final"] = deepcopy(row["initial"])
            row["final_paths"] = deepcopy(row.get("initial_paths", row["question"]["paths"]))
            row["source"], row["reason"], row["tool"] = "baseline", "replay_disabled", None
            row["pfit_calls"], row["call_results"] = 1, [deepcopy(row["initial"])]
        return rows
