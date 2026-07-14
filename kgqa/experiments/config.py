"""实验配置和 ``data/output/kgqa`` 目录结构的单一事实来源。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_DATASETS = {"webqsp", "metaqa", "cwq"}
SUPPORTED_BACKBONES = {"transfernet", "rearev"}


def load_json_config(path: str | Path) -> dict[str, Any]:
    """读取版本化 JSON 配置，并给出可操作的中文错误。"""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"找不到实验配置: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"实验配置不是合法 JSON: {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"实验配置必须是 JSON 对象: {config_path}")
    return value


def load_confirmed_config(path: str | Path) -> dict[str, Any]:
    """只允许下游读取人工确认过的第三章检索配置。"""
    config = load_json_config(path)
    if config.get("kind") != "ch3_retrieval_profile":
        raise ValueError(f"不是第三章检索配置: {path}")
    if config.get("status") != "confirmed":
        raise ValueError(
            f"检索配置尚未人工确认，不能作为下游输入: {path} "
            "（将 status 改为 confirmed 前，请填写确认理由。）"
        )
    required = ("dataset", "backbone", "config_id", "topk", "retrieve")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"已确认检索配置缺少字段: {', '.join(missing)}")
    return config


@dataclass(frozen=True)
class ExperimentPaths:
    """统一构造实验产物路径，避免命令脚本自行拼接旧目录。"""

    project_dir: Path

    @property
    def output_root(self) -> Path:
        return self.project_dir / "data" / "output" / "kgqa"

    def score_dir(self, dataset: str, backbone: str, score_id: str) -> Path:
        return self.output_root / "shared" / dataset / "backbones" / backbone / "scores" / score_id

    def ch3_saturation_dir(self, dataset: str, backbone: str, experiment_id: str) -> Path:
        return self.output_root / "ch3_retrieval" / dataset / backbone / "topk_saturation" / experiment_id

    def ch3_profile_dir(self, dataset: str, backbone: str, config_id: str) -> Path:
        return self.output_root / "ch3_retrieval" / dataset / backbone / "confirmed_profiles" / config_id

    def ch3_score_ablation_dir(self, dataset: str, backbone: str, config_id: str) -> Path:
        return self.output_root / "ch3_retrieval" / dataset / backbone / "score_component_ablations" / config_id

    def ch4_run_dir(self, dataset: str, config_id: str, experiment_id: str, seed: int) -> Path:
        return self.output_root / "ch4_pfit" / dataset / config_id / experiment_id / f"seed_{seed}"

    def ch5_dir(self, dataset: str, config_id: str, phase: str) -> Path:
        allowed = {"smoke", "benchmark", "replay_ablations", "sensitivity", "reports"}
        if phase not in allowed:
            raise ValueError(f"未知第五章阶段: {phase}")
        return self.output_root / "ch5_pv_gac" / dataset / config_id / phase
