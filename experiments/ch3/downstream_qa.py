"""第三章多检索路径下游 QA 的配置校验、命令构造和结果汇总。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.common import require_fields, resolve_path
from kgqa.core.contracts import MetricSpec, RetrieveResult
from kgqa.experiments import load_confirmed_config, load_json_config
from kgqa.retrieve.eval.path_eval import path_summary
from kgqa.runtime import file_fingerprint


CONDITION_IDS = (
    "no_path",
    "shortest_path",
    "score_beam",
    "terminal_score_beam",
    "fixed",
    "tarrs",
)

_EXPECTED_METHODS = {
    "no_path": {"no_paths": True},
    "shortest_path": {"method": "shortest_path_postprocess"},
    "score_beam": {
        "beam_size": 20, "lambda_val": 0.0, "eta": 0.0, "penalty_mode": "none",
    },
    "terminal_score_beam": {
        "beam_size": 20, "lambda_val": 0.0, "eta": 1.0, "penalty_mode": "none",
    },
    "fixed": {
        "beam_size": 20, "lambda_val": 0.2, "eta": 1.0, "penalty_mode": "fixed",
    },
    "tarrs": {
        "beam_size": 20, "lambda_val": 0.2, "eta": 1.0, "penalty_mode": "adaptive",
    },
}


def _read_jsonl_signature(path: Path, *, require_paths: bool) -> tuple[int, str]:
    """校验 QA 对齐字段并返回行数与可比较的流式签名。"""
    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"检索输入不是合法 JSONL: {path}:{line_number}") from exc
            question = row.get("question_raw", row.get("question")) if isinstance(row, dict) else None
            paths = row.get("paths", row.get("mmr_reason_paths")) if isinstance(row, dict) else None
            if question is None or not isinstance(row.get("golden"), list):
                raise ValueError(f"检索输入缺少 question/question_raw 或 golden: {path}:{line_number}")
            if require_paths and not isinstance(paths, list):
                raise ValueError(f"路径条件输入缺少 paths/mmr_reason_paths 列表: {path}:{line_number}")
            signature = {"question": question, "golden": row["golden"]}
            digest.update(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    if count == 0:
        raise ValueError(f"检索输入为空: {path}")
    return count, digest.hexdigest()


def load_downstream_config(config_path: str | Path, project_dir: Path) -> dict[str, Any]:
    """读取并严格校验六组下游 QA 对照配置。"""
    config_path = Path(config_path).resolve()
    config = load_json_config(config_path)
    require_fields(config, "kind", "dataset", "backbone", "config_id", "profile", "evaluation", "conditions")
    if config["kind"] != "ch3_downstream_qa":
        raise ValueError("不是第三章下游 QA 配置")
    if config["dataset"] != "webqsp" or config["backbone"] != "transfernet":
        raise ValueError("第三章下游 QA 当前只支持 WebQSP / TransferNet")

    profile_path = resolve_path(project_dir, config["profile"])
    profile = load_confirmed_config(profile_path)
    for key in ("dataset", "backbone", "config_id"):
        if config[key] != profile[key]:
            raise ValueError(f"下游 QA 配置与已确认检索配置的 {key} 不一致")

    evaluation = config["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation 必须是对象")
    require_fields(
        evaluation, "model", "format", "path_format", "entity_repr", "max_new_tokens",
        "batch_size", "path_budget",
    )
    if evaluation["path_budget"] != 20:
        raise ValueError("第三章下游 QA 的 path_budget 必须固定为 20")

    conditions = config["conditions"]
    if not isinstance(conditions, list):
        raise ValueError("conditions 必须是列表")
    condition_ids = [item.get("id") for item in conditions if isinstance(item, dict)]
    if len(condition_ids) != len(conditions) or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("conditions 必须由不重复的对象组成")
    if set(condition_ids) != set(CONDITION_IDS):
        raise ValueError("conditions 必须且只能包含六组标准对照")

    for condition in conditions:
        require_fields(condition, "id", "label", "input", "method")
        condition_id = condition["id"]
        expected = _EXPECTED_METHODS[condition_id]
        if condition["method"] != expected:
            raise ValueError(f"条件 {condition_id} 的方法定义不符合规范")
        if condition_id == "no_path" and condition.get("no_paths") is not True:
            raise ValueError("无路径条件必须显式设置 no_paths=true")
        if condition_id != "no_path" and condition.get("no_paths", False):
            raise ValueError(f"路径条件 {condition_id} 不能设置 no_paths")

    adapter = config.get("fixed_pfit_adapter")
    if adapter is not None and not isinstance(adapter, dict):
        raise ValueError("fixed_pfit_adapter 必须为对象或 null")
    return {**config, "_config_path": str(config_path), "_profile_path": str(profile_path), "_profile": profile}


def validate_condition_inputs(config: dict[str, Any], project_dir: Path) -> dict[str, dict[str, Any]]:
    """逐行核对六组题目与 golden，并返回输入指纹和路径。"""
    result: dict[str, dict[str, Any]] = {}
    anchor: tuple[int, str] | None = None
    for condition in config["conditions"]:
        input_path = resolve_path(project_dir, condition["input"])
        if not input_path.is_file():
            raise ValueError(f"找不到条件 {condition['id']} 的检索输入: {input_path}")
        count, signature = _read_jsonl_signature(
            input_path, require_paths=condition["id"] != "no_path"
        )
        if anchor is None:
            anchor = (count, signature)
        elif (count, signature) != anchor:
            raise ValueError(f"条件 {condition['id']} 的题目或 golden 与锚定输入不对齐")
        result[condition["id"]] = {
            "input": file_fingerprint(input_path),
            "samples": count,
            "qa_signature": signature,
        }
    return result


def write_stratified_smoke_inputs(
    config: dict[str, Any], project_dir: Path, output_dir: Path, sample_size: int
) -> dict[str, Path]:
    """按 WebQSP hop 均衡抽取共同样本，避免 ``--limit`` 只取文件开头。"""
    if sample_size < 2:
        raise ValueError("冒烟样本数至少为 2")
    anchor = condition_by_id(config, "no_path")
    anchor_path = resolve_path(project_dir, anchor["input"])
    by_hop: dict[int, list[int]] = {}
    with anchor_path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            hop = row.get("hop")
            if not isinstance(hop, int):
                raise ValueError(f"锚定输入缺少整数 hop: {anchor_path}:{position + 1}")
            by_hop.setdefault(hop, []).append(position)
    hops = sorted(by_hop)
    if len(hops) < 2:
        raise ValueError("冒烟输入至少需要两个 hop 分组")
    per_hop, remainder = divmod(sample_size, len(hops))
    selected: list[int] = []
    for offset, hop in enumerate(hops):
        take = per_hop + (1 if offset < remainder else 0)
        if len(by_hop[hop]) < take:
            raise ValueError(f"hop={hop} 样本不足，无法构造 {sample_size} 条分层冒烟集")
        selected.extend(by_hop[hop][:take])
    selected_positions = set(sorted(selected))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    for condition in config["conditions"]:
        source_path = resolve_path(project_dir, condition["input"])
        target_path = output_dir / f"{condition['id']}.jsonl"
        temporary = target_path.with_suffix(".jsonl.tmp")
        with source_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for position, line in enumerate(source):
                if position in selected_positions:
                    target.write(line)
        os.replace(temporary, target_path)
        generated[condition["id"]] = target_path
    return generated


def condition_by_id(config: dict[str, Any], condition_id: str) -> dict[str, Any]:
    """按稳定 ID 取条件，避免编排层依赖配置数组顺序。"""
    for condition in config["conditions"]:
        if condition["id"] == condition_id:
            return condition
    raise ValueError(f"未知下游 QA 条件: {condition_id}")


def build_eval_command(
    *, dataset: str,
    condition: dict[str, Any],
    evaluation: dict[str, Any],
    input_path: Path,
    exp_dir: Path,
    adapter: Path | None,
    run_dir: Path,
) -> list[str]:
    """构造现役 pfit 评测命令；唯一可变的评测输入是路径 JSONL。"""
    import sys

    command = [
        sys.executable, "-m", "kgqa.pfit.eval", "--dataset", dataset,
        "--input", str(input_path), "--exp_dir", str(exp_dir),
        "--format", evaluation["format"], "--path_format", evaluation["path_format"],
        "--entity_repr", evaluation["entity_repr"], "--model", evaluation["model"],
        "--max_new_tokens", str(evaluation["max_new_tokens"]),
        "--batch_size", str(evaluation["batch_size"]), "--run_dir", str(run_dir),
    ]
    if condition["id"] == "no_path":
        command.append("--no_paths")
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    return command


def resolve_fixed_adapter(config: dict[str, Any], project_dir: Path) -> tuple[str, Path, dict[str, Any]]:
    """校验第二层 adapter 及其训练输入只能来自已确认的 train.jsonl。"""
    adapter_config = config.get("fixed_pfit_adapter")
    if not isinstance(adapter_config, dict):
        raise ValueError("当前配置未定义固定路径监督 adapter，不能运行 fixed_pfit_adapter 层")
    require_fields(adapter_config, "id", "path")
    adapter_path = resolve_path(project_dir, adapter_config["path"])
    expected_root = project_dir / "data/output/kgqa/ch4_pfit" / config["dataset"] / config["config_id"]
    if adapter_path.name != "adapter" or not adapter_path.is_dir() or not adapter_path.is_relative_to(expected_root):
        raise ValueError("固定 adapter 必须位于当前配置的 ch4_pfit 正式实验目录")
    manifest_path = adapter_path.parent / "manifest.json"
    try:
        manifest = load_json_config(manifest_path)
        train_input = Path(manifest["build"]["inputs"]["retrieve"]["path"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"固定 adapter 缺少可追溯的训练清单: {manifest_path}") from exc
    expected_train = Path(config["_profile_path"]).parent / "train.jsonl"
    if train_input != expected_train.resolve():
        raise ValueError("固定 adapter 的训练输入不是当前已确认检索配置的 train.jsonl")
    model_file = adapter_path / "adapter_model.safetensors"
    if not model_file.is_file():
        raise ValueError(f"固定 adapter 缺少权重文件: {model_file}")
    return adapter_config["id"], adapter_path, file_fingerprint(model_file)


def extract_qa_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """提取论文主表使用的 QA 指标，不混入上游 backbone/path 指标。"""
    overall = summary.get("overall", summary)
    return {
        key: overall.get(key)
        for key in (
            "n", "hit1", "hit_any", "macro_p", "macro_r", "macro_f1",
            "micro_p", "micro_r", "micro_f1", "exact_match",
        )
    }


def summarize_input_paths(input_path: Path, *, no_paths: bool) -> dict[str, Any] | None:
    """按本次实际输入 JSONL 重算路径指标，保证与下游 QA 样本严格对齐。"""
    if no_paths:
        return None
    results: list[RetrieveResult] = []
    gold_by_index: dict[int, set[str]] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            paths = row.get("paths", row.get("mmr_reason_paths"))
            if not isinstance(paths, list):
                raise ValueError(f"路径输入缺少 paths/mmr_reason_paths 列表: {input_path}:{line_number}")
            sample_index = int(row.get("sample_index", line_number - 1))
            golden = [str(item) for item in row.get("golden", [])]
            results.append(RetrieveResult(
                question=str(row.get("question_raw", row.get("question", ""))),
                topics=[str(item) for item in row.get("topics", [])],
                hop=int(row.get("hop", 0)), paths=paths, prediction={}, elapsed_ms=0.0,
                sample_index=sample_index, golden=golden,
            ))
            gold_by_index[sample_index] = set(golden)
    if not results:
        raise ValueError(f"路径输入为空: {input_path}")
    return path_summary(results, gold_by_index, MetricSpec(group_by="hop"))["overall"]
