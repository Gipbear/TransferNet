"""第三章检索实验编排：top-k 饱和性与人工确认检索配置。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from experiments.common import ROOT, require_fields, resolve_path, run_command
from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.retrieve.cli.dump_scores import materialize_truncated_score_cache
from kgqa.retrieve.engine import validate_penalty_mode, validate_score_scheme
from kgqa.runtime import configure_runtime, emit_event, update_progress


def _default_config(project_dir: Path, dataset: str, backbone: str) -> Path:
    return project_dir / "experiments" / "configs" / "ch3" / f"{dataset}_{backbone}_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三章检索实验：top-k 饱和性与检索参数扫描")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa", "cwq"], required=True)
    parser.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"])
    parser.add_argument("--config", default="", help="版本化检索配置 JSON；默认按数据集与基础检索模型选择")
    parser.add_argument(
        "--phase",
        choices=["scores", "scan", "score_ablation", "penalty_ablation", "shortest_path", "publish", "all"],
        default="all",
    )
    parser.add_argument("--project_dir", default=str(ROOT), help="项目根目录，仅供实验编排定位配置与产物")
    parser.add_argument("--dry_run", action="store_true", help="只展示命令和目标目录，不执行模型")
    parser.add_argument("--no_progress", action="store_true", help="关闭全部 tqdm 进度条")
    parser.add_argument("--progress_interval", type=int, default=50,
                        help="每处理多少条样本更新一次进度文件，默认 50")
    return parser


def _score_source(config: dict[str, Any], split: str) -> dict[str, str]:
    source = config.get("score_source", {})
    split_source = source.get("splits", {}).get(split, {})
    required = {"ckpt": source.get("ckpt", ""), "input_dir": source.get("input_dir", ""), "qa_file": split_source.get("qa_file", "")}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"得分缓存配置缺少 {split} 的字段: {', '.join(missing)}")
    return required


def _retrieve_params(config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    params = {**config["retrieve"], **override}
    params.setdefault("step_score_mode", "joint")
    params.setdefault("penalty_mode", "adaptive")
    unsupported = set(params) - {
        "beam_size", "lambda_val", "threshold", "eta", "step_score_mode", "penalty_mode",
    }
    if unsupported:
        raise ValueError(f"检索配置不接受字段: {', '.join(sorted(unsupported))}")
    required = {"beam_size", "lambda_val", "threshold", "eta", "step_score_mode", "penalty_mode"}
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"检索配置缺少字段: {', '.join(missing)}")
    validate_score_scheme(params["step_score_mode"], params["eta"])
    validate_penalty_mode(params["penalty_mode"])
    return params


def _retrieve_args(config: dict[str, Any], override: dict[str, Any]) -> list[str]:
    params = _retrieve_params(config, override)
    return [
        "--beam_size", str(params["beam_size"]),
        "--lambda_val", str(params["lambda_val"]),
        "--threshold", str(params["threshold"]),
        "--eta", str(params["eta"]),
        "--step_score_mode", params["step_score_mode"],
        "--penalty_mode", params["penalty_mode"],
    ]


def _runtime_args(args: argparse.Namespace) -> list[str]:
    """将编排器的进度控制参数传递给每个现役子命令。"""
    return ["--progress_interval", str(args.progress_interval),
            *( ["--no_progress"] if args.no_progress else [])]


def _scan_splits(config: dict[str, Any]) -> list[str]:
    """参数扫描只在配置指定的数据划分上执行，默认使用测试集。"""
    split = config.get("selection_split", "test")
    available = config.get("score_source", {}).get("splits", {})
    if split not in available:
        raise ValueError(f"参数扫描数据划分不存在: {split}")
    return [split]


def _write_console_note(run_dir: Path, message: str) -> None:
    """为编排器内完成的轻量步骤保留控制台记录。"""
    path = run_dir / "logs" / "console.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _configure_run(args: argparse.Namespace, run_dir: Path, *, command: str, manifest: dict[str, Any]) -> None:
    """实际运行才创建运行时文件，演练只展示命令和目标路径。"""
    if args.dry_run:
        return
    configure_runtime(
        argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
        command=command,
        manifest=manifest,
    )


def _finish_run(args: argparse.Namespace, run_dir: Path, phase: str, **fields: Any) -> None:
    """实际运行结束后原子写入完成状态；演练不触碰已有进度。"""
    if args.dry_run:
        return
    update_progress(run_dir, completed=1, total=1, status="completed", phase=phase)
    emit_event(run_dir, "phase_end", phase=phase, **fields)


def _write_console_note_for_run(args: argparse.Namespace, run_dir: Path, message: str) -> None:
    if not args.dry_run:
        _write_console_note(run_dir, message)


def _parameter_scan_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    """将 beam、λ、eta 三维网格展开为稳定的候选编号。"""
    scan = config.get("parameter_scan")
    if not isinstance(scan, dict):
        raise ValueError("parameter_scan 必须是包含 beam_size、lambda_val 与 eta 列表的对象")
    unsupported = set(scan) - {"beam_size", "lambda_val", "eta"}
    if unsupported:
        raise ValueError(f"parameter_scan 不接受字段: {', '.join(sorted(unsupported))}")
    beam_sizes = scan.get("beam_size")
    lambda_values = scan.get("lambda_val")
    eta_values = scan.get("eta")
    if not isinstance(beam_sizes, list) or not beam_sizes:
        raise ValueError("parameter_scan.beam_size 必须是非空列表")
    if not isinstance(lambda_values, list) or not lambda_values:
        raise ValueError("parameter_scan.lambda_val 必须是非空列表")
    if not isinstance(eta_values, list) or not eta_values:
        raise ValueError("parameter_scan.eta 必须是非空列表")
    if any(not isinstance(value, int) or value <= 0 for value in beam_sizes):
        raise ValueError("parameter_scan.beam_size 必须是正整数列表")
    if any(not isinstance(value, (int, float)) or value < 0 for value in lambda_values):
        raise ValueError("parameter_scan.lambda_val 必须是非负数列表")
    if any(not isinstance(value, (int, float)) or value < 0 for value in eta_values):
        raise ValueError("parameter_scan.eta 必须是非负数列表")
    return [
        {
            "id": (
                f"beam{beam_size}_lambda{float(lambda_val):g}_eta{float(eta):g}"
                .replace(".", "")
            ),
            "label": f"beam={beam_size}，λ={lambda_val}，η={eta}",
            "retrieve": {"beam_size": beam_size, "lambda_val": lambda_val, "eta": eta},
        }
        for beam_size in beam_sizes
        for lambda_val in lambda_values
        for eta in eta_values
    ]


def _score_ablation_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    """读取显式列举的排序分数消融，避免扩展参数扫描的笛卡尔积。"""
    items = config.get("score_component_ablation", [])
    if not isinstance(items, list):
        raise ValueError("score_component_ablation 必须是实验项列表")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("score_component_ablation 的每项必须是对象")
        unsupported = set(item) - {"id", "label", "retrieve"}
        if unsupported:
            raise ValueError(f"score_component_ablation 不接受字段: {', '.join(sorted(unsupported))}")
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("score_component_ablation 每项必须提供非空 id")
        if not isinstance(item.get("label"), str) or not item["label"]:
            raise ValueError("score_component_ablation 每项必须提供非空中文 label")
        override = item.get("retrieve")
        if not isinstance(override, dict):
            raise ValueError("score_component_ablation 每项的 retrieve 必须是对象")
        normalized.append({
            "id": item["id"],
            "label": item["label"],
            "retrieve": _retrieve_params(config, override),
        })
    if len({item["id"] for item in normalized}) != len(normalized):
        raise ValueError("score_component_ablation 存在重复 id")
    return normalized


def _penalty_ablation_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    """读取无惩罚、固定惩罚与自适应惩罚三组显式对照。"""
    items = config.get("penalty_ablation", [])
    if not isinstance(items, list):
        raise ValueError("penalty_ablation 必须是实验项列表")
    if not items:
        return []
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("penalty_ablation 的每项必须是对象")
        unsupported = set(item) - {"id", "label", "retrieve"}
        if unsupported:
            raise ValueError(f"penalty_ablation 不接受字段: {', '.join(sorted(unsupported))}")
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("penalty_ablation 每项必须提供非空 id")
        if not isinstance(item.get("label"), str) or not item["label"]:
            raise ValueError("penalty_ablation 每项必须提供非空中文 label")
        override = item.get("retrieve")
        if not isinstance(override, dict):
            raise ValueError("penalty_ablation 每项的 retrieve 必须是对象")
        normalized.append({
            "id": item["id"],
            "label": item["label"],
            "retrieve": _retrieve_params(config, override),
        })
    if [item["id"] for item in normalized] != ["none", "fixed", "adaptive"]:
        raise ValueError("penalty_ablation 必须按 none、fixed、adaptive 顺序定义三组")
    return normalized


def _shortest_path_params(config: dict[str, Any]) -> dict[str, Any] | None:
    """校验候选答案最短路径后处理基线的显式实验配置。"""
    raw = config.get("shortest_path_baseline")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("shortest_path_baseline 必须是对象")
    allowed = {
        "id", "label", "candidate_topk", "max_paths_per_pair", "path_budget",
        "max_hop_source", "drop_loopback",
    }
    unsupported = set(raw) - allowed
    if unsupported:
        raise ValueError(f"shortest_path_baseline 不接受字段: {', '.join(sorted(unsupported))}")
    required = allowed
    missing = sorted(key for key in required if key not in raw)
    if missing:
        raise ValueError(f"shortest_path_baseline 缺少字段: {', '.join(missing)}")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise ValueError("shortest_path_baseline.id 必须是非空字符串")
    if not isinstance(raw["label"], str) or not raw["label"]:
        raise ValueError("shortest_path_baseline.label 必须是非空中文说明")
    for key in ("candidate_topk", "max_paths_per_pair", "path_budget"):
        if not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError(f"shortest_path_baseline.{key} 必须是正整数")
    if raw["max_hop_source"] != "available_steps":
        raise ValueError("shortest_path_baseline.max_hop_source 当前只支持 available_steps")
    if not isinstance(raw["drop_loopback"], bool):
        raise ValueError("shortest_path_baseline.drop_loopback 必须是布尔值")
    return raw


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config) if args.config else _default_config(project_dir, args.dataset, args.backbone)
    config = load_json_config(config_path)
    require_fields(config, "dataset", "backbone", "config_id", "topk", "retrieve")
    if args.phase != "publish":
        require_fields(config["retrieve"], "beam_size", "lambda_val", "threshold", "eta")
    if config["dataset"] != args.dataset or config["backbone"] != args.backbone:
        raise ValueError("命令行数据集/基础检索模型与配置不一致")
    if args.backbone == "rearev" and args.phase in {"scores", "all"}:
        raise ValueError("ReaRev 当前只消费既有离线得分缓存，不能生成新的 score 缓存")

    paths = ExperimentPaths(project_dir)
    saturation_dir = paths.ch3_saturation_dir(args.dataset, args.backbone, config["config_id"])
    profile_dir = paths.ch3_profile_dir(args.dataset, args.backbone, config["config_id"])
    score_ablation_dir = paths.ch3_score_ablation_dir(args.dataset, args.backbone, config["config_id"])
    penalty_ablation_dir = paths.ch3_penalty_ablation_dir(args.dataset, args.backbone, config["config_id"])
    shortest_path_dir = paths.ch3_shortest_path_dir(args.dataset, args.backbone, config["config_id"])
    splits = list(config.get("score_source", {}).get("splits", {}))
    scan_splits = _scan_splits(config)
    topk_values = config.get("topk_candidates", [100, 250, 500, 1000])
    scan_items = _parameter_scan_items(config) if args.phase in {"scan", "all"} else []
    score_ablation_items = _score_ablation_items(config) if args.phase in {"score_ablation", "all"} else []
    penalty_ablation_items = _penalty_ablation_items(config) if args.phase in {"penalty_ablation", "all"} else []
    shortest_path_params = _shortest_path_params(config)
    if args.phase == "score_ablation" and not score_ablation_items:
        raise ValueError("当前配置未定义 score_component_ablation，无法执行排序分数消融")
    if args.phase == "penalty_ablation" and not penalty_ablation_items:
        raise ValueError("当前配置未定义 penalty_ablation，无法执行冗余惩罚对照")
    if args.phase == "shortest_path" and shortest_path_params is None:
        raise ValueError("当前配置未定义 shortest_path_baseline，无法执行最短路径后处理基线")
    task_total = 0
    if args.phase in {"scores", "all"}:
        task_total += len(topk_values) * len(splits) * 2
    if args.phase in {"scan", "all"}:
        task_total += len(scan_items) * len(scan_splits)
    if args.phase in {"score_ablation", "all"}:
        task_total += len(score_ablation_items) * len(scan_splits)
    if args.phase in {"penalty_ablation", "all"}:
        task_total += len(penalty_ablation_items) * len(scan_splits)
    if args.phase == "shortest_path" or (args.phase == "all" and shortest_path_params is not None):
        task_total += len(scan_splits)
    if args.phase == "publish":
        task_total = len(splits)
    task_progress = tqdm(total=task_total, desc="第三章实验任务", unit="项", dynamic_ncols=True,
                         disable=args.no_progress or args.dry_run)

    if args.phase in {"scores", "all"}:
        max_topk = max(topk_values)
        for split in splits:
            source = _score_source(config, split)
            cache_paths = {
                topk: paths.score_dir(args.dataset, args.backbone, f"topk{topk}_{split}") / f"{split}.pt"
                for topk in topk_values
            }
            max_score_id = f"topk{max_topk}_{split}"
            max_score_run_dir = saturation_dir / max_score_id / "score"
            _configure_run(
                args, max_score_run_dir,
                command="第三章 top-k 饱和性得分缓存",
                manifest={"config_path": str(config_path), "topk": max_topk, "split": split,
                          "output": str(cache_paths[max_topk])},
            )
            command = [
                sys.executable, "-m", "kgqa.retrieve.cli.dump_scores",
                "--dataset", args.dataset, "--ckpt", str(resolve_path(project_dir, source["ckpt"])),
                "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                "--qa_file", str(resolve_path(project_dir, source["qa_file"])),
                "--split", split, "--topk", str(max_topk), "--output", str(cache_paths[max_topk]),
                "--run_dir", str(max_score_run_dir), *_runtime_args(args),
            ]
            run_command(command, max_score_run_dir, dry_run=args.dry_run)
            _finish_run(args, max_score_run_dir, "得分缓存")
            task_progress.update(1)

            for topk in topk_values:
                if topk == max_topk:
                    continue
                score_id = f"topk{topk}_{split}"
                score_run_dir = saturation_dir / score_id / "score"
                _configure_run(
                    args, score_run_dir,
                    command="第三章 top-k 饱和性得分缓存",
                    manifest={"config_path": str(config_path), "topk": topk, "split": split,
                              "output": str(cache_paths[topk]), "source_cache": str(cache_paths[max_topk])},
                )
                if args.dry_run:
                    message = f"[演练] 由 {cache_paths[max_topk]} 裁剪 topk={topk} → {cache_paths[topk]}"
                    print(message)
                else:
                    materialize_truncated_score_cache(cache_paths[max_topk], cache_paths[topk], topk)
                    message = f"[INFO] 由 {cache_paths[max_topk]} 裁剪 topk={topk} → {cache_paths[topk]}"
                _write_console_note_for_run(args, score_run_dir, message)
                _finish_run(args, score_run_dir, "得分缓存", source_topk=max_topk)
                task_progress.update(1)

            for topk in topk_values:
                score_id = f"topk{topk}_{split}"
                evaluation_run_dir = saturation_dir / score_id / "evaluation"
                output = saturation_dir / score_id / f"{split}.jsonl"
                summary = saturation_dir / score_id / f"{split}_summary.json"
                retrieve_params = _retrieve_params(config, {})
                _configure_run(
                    args, evaluation_run_dir,
                    command="第三章 top-k 饱和性检索评测",
                    manifest={"config_path": str(config_path), "topk": topk, "split": split,
                              "cache": str(cache_paths[topk]),
                              "score_scheme": {
                                  "candidate_gate": "intersection",
                                  "step_score_mode": retrieve_params["step_score_mode"],
                                  "terminal_entity_eta": retrieve_params["eta"],
                              }},
                )
                evaluation_command = [
                    sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                    "--backend", "offline", "--cache", str(cache_paths[topk]),
                    "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                    "--output", str(output), "--summary", str(summary), "--run_dir", str(evaluation_run_dir),
                    *_retrieve_args(config, {}), *_runtime_args(args),
                ]
                run_command(evaluation_command, evaluation_run_dir, dry_run=args.dry_run)
                _finish_run(args, evaluation_run_dir, "top-k 饱和性评测")
                task_progress.update(1)

    if args.phase in {"scan", "all"}:
        jobs = []
        for item in scan_items:
            scan_id = item["id"]
            for split in scan_splits:
                score_id = f"topk{config['topk']}_{split}"
                cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
                run_dir = saturation_dir / "parameter_scan" / scan_id / split
                output = profile_dir / "candidates" / scan_id / f"{split}.jsonl"
                summary = profile_dir / "candidates" / scan_id / f"{split}_summary.json"
                source = _score_source(config, split)
                retrieve_params = _retrieve_params(config, item.get("retrieve", {}))
                _configure_run(
                    args, run_dir,
                    command="第三章检索参数扫描",
                    manifest={
                        "config_path": str(config_path), "candidate": scan_id,
                        "candidate_label": item["label"], "split": split, "cache": str(cache_path),
                        "score_scheme": {
                            "candidate_gate": "intersection",
                            "step_score_mode": retrieve_params["step_score_mode"],
                            "terminal_entity_eta": retrieve_params["eta"],
                        },
                    },
                )
                _write_console_note_for_run(
                    args, run_dir,
                    f"[INFO] 该候选由参数扫描批处理进程执行；完整批处理输出见 "
                    f"{saturation_dir / 'parameter_scan' / 'batch' / 'logs' / 'console.log'}",
                )
                jobs.append({
                    "id": f"{scan_id}_{split}", "run_dir": str(run_dir),
                    "output": str(output), "summary": str(summary),
                    **retrieve_params,
                })

        if jobs:
            batch_dir = saturation_dir / "parameter_scan" / "batch"
            jobs_file = saturation_dir / "parameter_scan" / "jobs.json"
            if not args.dry_run:
                jobs_file.parent.mkdir(parents=True, exist_ok=True)
                jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            source = _score_source(config, scan_splits[0])
            score_id = f"topk{config['topk']}_{scan_splits[0]}"
            cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{scan_splits[0]}.pt"
            _configure_run(
                args, batch_dir,
                command="第三章检索参数扫描批处理",
                manifest={"config_path": str(config_path), "jobs_file": str(jobs_file), "jobs": len(jobs),
                          "cache": str(cache_path)},
            )
            command = [
                sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                "--backend", "offline", "--cache", str(cache_path),
                "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                "--jobs_file", str(jobs_file), "--run_dir", str(batch_dir), *_runtime_args(args),
            ]
            run_command(command, batch_dir, dry_run=args.dry_run)
            task_progress.update(len(jobs))

    if args.phase in {"score_ablation", "all"}:
        jobs = []
        for item in score_ablation_items:
            for split in scan_splits:
                score_id = f"topk{config['topk']}_{split}"
                cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
                run_dir = score_ablation_dir / item["id"] / split
                output = score_ablation_dir / item["id"] / f"{split}.jsonl"
                summary = score_ablation_dir / item["id"] / f"{split}_summary.json"
                _configure_run(
                    args, run_dir,
                    command="第三章排序分数消融",
                    manifest={
                        "config_path": str(config_path),
                        "experiment": item["id"],
                        "experiment_label": item["label"],
                        "split": split,
                        "cache": str(cache_path),
                        "score_scheme": {
                            "candidate_gate": "intersection",
                            "step_score_mode": item["retrieve"]["step_score_mode"],
                            "terminal_entity_eta": item["retrieve"]["eta"],
                        },
                    },
                )
                _write_console_note_for_run(
                    args, run_dir,
                    f"[INFO] 该消融由批处理进程执行；完整批处理输出见 "
                    f"{score_ablation_dir / 'batch' / 'logs' / 'console.log'}",
                )
                if args.dry_run:
                    print(f"[演练] 排序分数消融 {item['label']}：{cache_path} → {output}")
                jobs.append({
                    "id": f"{item['id']}_{split}",
                    "run_dir": str(run_dir),
                    "output": str(output),
                    "summary": str(summary),
                    **item["retrieve"],
                })

        if jobs:
            batch_dir = score_ablation_dir / "batch"
            jobs_file = score_ablation_dir / "jobs.json"
            if not args.dry_run:
                jobs_file.parent.mkdir(parents=True, exist_ok=True)
                jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            source = _score_source(config, scan_splits[0])
            cache_path = paths.score_dir(
                args.dataset, args.backbone, f"topk{config['topk']}_{scan_splits[0]}"
            ) / f"{scan_splits[0]}.pt"
            _configure_run(
                args, batch_dir,
                command="第三章排序分数消融批处理",
                manifest={"config_path": str(config_path), "jobs_file": str(jobs_file),
                          "jobs": len(jobs), "cache": str(cache_path),
                          "candidate_gate": "intersection"},
            )
            command = [
                sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                "--backend", "offline", "--cache", str(cache_path),
                "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                "--jobs_file", str(jobs_file), "--run_dir", str(batch_dir), *_runtime_args(args),
            ]
            run_command(command, batch_dir, dry_run=args.dry_run)
            task_progress.update(len(jobs))

    if args.phase in {"penalty_ablation", "all"}:
        jobs = []
        for item in penalty_ablation_items:
            for split in scan_splits:
                score_id = f"topk{config['topk']}_{split}"
                cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
                run_dir = penalty_ablation_dir / item["id"] / split
                output = penalty_ablation_dir / item["id"] / f"{split}.jsonl"
                summary = penalty_ablation_dir / item["id"] / f"{split}_summary.json"
                _configure_run(
                    args, run_dir,
                    command="第三章关系冗余惩罚对照",
                    manifest={
                        "config_path": str(config_path),
                        "experiment": item["id"],
                        "experiment_label": item["label"],
                        "split": split,
                        "cache": str(cache_path),
                        "retrieve": item["retrieve"],
                    },
                )
                _write_console_note_for_run(
                    args, run_dir,
                    f"[INFO] 该对照由批处理进程执行；完整批处理输出见 "
                    f"{penalty_ablation_dir / 'batch' / 'logs' / 'console.log'}",
                )
                if args.dry_run:
                    print(f"[演练] 关系冗余惩罚对照 {item['label']}：{cache_path} → {output}")
                jobs.append({
                    "id": f"{item['id']}_{split}",
                    "run_dir": str(run_dir),
                    "output": str(output),
                    "summary": str(summary),
                    **item["retrieve"],
                })

        if jobs:
            batch_dir = penalty_ablation_dir / "batch"
            jobs_file = penalty_ablation_dir / "jobs.json"
            if not args.dry_run:
                jobs_file.parent.mkdir(parents=True, exist_ok=True)
                jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            source = _score_source(config, scan_splits[0])
            cache_path = paths.score_dir(
                args.dataset, args.backbone, f"topk{config['topk']}_{scan_splits[0]}"
            ) / f"{scan_splits[0]}.pt"
            _configure_run(
                args, batch_dir,
                command="第三章关系冗余惩罚对照批处理",
                manifest={"config_path": str(config_path), "jobs_file": str(jobs_file),
                          "jobs": len(jobs), "cache": str(cache_path)},
            )
            command = [
                sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                "--backend", "offline", "--cache", str(cache_path),
                "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                "--jobs_file", str(jobs_file), "--run_dir", str(batch_dir), *_runtime_args(args),
            ]
            run_command(command, batch_dir, dry_run=args.dry_run)
            task_progress.update(len(jobs))

    if args.phase == "shortest_path" or (args.phase == "all" and shortest_path_params is not None):
        assert shortest_path_params is not None
        for split in scan_splits:
            source = _score_source(config, split)
            score_id = f"topk{config['topk']}_{split}"
            cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
            experiment_dir = shortest_path_dir / shortest_path_params["id"]
            run_dir = experiment_dir / split
            output = experiment_dir / f"{split}.jsonl"
            summary = experiment_dir / f"{split}_summary.json"
            command = [
                sys.executable, "-m", "kgqa.retrieve.cli.shortest_path",
                "--dataset", args.dataset,
                "--backbone", args.backbone,
                "--cache", str(cache_path),
                "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                "--candidate_topk", str(shortest_path_params["candidate_topk"]),
                "--max_paths_per_pair", str(shortest_path_params["max_paths_per_pair"]),
                "--path_budget", str(shortest_path_params["path_budget"]),
                *( ["--drop_loopback"] if shortest_path_params["drop_loopback"] else ["--no-drop_loopback"]),
                "--output", str(output),
                "--summary", str(summary),
                "--run_dir", str(run_dir),
                *_runtime_args(args),
            ]
            run_command(command, run_dir, dry_run=args.dry_run)
            task_progress.update(1)

    if args.phase == "publish":
        # 人工确认后仅发布已选候选产物；不复制未确认的测试集候选。
        confirmed = load_confirmed_config(config_path)
        candidate_id = confirmed.get("selected_candidate")
        if not candidate_id:
            raise ValueError("已确认检索配置缺少 selected_candidate，无法发布正式检索结果")
        selected_item = None
        run_dir = profile_dir / "publish"
        _configure_run(
            args, run_dir,
            command="发布已确认检索配置",
            manifest={"config_path": str(config_path), "selected_candidate": candidate_id},
        )
        for split in confirmed.get("score_source", {}).get("splits", {}):
            source = profile_dir / "candidates" / candidate_id / f"{split}.jsonl"
            target = profile_dir / f"{split}.jsonl"
            if not source.is_file():
                if selected_item is None:
                    selected_item = next(
                        (item for item in _parameter_scan_items(config) if item["id"] == candidate_id),
                        None,
                    )
                if selected_item is None:
                    raise ValueError(f"已确认检索配置引用了未知参数组: {candidate_id}")
                source_config = _score_source(config, split)
                cache_path = paths.score_dir(args.dataset, args.backbone, f"topk{config['topk']}_{split}") / f"{split}.pt"
                candidate_run_dir = saturation_dir / "parameter_scan" / candidate_id / split
                summary = profile_dir / "candidates" / candidate_id / f"{split}_summary.json"
                command = [
                    sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                    "--backend", "offline", "--cache", str(cache_path),
                    "--input_dir", str(resolve_path(project_dir, source_config["input_dir"])),
                    "--output", str(source), "--summary", str(summary), "--run_dir", str(candidate_run_dir),
                    *_retrieve_args(config, selected_item["retrieve"]), *_runtime_args(args),
                ]
                if args.dry_run:
                    print(f"[演练] 为已确认参数组补生成 {split} 检索结果")
                else:
                    _configure_run(
                        args, candidate_run_dir,
                        command="发布前补生成已确认检索结果",
                        manifest={"config_path": str(config_path), "candidate": candidate_id, "split": split},
                    )
                    run_command(command, candidate_run_dir, dry_run=False)
            if args.dry_run:
                print(f"[演练] 发布 {source} → {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            task_progress.update(1)
        if not args.dry_run:
            shutil.copy2(config_path, profile_dir / "confirmed_config.json")
        _finish_run(args, run_dir, "发布已确认检索配置", candidate=candidate_id)
    task_progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
