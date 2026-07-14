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
from kgqa.runtime import configure_runtime, emit_event, update_progress


def _default_config(project_dir: Path, dataset: str, backbone: str) -> Path:
    return project_dir / "experiments" / "configs" / "ch3" / f"{dataset}_{backbone}_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三章检索实验：top-k 饱和性与检索参数扫描")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa", "cwq"], required=True)
    parser.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"])
    parser.add_argument("--config", default="", help="版本化检索配置 JSON；默认按数据集与基础检索模型选择")
    parser.add_argument("--phase", choices=["scores", "scan", "publish", "all"], default="all")
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


def _retrieve_args(config: dict[str, Any], override: dict[str, Any]) -> list[str]:
    params = {**config["retrieve"], **override}
    unsupported = set(params) - {"beam_size", "lambda_val", "threshold", "eta"}
    if unsupported:
        raise ValueError(f"检索配置不接受字段: {', '.join(sorted(unsupported))}")
    return [
        "--beam_size", str(params["beam_size"]),
        "--lambda_val", str(params["lambda_val"]),
        "--threshold", str(params["threshold"]),
        "--eta", str(params["eta"]),
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
    splits = list(config.get("score_source", {}).get("splits", {}))
    scan_splits = _scan_splits(config)
    topk_values = config.get("topk_candidates", [100, 250, 500, 1000])
    scan_items = _parameter_scan_items(config) if args.phase in {"scan", "all"} else []
    task_total = 0
    if args.phase in {"scores", "all"}:
        task_total += len(topk_values) * len(splits) * 2
    if args.phase in {"scan", "all"}:
        task_total += len(scan_items) * len(scan_splits)
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
            configure_runtime(
                argparse.Namespace(run_dir=str(max_score_run_dir), log_level="INFO"),
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
            update_progress(max_score_run_dir, completed=1, total=1, status="completed", phase="得分缓存")
            emit_event(max_score_run_dir, "phase_end", phase="得分缓存")
            task_progress.update(1)

            for topk in topk_values:
                if topk == max_topk:
                    continue
                score_id = f"topk{topk}_{split}"
                score_run_dir = saturation_dir / score_id / "score"
                configure_runtime(
                    argparse.Namespace(run_dir=str(score_run_dir), log_level="INFO"),
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
                _write_console_note(score_run_dir, message)
                update_progress(score_run_dir, completed=1, total=1, status="completed", phase="得分缓存")
                emit_event(score_run_dir, "phase_end", phase="得分缓存", source_topk=max_topk)
                task_progress.update(1)

            for topk in topk_values:
                score_id = f"topk{topk}_{split}"
                evaluation_run_dir = saturation_dir / score_id / "evaluation"
                output = saturation_dir / score_id / f"{split}.jsonl"
                summary = saturation_dir / score_id / f"{split}_summary.json"
                configure_runtime(
                    argparse.Namespace(run_dir=str(evaluation_run_dir), log_level="INFO"),
                    command="第三章 top-k 饱和性检索评测",
                    manifest={"config_path": str(config_path), "topk": topk, "split": split,
                              "cache": str(cache_paths[topk])},
                )
                evaluation_command = [
                    sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                    "--backend", "offline", "--cache", str(cache_paths[topk]),
                    "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                    "--output", str(output), "--summary", str(summary), "--run_dir", str(evaluation_run_dir),
                    *_retrieve_args(config, {}), *_runtime_args(args),
                ]
                run_command(evaluation_command, evaluation_run_dir, dry_run=args.dry_run)
                update_progress(evaluation_run_dir, completed=1, total=1, status="completed", phase="top-k 饱和性评测")
                emit_event(evaluation_run_dir, "phase_end", phase="top-k 饱和性评测")
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
                configure_runtime(
                    argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
                    command="第三章检索参数扫描",
                    manifest={"config_path": str(config_path), "candidate": scan_id, "candidate_label": item["label"], "split": split, "cache": str(cache_path)},
                )
                _write_console_note(
                    run_dir,
                    f"[INFO] 该候选由参数扫描批处理进程执行；完整批处理输出见 "
                    f"{saturation_dir / 'parameter_scan' / 'batch' / 'logs' / 'console.log'}",
                )
                jobs.append({
                    "id": f"{scan_id}_{split}", "run_dir": str(run_dir),
                    "output": str(output), "summary": str(summary),
                    **{**config["retrieve"], **item.get("retrieve", {})},
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
            configure_runtime(
                argparse.Namespace(run_dir=str(batch_dir), log_level="INFO"),
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

    if args.phase == "publish":
        # 人工确认后仅发布已选候选产物；不复制未确认的测试集候选。
        confirmed = load_confirmed_config(config_path)
        candidate_id = confirmed.get("selected_candidate")
        if not candidate_id:
            raise ValueError("已确认检索配置缺少 selected_candidate，无法发布正式检索结果")
        selected_item = None
        run_dir = profile_dir / "publish"
        configure_runtime(
            argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
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
                    configure_runtime(
                        argparse.Namespace(run_dir=str(candidate_run_dir), log_level="INFO"),
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
        update_progress(run_dir, completed=1, total=1, status="completed", phase="发布已确认检索配置")
        emit_event(run_dir, "phase_end", phase="发布已确认检索配置", candidate=candidate_id)
    task_progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
